"""Atomic long-only accounting for externally supplied fills.

No matching, price discovery, reservations, calendar, or data retrieval.
Calls must be serialized by the owning simulation; this is not thread-safe.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import ROUND_HALF_UP, Context, Decimal, DecimalException, localcontext
from types import MappingProxyType

from market_data.models import validate_symbol
from simulation.orders import (
    ZERO,
    Fill,
    MarketOrder,
    OrderSide,
    OrderState,
    OrderStatus,
    _cent_amount,
    _exact_context,
    _identifier,
    _quantity,
    _round_cents,
)


class AccountingError(ValueError):
    """An invalid lifecycle action or fill; the ledger is unchanged."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Position:
    symbol: str
    quantity: int
    cost_basis: Decimal

    def __post_init__(self) -> None:
        validate_symbol(self.symbol)
        _quantity(self.quantity)
        object.__setattr__(
            self, "cost_basis", _cent_amount(self.cost_basis, "cost_basis")
        )

    @property
    def average_cost(self) -> Decimal:
        """Fee-inclusive remaining basis per share, not rounded to cents."""
        with localcontext(Context(prec=28, rounding=ROUND_HALF_UP)):
            return self.cost_basis / self.quantity


@dataclass(frozen=True, slots=True, kw_only=True)
class Trade:
    """Accepted fill plus its immutable ledger effects; buys realize zero P&L."""

    fill: Fill
    notional: Decimal
    cash_delta: Decimal
    cost_basis_change: Decimal
    realized_pnl: Decimal
    cash_after: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class PortfolioValuation:
    cash: Decimal
    market_value: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class _Ledger:
    cash: Decimal
    realized_pnl: Decimal
    positions: Mapping[str, Position]
    orders: Mapping[str, OrderState]
    trades: tuple[Trade, ...]
    fill_ids: frozenset[str]
    last_execution_at: datetime | None


class Portfolio:
    """Single-currency portfolio. Available cash equals cash: no reservations."""

    def __init__(self, initial_cash: Decimal) -> None:
        self._initial_cash = _cent_amount(initial_cash, "initial_cash")
        self._ledger = _Ledger(
            self._initial_cash,
            ZERO,
            MappingProxyType({}),
            MappingProxyType({}),
            (),
            frozenset(),
            None,
        )

    @property
    def initial_cash(self) -> Decimal:
        return self._initial_cash

    @property
    def cash(self) -> Decimal:
        return self._ledger.cash

    @property
    def available_cash(self) -> Decimal:
        return self._ledger.cash

    @property
    def realized_pnl(self) -> Decimal:
        return self._ledger.realized_pnl

    @property
    def positions(self) -> Mapping[str, Position]:
        return self._ledger.positions

    @property
    def orders(self) -> Mapping[str, OrderState]:
        return self._ledger.orders

    @property
    def trade_history(self) -> tuple[Trade, ...]:
        return self._ledger.trades

    def get_order(self, order_id: str) -> OrderState:
        _identifier(order_id, "order_id")
        try:
            return self._ledger.orders[order_id]
        except KeyError as exc:
            raise AccountingError(f"unknown order: {order_id}") from exc

    def _set_order(self, state: OrderState) -> OrderState:
        orders = dict(self._ledger.orders)
        orders[state.order.order_id] = state
        self._ledger = replace(self._ledger, orders=MappingProxyType(orders))
        return state

    def submit_order(self, order: MarketOrder) -> OrderState:
        """Register an instruction without reserving funds/shares or executing."""
        if not isinstance(order, MarketOrder):
            raise TypeError("order must be a MarketOrder")
        if order.order_id in self._ledger.orders:
            raise AccountingError(f"duplicate order ID: {order.order_id}")
        return self._set_order(OrderState(order=order))

    def reject_order(self, order_id: str, *, reason: str) -> OrderState:
        """Reject only an unexecuted submitted order; never reverse a fill."""
        state = self.get_order(order_id)
        if state.status is not OrderStatus.SUBMITTED:
            raise AccountingError("only submitted, unfilled orders can be rejected")
        _identifier(reason, "reason")
        return self._set_order(
            replace(
                state,
                status=OrderStatus.REJECTED,
                rejection_reason=reason,
            )
        )

    def cancel_order(self, order_id: str) -> OrderState:
        """Cancel only the unexecuted remainder of an active order."""
        state = self.get_order(order_id)
        if state.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
            raise AccountingError("only active orders can be cancelled")
        return self._set_order(replace(state, status=OrderStatus.CANCELLED))

    def _check_fill(self, fill: Fill) -> OrderState:
        if not isinstance(fill, Fill):
            raise TypeError("fill must be a Fill")
        if fill.fill_id in self._ledger.fill_ids:
            raise AccountingError(f"duplicate fill ID: {fill.fill_id}")
        state = self.get_order(fill.order_id)
        order = state.order
        if state.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
            raise AccountingError("cannot fill a terminal order")
        if fill.symbol != order.symbol or fill.side is not order.side:
            raise AccountingError("fill symbol/side does not match order")
        if fill.quantity > state.remaining_quantity:
            raise AccountingError("fill exceeds remaining order quantity")
        if fill.executed_at < order.submitted_at:
            raise AccountingError("execution precedes submission")
        last = self._ledger.last_execution_at
        if last is not None and fill.executed_at < last:
            raise AccountingError("execution timestamps must be nondecreasing")
        return state

    def apply_fill(self, fill: Fill) -> Trade:
        """Validate and calculate first; publish all ledger changes together.

        Failures do not consume the fill ID, advance execution time, change an
        order status, or mutate cash/positions/history. Stage B may separately
        reject or cancel an order after receiving an AccountingError.
        """
        state = self._check_fill(fill)
        try:
            with localcontext(_exact_context()):
                return self._apply_checked_fill(fill, state)
        except DecimalException as exc:
            raise AccountingError("fill exceeds supported monetary precision") from exc

    def _apply_checked_fill(self, fill: Fill, state: OrderState) -> Trade:
        old = self._ledger
        notional = fill.notional
        if notional <= 0:
            raise AccountingError("fill notional must round to at least one cent")
        position = old.positions.get(fill.symbol)
        quantity = position.quantity if position else 0
        basis = position.cost_basis if position else ZERO
        realized = ZERO

        if fill.side is OrderSide.BUY:
            basis_change = notional + fill.fee
            cash_delta = -basis_change
            new_quantity = quantity + fill.quantity
        else:
            if fill.quantity > quantity:
                raise AccountingError("sale exceeds available holdings")
            if fill.quantity == quantity:
                allocated_basis = basis
            else:
                # Multiply exactly first; only the proportional division and
                # explicit cent allocation may round.
                numerator = basis * fill.quantity
                with localcontext(Context(prec=28, rounding=ROUND_HALF_UP)):
                    allocated_basis = _round_cents(numerator / quantity)
            basis_change = -allocated_basis
            cash_delta = notional - fill.fee
            realized = cash_delta - allocated_basis
            new_quantity = quantity - fill.quantity

        new_cash = old.cash + cash_delta
        if new_cash < 0:
            raise AccountingError("insufficient cash including fees")

        positions = dict(old.positions)
        if new_quantity:
            positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=new_quantity,
                cost_basis=basis + basis_change,
            )
        else:
            positions.pop(fill.symbol, None)

        filled = state.filled_quantity + fill.quantity
        status = (
            OrderStatus.FILLED
            if filled == state.order.quantity
            else OrderStatus.PARTIALLY_FILLED
        )
        orders = dict(old.orders)
        orders[fill.order_id] = replace(state, status=status, filled_quantity=filled)
        trade = Trade(
            fill=fill,
            notional=notional,
            cash_delta=cash_delta,
            cost_basis_change=basis_change,
            realized_pnl=realized,
            cash_after=new_cash,
        )
        updated = _Ledger(
            cash=new_cash,
            realized_pnl=old.realized_pnl + realized,
            positions=MappingProxyType(positions),
            orders=MappingProxyType(orders),
            trades=old.trades + (trade,),
            fill_ids=old.fill_ids | {fill.fill_id},
            last_execution_at=fill.executed_at,
        )
        self._ledger = updated
        return trade

    def value(self, prices: Mapping[str, Decimal]) -> PortfolioValuation:
        """Value only from supplied marks; no fetching or availability checks.

        Marks must be finite, nonnegative Decimals. Zero supports a total loss.
        Marks need not be tick-aligned. All supplied entries are validated;
        valid extra symbols are ignored. Every held symbol requires a mark.
        """
        if not isinstance(prices, Mapping):
            raise TypeError("prices must be a mapping")
        marks = dict(prices)
        for symbol, price in marks.items():
            validate_symbol(symbol)
            if not isinstance(price, Decimal):
                raise TypeError("valuation prices must be Decimals")
            if not price.is_finite() or price < 0:
                raise ValueError("valuation prices must be finite and nonnegative")
        ledger = self._ledger
        missing = ledger.positions.keys() - marks.keys()
        if missing:
            raise ValueError(f"missing prices for held symbols: {sorted(missing)}")
        try:
            with localcontext(_exact_context()):
                market_value = ZERO
                basis = ZERO
                for symbol, position in ledger.positions.items():
                    market_value += _round_cents(marks[symbol] * position.quantity)
                    basis += position.cost_basis
                return PortfolioValuation(
                    cash=ledger.cash,
                    market_value=market_value,
                    equity=ledger.cash + market_value,
                    realized_pnl=ledger.realized_pnl,
                    unrealized_pnl=market_value - basis,
                )
        except DecimalException as exc:
            raise ValueError("valuation exceeds supported monetary precision") from exc
