"""Immutable submitted orders, lifecycle snapshots, and supplied executions."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import (
    ROUND_HALF_UP,
    Context,
    Decimal,
    DecimalException,
    Inexact,
    localcontext,
)
from enum import StrEnum
from uuid import uuid4

from market_data.models import (
    DEFAULT_PRICE_TICK,
    normalize_timestamp,
    validate_price,
    validate_symbol,
)

CENT = Decimal("0.01")
ZERO = Decimal("0.00")


def _exact_context() -> Context:
    """Reject loss of significant digits in ledger arithmetic."""
    context = Context(prec=28, rounding=ROUND_HALF_UP)
    context.traps[Inexact] = True
    return context


def _round_cents(value: Decimal) -> Decimal:
    """Explicit settlement/valuation rounding boundary."""
    with localcontext(Context(prec=28, rounding=ROUND_HALF_UP)):
        return value.quantize(CENT)


def _cent_amount(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        with localcontext(_exact_context()):
            return value.quantize(CENT)
    except DecimalException as exc:
        raise ValueError(f"{name} must be exact cents within precision 28") from exc


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty without outer whitespace")


def _quantity(value: int) -> None:
    if type(value) is not int:
        raise TypeError("quantity must be a whole-share integer")
    if value <= 0:
        raise ValueError("quantity must be positive")


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"


class OrderStatus(StrEnum):
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketOrder:
    """An instruction, not an execution. Supply IDs for deterministic runs."""

    symbol: str
    side: OrderSide
    quantity: int
    submitted_at: datetime
    order_id: str = field(default_factory=lambda: str(uuid4()))
    order_type: OrderType = field(default=OrderType.MARKET, init=False)

    def __post_init__(self) -> None:
        _identifier(self.order_id, "order_id")
        validate_symbol(self.symbol)
        if not isinstance(self.side, OrderSide):
            raise TypeError("side must be an OrderSide")
        _quantity(self.quantity)
        object.__setattr__(self, "submitted_at", normalize_timestamp(self.submitted_at))


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderState:
    """A validated immutable snapshot of an order's lifecycle."""

    order: MarketOrder
    status: OrderStatus = OrderStatus.SUBMITTED
    filled_quantity: int = 0
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.order, MarketOrder):
            raise TypeError("order must be a MarketOrder")
        if not isinstance(self.status, OrderStatus):
            raise TypeError("status must be an OrderStatus")
        if type(self.filled_quantity) is not int:
            raise TypeError("filled_quantity must be an integer")
        filled = self.filled_quantity
        total = self.order.quantity
        if not 0 <= filled <= total:
            raise ValueError("filled_quantity must be within the order quantity")
        if self.status in (OrderStatus.SUBMITTED, OrderStatus.REJECTED) and filled:
            raise ValueError("submitted/rejected orders cannot have executions")
        if self.status is OrderStatus.PARTIALLY_FILLED and not 0 < filled < total:
            raise ValueError("partial status requires an incomplete execution")
        if self.status is OrderStatus.FILLED and filled != total:
            raise ValueError("filled status requires the entire order quantity")
        if self.status is OrderStatus.CANCELLED and filled == total:
            raise ValueError("a fully filled order cannot be cancelled")
        if self.status is OrderStatus.REJECTED:
            _identifier(self.rejection_reason, "rejection_reason")
        elif self.rejection_reason is not None:
            raise ValueError("only rejected orders can have a rejection reason")

    @property
    def remaining_quantity(self) -> int:
        """Unexecuted quantity; cancelled/rejected remainders are not executable."""
        return self.order.quantity - self.filled_quantity


@dataclass(frozen=True, slots=True, kw_only=True)
class Fill:
    """An externally supplied execution, with an explicit fee for this fill."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    executed_at: datetime
    fee: Decimal = ZERO
    tick_size: Decimal = DEFAULT_PRICE_TICK
    fill_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        _identifier(self.fill_id, "fill_id")
        _identifier(self.order_id, "order_id")
        validate_symbol(self.symbol)
        if not isinstance(self.side, OrderSide):
            raise TypeError("side must be an OrderSide")
        _quantity(self.quantity)
        validate_price(self.price, self.tick_size)
        object.__setattr__(self, "fee", _cent_amount(self.fee, "fee"))
        object.__setattr__(self, "executed_at", normalize_timestamp(self.executed_at))

    @property
    def notional(self) -> Decimal:
        """Round price times shares once, to cents; precision failures raise."""
        with localcontext(_exact_context()):
            raw = self.price * self.quantity
        return _round_cents(raw)
