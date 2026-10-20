"""Accounting examples, ledger invariants, atomicity, and order transitions."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal, localcontext
from random import Random

import pytest

from simulation import (
    AccountingError,
    Fill,
    MarketOrder,
    OrderSide,
    OrderStatus,
    Portfolio,
    Position,
)

D = Decimal
NOW = datetime(2026, 1, 5, 21, tzinfo=UTC)


def submit(
    portfolio, oid="o1", side=OrderSide.BUY, quantity=10, symbol="AAA", **kwargs
):
    order = MarketOrder(
        order_id=oid,
        symbol=symbol,
        side=side,
        quantity=quantity,
        submitted_at=kwargs.get("submitted_at", NOW),
    )
    portfolio.submit_order(order)
    return order


def execution(order, fid="f1", quantity=None, price="10.00", fee="0.00", **kwargs):
    values = {
        "fill_id": fid,
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity if quantity is None else quantity,
        "price": D(price),
        "fee": D(fee),
        "executed_at": order.submitted_at,
    }
    values.update(kwargs)
    return Fill(**values)


def trade(portfolio, oid, side, quantity, price, fee="0.00", symbol="AAA", **kwargs):
    order = submit(portfolio, oid, side, quantity, symbol)
    return portfolio.apply_fill(
        execution(order, fid=f"fill-{oid}", price=price, fee=fee, **kwargs)
    )


def snapshot(portfolio):
    return (
        portfolio.initial_cash,
        portfolio.cash,
        portfolio.available_cash,
        portfolio.realized_pnl,
        dict(portfolio.positions),
        dict(portfolio.orders),
        portfolio.trade_history,
    )


def invariant(portfolio, marks):
    value = portfolio.value(marks)
    assert (
        value.equity - portfolio.initial_cash
        == value.realized_pnl + value.unrealized_pnl
    )
    assert value.equity == value.cash + value.market_value
    assert value.cash == portfolio.available_cash == portfolio.cash
    assert portfolio.cash >= 0
    assert all(
        p.quantity > 0 and p.cost_basis >= 0 for p in portfolio.positions.values()
    )
    assert (
        sum((t.realized_pnl for t in portfolio.trade_history), D(0))
        == portfolio.realized_pnl
    )
    assert (
        portfolio.initial_cash
        + sum((t.cash_delta for t in portfolio.trade_history), D(0))
        == portfolio.cash
    )
    for symbol, position in portfolio.positions.items():
        assert (
            sum(
                (
                    t.cost_basis_change
                    for t in portfolio.trade_history
                    if t.fill.symbol == symbol
                ),
                D(0),
            )
            == position.cost_basis
        )
    for state in portfolio.orders.values():
        assert (
            sum(
                t.fill.quantity
                for t in portfolio.trade_history
                if t.fill.order_id == state.order.order_id
            )
            == state.filled_quantity
        )
    return value


@pytest.mark.parametrize("amount", [D(0), D("1.00"), D("1234.5600")])
def test_initial_cash(amount):
    portfolio = Portfolio(amount)
    assert portfolio.initial_cash == amount
    assert portfolio.cash.as_tuple().exponent == -2
    assert portfolio.positions == {}
    assert portfolio.orders == {}
    assert portfolio.trade_history == ()
    assert invariant(portfolio, {}).equity == amount


@pytest.mark.parametrize(
    "amount,error",
    [
        (0, TypeError),
        (1.0, TypeError),
        (None, TypeError),
        (True, TypeError),
        (D("-0.01"), ValueError),
        (D("0.001"), ValueError),
        (D("NaN"), ValueError),
        (D("Infinity"), ValueError),
        (D("1E100"), ValueError),
    ],
)
def test_invalid_initial_cash(amount, error):
    with pytest.raises(error):
        Portfolio(amount)


def test_submit_does_not_reserve_or_execute():
    portfolio = Portfolio(D("10.00"))
    submit(portfolio, "buy", quantity=1_000_000)
    submit(portfolio, "sell", side=OrderSide.SELL, quantity=1_000_000)
    assert portfolio.cash == portfolio.available_cash == D("10.00")
    assert portfolio.positions == {}
    assert portfolio.trade_history == ()


def test_multiple_buys_partial_sell_rebuy_and_liquidation():
    p = Portfolio(D("10000.00"))
    trade(p, "b1", OrderSide.BUY, 10, "100.00", "1.00")
    assert p.positions["AAA"].cost_basis == D("1001.00")
    assert p.positions["AAA"].average_cost == D("100.10")
    assert invariant(p, {"AAA": D("100.00")}).unrealized_pnl == D("-1.00")
    trade(p, "b2", OrderSide.BUY, 10, "120.00", "1.00")
    assert p.positions["AAA"].cost_basis == D("2202.00")
    assert p.positions["AAA"].average_cost == D("110.10")
    assert p.cash == D("7798.00")
    invariant(p, {"AAA": D("120.00")})
    sold = trade(p, "s1", OrderSide.SELL, 5, "130.00", "1.00")
    assert sold.cash_delta == D("649.00")
    assert sold.cost_basis_change == D("-550.50")
    assert sold.realized_pnl == D("98.50")
    assert p.positions["AAA"].cost_basis == D("1651.50")
    value = invariant(p, {"AAA": D("130.00")})
    assert value.equity == D("10397.00")
    assert value.unrealized_pnl == D("298.50")
    trade(p, "b3", OrderSide.BUY, 5, "90.00", "0.50")
    assert p.positions["AAA"].cost_basis == D("2102.00")
    assert p.positions["AAA"].average_cost == D("105.10")
    invariant(p, {"AAA": D("110.00")})
    trade(p, "s2", OrderSide.SELL, 20, "110.00", "2.00")
    assert p.positions == {}
    assert p.cash == D("10194.50")
    assert invariant(p, {}).realized_pnl == D("194.50")
    assert sum((t.cost_basis_change for t in p.trade_history), D(0)) == 0


def test_partial_buy_fills_have_per_fill_fees_and_order_transitions():
    p = Portfolio(D("1000.00"))
    order = submit(p, quantity=3)
    original_state = p.get_order(order.order_id)
    p.apply_fill(execution(order, fid="a", quantity=1, price="10.00", fee="0.01"))
    assert p.get_order(order.order_id).status is OrderStatus.PARTIALLY_FILLED
    assert p.get_order(order.order_id).remaining_quantity == 2
    invariant(p, {"AAA": D("12.00")})
    p.apply_fill(execution(order, fid="b", quantity=1, price="11.00", fee="0.02"))
    assert p.get_order(order.order_id).status is OrderStatus.PARTIALLY_FILLED
    assert p.get_order(order.order_id).remaining_quantity == 1
    invariant(p, {"AAA": D("12.00")})
    p.apply_fill(execution(order, fid="c", quantity=1, price="12.00", fee="0.03"))
    assert p.get_order(order.order_id).status is OrderStatus.FILLED
    assert p.get_order(order.order_id).remaining_quantity == 0
    assert original_state.status is OrderStatus.SUBMITTED
    assert p.positions["AAA"].cost_basis == D("33.06")
    assert p.cash == D("966.94")
    invariant(p, {"AAA": D("12.00")})


def test_repeated_partial_sales_allocate_all_basis_without_residue():
    p = Portfolio(D("1000.00"))
    trade(p, "buy", OrderSide.BUY, 3, "3.33", "0.01")
    sell = submit(p, "sell", OrderSide.SELL, 3)
    invariant(p, {"AAA": D("4.00")})
    for i, expected in enumerate(["3.33", "3.34", "3.33"]):
        result = p.apply_fill(
            execution(sell, fid=f"s{i}", quantity=1, price="4.00", fee="0.01")
        )
        assert -result.cost_basis_change == D(expected)
        state = p.get_order("sell")
        assert state.filled_quantity == i + 1
        expected_status = OrderStatus.FILLED if i == 2 else OrderStatus.PARTIALLY_FILLED
        assert state.status is expected_status
        invariant(p, {"AAA": D("4.00")})
    assert p.positions == {}
    assert p.realized_pnl == D("1.97")
    assert p.cash == D("1001.97")


def test_half_cent_basis_allocation_rounds_up_and_final_sale_clears_remainder():
    p = Portfolio(D("10.00"))
    trade(p, "buy", OrderSide.BUY, 2, "0.50", "0.01")
    first = trade(p, "s1", OrderSide.SELL, 1, "1.00")
    assert first.cost_basis_change == D("-0.51")
    assert p.positions["AAA"].cost_basis == D("0.50")
    invariant(p, {"AAA": D("1.00")})
    trade(p, "s2", OrderSide.SELL, 1, "1.00")
    assert p.positions == {}
    assert invariant(p, {}).realized_pnl == D("0.99")


def test_settlement_rounds_per_fill_and_accepts_subcent_ticks():
    def run(parts):
        p = Portfolio(D("1.00"))
        order = submit(p, quantity=3)
        for i, quantity in enumerate(parts):
            p.apply_fill(
                execution(
                    order,
                    fid=str(i),
                    quantity=quantity,
                    price="0.005",
                    tick_size=D("0.001"),
                )
            )
            invariant(p, {"AAA": D("0.005")})
        return p

    together, split = run([3]), run([1, 1, 1])
    assert together.positions["AAA"].cost_basis == D("0.02")
    assert split.positions["AAA"].cost_basis == D("0.03")


def test_zero_remaining_basis_is_valid_before_final_share_is_sold():
    p = Portfolio(D("1.00"))
    trade(p, "b", OrderSide.BUY, 3, "0.002", tick_size=D("0.001"))
    for i in range(3):
        trade(p, f"s{i}", OrderSide.SELL, 1, "0.01")
        invariant(p, {"AAA": D("0.01")})
        if i == 1:
            assert p.positions["AAA"].cost_basis == 0
    assert p.positions == {}
    assert p.realized_pnl == D("0.02")


@pytest.mark.parametrize(
    "price,expected",
    [("0.004", "0.00"), ("0.005", "0.01"), ("0.006", "0.01"), ("20.00", "20.00")],
)
def test_mark_rounding_and_price_changes_do_not_modify_ledger(price, expected):
    p = Portfolio(D("100.00"))
    trade(p, "b", OrderSide.BUY, 1, "10.00", "0.01")
    before = snapshot(p)
    assert invariant(p, {"AAA": D(price)}).market_value == D(expected)
    assert snapshot(p) == before


def test_marks_are_explicit_and_required_on_every_call():
    p = Portfolio(D("100.00"))
    trade(p, "a", OrderSide.BUY, 1, "10.00")
    trade(p, "b", OrderSide.BUY, 2, "20.00", symbol="BBB")
    first = p.value({"AAA": D("11.00"), "BBB": D("21.00"), "EXTRA": D(999)})
    second = p.value({"AAA": D(0), "BBB": D(0)})
    assert first.market_value == D("53.00")
    assert second.market_value == 0
    assert invariant(p, {"AAA": D(0), "BBB": D(0)}).unrealized_pnl == D("-50.00")
    with pytest.raises(ValueError, match="missing prices"):
        p.value({"AAA": D("11.00")})
    with pytest.raises(ValueError, match="missing prices"):
        p.value({})


@pytest.mark.parametrize(
    "marks,error",
    [
        (None, TypeError),
        ({"AAA": 1.0}, TypeError),
        ({"AAA": None}, TypeError),
        ({"AAA": D(-1)}, ValueError),
        ({"AAA": D("NaN")}, ValueError),
        ({"AAA": D("Infinity")}, ValueError),
        ({"": D(1)}, ValueError),
    ],
)
def test_invalid_marks_are_rejected_without_changes(marks, error):
    p = Portfolio(D("100.00"))
    before = snapshot(p)
    with pytest.raises(error):
        p.value(marks)
    assert snapshot(p) == before


@pytest.mark.parametrize(
    "case",
    [
        "insufficient",
        "buy_fee",
        "oversell",
        "overfill",
        "wrong_symbol",
        "wrong_side",
        "unknown",
        "before_submission",
        "backward_time",
        "zero_notional",
        "precision",
        "duplicate_same",
        "duplicate_other",
        "cancelled",
        "rejected",
        "filled",
        "sell_fee",
    ],
)
def test_rejected_fill_is_atomic(case):
    p = Portfolio(D("100.00"))
    trade(p, "seed", OrderSide.BUY, 2, "10.00")
    side = OrderSide.SELL if case in ("oversell", "sell_fee") else OrderSide.BUY
    order = submit(p, "target", side, quantity=10)
    candidate = execution(order, fid="candidate", quantity=1)
    if case == "insufficient":
        candidate = replace(candidate, quantity=9)
    elif case == "buy_fee":
        candidate = replace(candidate, quantity=8, fee=D("0.01"))
    elif case == "oversell":
        candidate = replace(candidate, quantity=3)
    elif case == "overfill":
        candidate = replace(candidate, quantity=11, price=D("0.01"))
    elif case == "wrong_symbol":
        candidate = replace(candidate, symbol="BBB")
    elif case == "wrong_side":
        candidate = replace(candidate, side=OrderSide.SELL)
    elif case == "unknown":
        candidate = replace(candidate, order_id="missing")
    elif case == "before_submission":
        candidate = replace(candidate, executed_at=NOW - timedelta(seconds=1))
    elif case == "backward_time":
        p.apply_fill(
            execution(
                order, fid="later", quantity=1, executed_at=NOW + timedelta(seconds=1)
            )
        )
    elif case == "zero_notional":
        candidate = replace(candidate, price=D("0.001"), tick_size=D("0.001"))
    elif case == "precision":
        candidate = replace(
            candidate, price=D("99999999999999999999999999.99"), quantity=9
        )
    elif case == "duplicate_same":
        p.apply_fill(candidate)
    elif case == "duplicate_other":
        candidate = replace(candidate, fill_id="fill-seed")
    elif case == "cancelled":
        p.cancel_order(order.order_id)
    elif case == "rejected":
        p.reject_order(order.order_id, reason="Risk check")
    elif case == "filled":
        p.apply_fill(execution(order, fid="complete", price="0.01"))
    elif case == "sell_fee":
        candidate = replace(candidate, fee=D("90.01"))
    before = snapshot(p)
    ledger_before = p._ledger
    with pytest.raises(AccountingError):
        p.apply_fill(candidate)
    assert snapshot(p) == before
    # Checks internal timestamp and duplicate-ID registry too.
    assert p._ledger is ledger_before
    invariant(p, {"AAA": D("10.00")})


def test_failed_fill_does_not_consume_id_or_advance_time():
    p = Portfolio(D("1.00"))
    order = submit(p, quantity=1)
    failed = execution(order, executed_at=NOW + timedelta(days=1))
    with pytest.raises(AccountingError):
        p.apply_fill(failed)
    accepted = replace(failed, price=D("1.00"), executed_at=NOW)
    p.apply_fill(accepted)
    assert p.cash == 0
    assert p.trade_history[0].fill.fill_id == failed.fill_id


@pytest.mark.parametrize("partial", [False, True])
def test_cancellation_and_rejection_transitions(partial):
    p = Portfolio(D("1000.00"))
    order = submit(p, quantity=3)
    if partial:
        p.apply_fill(execution(order, quantity=1))
        before = snapshot(p)
        with pytest.raises(AccountingError):
            p.reject_order(order.order_id, reason="Too late")
        assert snapshot(p) == before
    cash, history = p.cash, p.trade_history
    state = p.cancel_order(order.order_id)
    assert state.status is OrderStatus.CANCELLED
    assert state.remaining_quantity == (2 if partial else 3)
    assert p.cash == cash and p.trade_history == history
    before = snapshot(p)
    for action in [
        lambda: p.cancel_order(order.order_id),
        lambda: p.reject_order(order.order_id, reason="No"),
        lambda: p.apply_fill(execution(order, fid="later", quantity=1)),
    ]:
        with pytest.raises(AccountingError):
            action()
        assert snapshot(p) == before


@pytest.mark.parametrize("terminal", ["filled", "rejected"])
def test_terminal_orders_cannot_transition_or_execute(terminal):
    p = Portfolio(D("1000.00"))
    order = submit(p, quantity=1)
    if terminal == "filled":
        p.apply_fill(execution(order))
    else:
        state = p.reject_order(order.order_id, reason="Risk rule")
        assert state.rejection_reason == "Risk rule"
    before = snapshot(p)
    for action in [
        lambda: p.cancel_order(order.order_id),
        lambda: p.reject_order(order.order_id, reason="Again"),
        lambda: p.apply_fill(execution(order, fid="again")),
    ]:
        with pytest.raises(AccountingError):
            action()
        assert snapshot(p) == before


def test_partial_fill_failures_can_be_followed_by_valid_completion():
    p = Portfolio(D("100.00"))
    order = submit(p, quantity=3)
    p.apply_fill(execution(order, quantity=1))
    for quantity, price in [(3, "1.00"), (2, "100.00")]:
        before = snapshot(p)
        with pytest.raises(AccountingError):
            p.apply_fill(execution(order, fid="retry", quantity=quantity, price=price))
        assert snapshot(p) == before
    p.apply_fill(execution(order, fid="retry", quantity=2))
    assert p.get_order(order.order_id).status is OrderStatus.FILLED
    invariant(p, {"AAA": D("10.00")})


def test_order_lookup_registration_and_bad_reasons_are_atomic():
    p = Portfolio(D("100.00"))
    order = submit(p)
    before = snapshot(p)
    for action, error in [
        (lambda: p.submit_order(order), AccountingError),
        (lambda: p.submit_order(None), TypeError),
        (lambda: p.apply_fill(None), TypeError),
        (lambda: p.get_order("unknown"), AccountingError),
        (lambda: p.cancel_order("unknown"), AccountingError),
        (lambda: p.reject_order("unknown", reason="no"), AccountingError),
        (lambda: p.reject_order(order.order_id, reason=" "), ValueError),
        (lambda: p.reject_order(order.order_id, reason=None), TypeError),
    ]:
        with pytest.raises(error):
            action()
        assert snapshot(p) == before


def test_read_only_views_are_stable_snapshots():
    p = Portfolio(D("100.00"))
    order = submit(p)
    old_positions, old_orders, old_history = p.positions, p.orders, p.trade_history
    result = p.apply_fill(execution(order, quantity=1))
    assert old_positions == {}
    assert old_orders[order.order_id].status is OrderStatus.SUBMITTED
    assert old_history == ()
    with pytest.raises(TypeError):
        p.positions["AAA"] = None
    with pytest.raises(TypeError):
        p.orders[order.order_id] = None
    for obj, attribute, value in [
        (p.positions["AAA"], "quantity", 0),
        (result, "cash_after", D(0)),
        (p.value({"AAA": D(1)}), "equity", D(0)),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attribute, value)
    with pytest.raises(AttributeError):
        p.initial_cash = D(0)


def test_portfolios_are_independent_and_explicit_histories_repeat():
    a, b = Portfolio(D("100.00")), Portfolio(D("100.00"))
    order = submit(a, quantity=2)
    b.submit_order(order)
    f = execution(order)
    a.apply_fill(f)
    assert b.positions == {} and b.cash == D("100.00")
    b.apply_fill(f)
    assert snapshot(a) == snapshot(b)


def test_exact_cash_including_fee_can_be_spent():
    p = Portfolio(D("10.01"))
    trade(p, "b", OrderSide.BUY, 1, "10.00", "0.01")
    assert p.available_cash == 0
    invariant(p, {"AAA": D("10.00")})


def test_sale_fee_may_exceed_proceeds_if_cash_covers_it():
    p = Portfolio(D("10.00"))
    trade(p, "b", OrderSide.BUY, 1, "1.00")
    trade(p, "s", OrderSide.SELL, 1, "1.00", "2.00")
    assert p.cash == D("8.00")
    assert invariant(p, {}).realized_pnl == D("-2.00")


def test_accounting_is_independent_of_ambient_precision_and_rounding():
    def run():
        p = Portfolio(D("100.00"))
        trade(p, "b", OrderSide.BUY, 3, "3.33", "0.01")
        average = p.positions["AAA"].average_cost
        trade(p, "s", OrderSide.SELL, 1, "4.00", "0.01")
        return snapshot(p), average, p.value({"AAA": D("4.005")})

    expected = run()
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        assert run() == expected


def test_late_precision_failure_is_atomic_and_failed_id_is_reusable():
    # Large input is individually representable but adding one cent would lose
    # precision. Fail after position/order candidates have been calculated.
    p = Portfolio(D("99999999999999999999999999.99"))
    trade(p, "b", OrderSide.BUY, 1, "0.01")
    order = submit(p, "s", OrderSide.SELL, 1)
    candidate = execution(order, price="0.03")
    before = snapshot(p)
    with pytest.raises(AccountingError, match="precision"):
        p.apply_fill(candidate)
    assert snapshot(p) == before
    p.apply_fill(replace(candidate, price=D("0.01")))
    assert p.cash == p.initial_cash


def test_valuation_precision_failure_is_read_only():
    p = Portfolio(D("100.00"))
    trade(p, "b", OrderSide.BUY, 2, "1.00")
    before = snapshot(p)
    with pytest.raises(ValueError, match="precision"):
        p.value({"AAA": D("99999999999999999999999999.99")})
    assert snapshot(p) == before


def test_position_average_retains_precision():
    position = Position(symbol="AAA", quantity=3, cost_basis=D("10.00"))
    assert position.average_cost == D("3.333333333333333333333333333")
    with pytest.raises(ValueError):
        Position(symbol="AAA", quantity=0, cost_basis=D(0))
    with pytest.raises(ValueError):
        Position(symbol="AAA", quantity=1, cost_basis=D("0.001"))


def test_seeded_sequences_reconcile_after_every_fill_and_full_liquidation():
    rng = Random(2026)
    p = Portfolio(D("100000.00"))
    # Independent integer-cent cash oracle; costs and quantities vary across
    # multiple symbols, partial fills, fees, repeated sales, and re-entry.
    cash_cents = 10_000_000
    for i in range(100):
        symbol = rng.choice(["AAA", "BBB", "CCC"])
        held = p.positions[symbol].quantity if symbol in p.positions else 0
        side = OrderSide.SELL if held and rng.randrange(2) else OrderSide.BUY
        quantity = (
            rng.randint(1, min(held, 6))
            if side is OrderSide.SELL
            else rng.randint(1, 6)
        )
        order = submit(p, f"o{i}", side, quantity, symbol)
        remaining = quantity
        j = 0
        while remaining:
            part = rng.randint(1, remaining)
            price_cents, fee_cents = rng.randint(1, 2000), rng.randint(0, 9)
            p.apply_fill(
                execution(
                    order,
                    fid=f"f{i}-{j}",
                    quantity=part,
                    price=str(D(price_cents) / 100),
                    fee=str(D(fee_cents) / 100),
                )
            )
            signed = (
                -price_cents * part if side is OrderSide.BUY else price_cents * part
            )
            cash_cents += signed - fee_cents
            assert p.cash == D(cash_cents) / 100
            invariant(p, {s: D("12.345") for s in p.positions})
            remaining -= part
            j += 1
    for symbol, position in list(p.positions.items()):
        trade(
            p,
            f"liquidate-{symbol}",
            OrderSide.SELL,
            position.quantity,
            "10.00",
            "0.01",
            symbol,
        )
        cash_cents += position.quantity * 1000 - 1
        invariant(p, {s: D("10.00") for s in p.positions})
    assert p.positions == {}
    assert p.cash == D(cash_cents) / 100
    assert p.realized_pnl == p.cash - p.initial_cash
    assert sum((t.cost_basis_change for t in p.trade_history), D(0)) == 0
