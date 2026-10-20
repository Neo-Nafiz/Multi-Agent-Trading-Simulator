"""Validation, immutability, IDs, and lifecycle snapshot consistency."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from uuid import UUID

import pytest

from simulation import Fill, MarketOrder, OrderSide, OrderState, OrderStatus, OrderType

D = Decimal
NOW = datetime(2026, 1, 5, 21, tzinfo=UTC)


def order(**changes):
    values = {
        "symbol": "AAA",
        "side": OrderSide.BUY,
        "quantity": 10,
        "submitted_at": NOW,
        "order_id": "order-1",
    }
    values.update(changes)
    return MarketOrder(**values)


def fill(**changes):
    values = {
        "order_id": "order-1",
        "fill_id": "fill-1",
        "symbol": "AAA",
        "side": OrderSide.BUY,
        "quantity": 2,
        "price": D("10.00"),
        "executed_at": NOW,
        "fee": D("0.01"),
    }
    values.update(changes)
    return Fill(**values)


def test_explicit_ids_make_models_reproducible():
    assert order() == order()
    assert fill() == fill()
    assert order().order_type is OrderType.MARKET


def test_explicit_ids_do_not_call_uuid_generator(monkeypatch):
    def unexpected_uuid():
        raise AssertionError("explicit IDs must not generate a UUID")

    monkeypatch.setattr("simulation.orders.uuid4", unexpected_uuid)
    assert order().order_id == "order-1"
    assert fill().fill_id == "fill-1"


def test_defaults_generate_distinct_uuid_strings():
    values = {
        "symbol": "AAA",
        "side": OrderSide.BUY,
        "quantity": 1,
        "submitted_at": NOW,
    }
    first, second = MarketOrder(**values), MarketOrder(**values)
    assert UUID(first.order_id)
    assert first.order_id != second.order_id
    values = {
        "order_id": first.order_id,
        "symbol": "AAA",
        "side": OrderSide.BUY,
        "quantity": 1,
        "price": D("1.00"),
        "executed_at": NOW,
    }
    a, b = Fill(**values), Fill(**values)
    assert UUID(a.fill_id)
    assert a.fill_id != b.fill_id


def test_timestamps_normalized_and_fees_canonicalized():
    stamp = datetime(2026, 1, 5, 16, tzinfo=timezone(timedelta(hours=-5)))
    assert order(submitted_at=stamp).submitted_at == NOW
    execution = fill(executed_at=stamp, fee=D("1.000"))
    assert execution.executed_at == NOW
    assert execution.executed_at.tzinfo is UTC
    assert execution.fee.as_tuple().exponent == -2


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"symbol": ""}, ValueError),
        ({"symbol": " AAA"}, ValueError),
        ({"symbol": None}, TypeError),
        ({"side": "buy"}, TypeError),
        ({"side": None}, TypeError),
        ({"quantity": 0}, ValueError),
        ({"quantity": -1}, ValueError),
        ({"quantity": True}, TypeError),
        ({"quantity": 1.0}, TypeError),
        ({"quantity": D(1)}, TypeError),
        ({"quantity": None}, TypeError),
        ({"order_id": ""}, ValueError),
        ({"order_id": "id "}, ValueError),
        ({"order_id": 1}, TypeError),
        ({"submitted_at": datetime(2026, 1, 5)}, ValueError),
        ({"submitted_at": None}, TypeError),
    ],
)
def test_invalid_orders(changes, error):
    with pytest.raises(error):
        order(**changes)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"fill_id": ""}, ValueError),
        ({"fill_id": " id"}, ValueError),
        ({"fill_id": None}, TypeError),
        ({"order_id": ""}, ValueError),
        ({"order_id": None}, TypeError),
        ({"symbol": ""}, ValueError),
        ({"side": "sell"}, TypeError),
        ({"quantity": 0}, ValueError),
        ({"quantity": -1}, ValueError),
        ({"quantity": True}, TypeError),
        ({"quantity": 1.5}, TypeError),
        ({"price": 10.0}, TypeError),
        ({"price": None}, TypeError),
        ({"price": D(0)}, ValueError),
        ({"price": D(-1)}, ValueError),
        ({"price": D("NaN")}, ValueError),
        ({"price": D("Infinity")}, ValueError),
        ({"price": D("10.001")}, ValueError),
        ({"price": D("10.01"), "tick_size": D("0.05")}, ValueError),
        ({"tick_size": D(0)}, ValueError),
        ({"tick_size": 0.01}, TypeError),
        ({"fee": 0.0}, TypeError),
        ({"fee": None}, TypeError),
        ({"fee": D("-0.01")}, ValueError),
        ({"fee": D("0.001")}, ValueError),
        ({"fee": D("NaN")}, ValueError),
        ({"fee": D("Infinity")}, ValueError),
        ({"executed_at": datetime(2026, 1, 5)}, ValueError),
        ({"executed_at": None}, TypeError),
    ],
)
def test_invalid_fills(changes, error):
    with pytest.raises(error):
        fill(**changes)


def test_tick_and_notional_rounding_are_separate():
    assert fill(price=D("10.15"), tick_size=D("0.05")).price == D("10.15")
    tiny = fill(price=D("0.005"), tick_size=D("0.001"), quantity=3)
    assert tiny.notional == D("0.02")
    with localcontext() as context:
        context.prec = 2
        assert tiny.notional == D("0.02")


@pytest.mark.parametrize(
    "status,filled",
    [
        (OrderStatus.SUBMITTED, 0),
        (OrderStatus.PARTIALLY_FILLED, 1),
        (OrderStatus.PARTIALLY_FILLED, 9),
        (OrderStatus.FILLED, 10),
        (OrderStatus.CANCELLED, 0),
        (OrderStatus.CANCELLED, 9),
    ],
)
def test_valid_lifecycle_snapshots(status, filled):
    state = OrderState(order=order(), status=status, filled_quantity=filled)
    assert state.remaining_quantity == 10 - filled


def test_rejected_state_requires_a_reason():
    state = OrderState(
        order=order(), status=OrderStatus.REJECTED, rejection_reason="Risk limit"
    )
    assert state.remaining_quantity == 10
    assert state.rejection_reason == "Risk limit"


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"order": None}, TypeError),
        ({"status": "submitted"}, TypeError),
        ({"filled_quantity": True}, TypeError),
        ({"filled_quantity": 1.0}, TypeError),
        ({"filled_quantity": -1}, ValueError),
        ({"filled_quantity": 11}, ValueError),
        ({"filled_quantity": 1}, ValueError),
        ({"status": OrderStatus.PARTIALLY_FILLED}, ValueError),
        ({"status": OrderStatus.PARTIALLY_FILLED, "filled_quantity": 10}, ValueError),
        ({"status": OrderStatus.FILLED, "filled_quantity": 9}, ValueError),
        ({"status": OrderStatus.CANCELLED, "filled_quantity": 10}, ValueError),
        ({"status": OrderStatus.REJECTED}, TypeError),
        ({"status": OrderStatus.REJECTED, "rejection_reason": " "}, ValueError),
        (
            {
                "status": OrderStatus.REJECTED,
                "rejection_reason": "no",
                "filled_quantity": 1,
            },
            ValueError,
        ),
        ({"rejection_reason": "no"}, ValueError),
    ],
)
def test_inconsistent_states_are_rejected(changes, error):
    values = {"order": order()}
    values.update(changes)
    with pytest.raises(error):
        OrderState(**values)


def test_models_are_frozen_and_original_instruction_is_preserved():
    instruction = order()
    state = OrderState(order=instruction)
    changed = replace(state, status=OrderStatus.PARTIALLY_FILLED, filled_quantity=2)
    assert changed.order is instruction
    assert state.status is OrderStatus.SUBMITTED
    for obj, attribute, value in [
        (instruction, "quantity", 1),
        (state, "filled_quantity", 1),
        (fill(), "price", D("1.00")),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, attribute, value)
