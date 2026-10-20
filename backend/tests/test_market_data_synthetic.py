"""Tests for synthetic market data generation."""

from datetime import UTC, date, time
from decimal import Decimal, localcontext
from random import getstate

import pytest

from market_data import generate_synthetic_bars


def generate(**changes):
    arguments = {
        "starting_prices": {
            "AAA": Decimal("100.00"),
            "BBB": Decimal("50.00"),
        },
        "start_date": date(2026, 1, 2),
        "sessions": 5,
        "seed": 42,
    }
    arguments.update(changes)
    return generate_synthetic_bars(**arguments)


def test_identical_seed_produces_identical_data():
    assert generate() == generate()


def test_different_seeds_produce_different_data():
    assert generate(seed=42) != generate(seed=43)


def test_mapping_order_does_not_change_data():
    reversed_prices = {
        "BBB": Decimal("50.00"),
        "AAA": Decimal("100.00"),
    }

    assert generate() == generate(starting_prices=reversed_prices)


def test_generator_does_not_change_global_random_state():
    before = getstate()

    generate()

    assert getstate() == before


def test_generator_does_not_mutate_input():
    prices = {"AAA": Decimal("100.00")}
    ticks = {"AAA": Decimal("0.05")}

    generate(starting_prices=prices, tick_sizes=ticks)

    assert prices == {"AAA": Decimal("100.00")}
    assert ticks == {"AAA": Decimal("0.05")}


def test_generation_is_independent_of_ambient_decimal_precision():
    expected = generate()

    with localcontext() as context:
        context.prec = 6
        actual = generate()

    assert actual == expected


def test_bars_cover_all_symbols_and_sessions():
    bars = generate()

    assert len(bars) == 10
    assert {bar.symbol for bar in bars} == {"AAA", "BBB"}
    assert len({bar.session_date for bar in bars}) == 5


def test_bars_are_chronological_and_skip_weekends():
    bars = generate()
    keys = [(bar.session_date, bar.symbol) for bar in bars]

    assert keys == sorted(keys)
    assert all(bar.session_date.weekday() < 5 for bar in bars)

    dates = sorted({bar.session_date for bar in bars})
    assert dates[:2] == [date(2026, 1, 2), date(2026, 1, 5)]


def test_weekend_start_moves_to_monday():
    bars = generate(start_date=date(2026, 1, 3), sessions=1)

    assert {bar.session_date for bar in bars} == {date(2026, 1, 5)}


def test_generated_bars_have_valid_prices_and_availability():
    for bar in generate():
        assert Decimal(0) < bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.volume >= 0
        assert bar.available_at.date() == bar.session_date
        assert bar.available_at.timetz() == time(
            21, 0, tzinfo=UTC
        )

        for price in (bar.open, bar.high, bar.low, bar.close):
            assert price % bar.tick_size == 0


def test_instrument_specific_ticks():
    bars = generate(
        tick_sizes={
            "AAA": Decimal("0.05"),
            "BBB": Decimal("0.001"),
        },
    )

    for bar in bars:
        expected = Decimal("0.05") if bar.symbol == "AAA" else Decimal("0.001")
        assert bar.tick_size == expected

        for price in (bar.open, bar.high, bar.low, bar.close):
            assert price % expected == 0


def test_minimum_price_remains_positive():
    bars = generate(
        starting_prices={"TINY": Decimal("0.01")},
        sessions=50,
    )

    assert all(bar.low >= Decimal("0.01") for bar in bars)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"starting_prices": {}}, ValueError),
        ({"starting_prices": {"": Decimal("10.00")}}, ValueError),
        ({"starting_prices": {"AAA": 100.0}}, TypeError),
        ({"starting_prices": {"AAA": Decimal(0)}}, ValueError),
        ({"starting_prices": {"AAA": Decimal("100.001")}}, ValueError),
        ({"sessions": 0}, ValueError),
        ({"sessions": -1}, ValueError),
        ({"sessions": True}, TypeError),
        ({"sessions": 1.5}, TypeError),
        ({"seed": True}, TypeError),
        ({"seed": 1.5}, TypeError),
        ({"start_date": "2026-01-02"}, TypeError),
        ({"tick_sizes": {"UNKNOWN": Decimal("0.01")}}, ValueError),
        ({"tick_sizes": {"AAA": Decimal(0)}}, ValueError),
    ],
)
def test_invalid_generator_configuration(changes, error):
    with pytest.raises(error):
        generate(**changes)
