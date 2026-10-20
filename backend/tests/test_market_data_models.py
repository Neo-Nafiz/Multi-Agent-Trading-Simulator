"""Tests for daily bar validation."""

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from market_data import OHLCVBar


def make_bar(**changes):
    values = {
        "symbol": "AAA",
        "session_date": date(2026, 1, 5),
        "available_at": datetime(2026, 1, 5, 21, tzinfo=UTC),
        "open": Decimal("100.00"),
        "high": Decimal("102.00"),
        "low": Decimal("99.00"),
        "close": Decimal("101.00"),
        "volume": 1000,
    }
    values.update(changes)
    return OHLCVBar(**values)


def test_valid_bar_uses_default_tick():
    bar = make_bar()

    assert bar.tick_size == Decimal("0.01")
    assert bar.session_date == date(2026, 1, 5)
    assert bar.close == Decimal("101.00")


def test_zero_volume_is_valid():
    assert make_bar(volume=0).volume == 0


def test_flat_bar_is_valid():
    bar = make_bar(
        open=Decimal("100.00"),
        high=Decimal("100.00"),
        low=Decimal("100.00"),
        close=Decimal("100.00"),
    )

    assert bar.low == bar.high


def test_timestamp_is_normalized_to_utc():
    eastern_offset = timezone(timedelta(hours=-5))

    bar = make_bar(
        available_at=datetime(2026, 1, 5, 16, tzinfo=eastern_offset),
    )

    assert bar.available_at == datetime(
        2026, 1, 5, 21, tzinfo=UTC
    )
    assert bar.available_at.tzinfo is UTC


def test_session_date_is_distinct_from_availability():
    bar = make_bar(
        available_at=datetime(2026, 1, 6, 1, tzinfo=UTC),
    )

    assert bar.session_date == date(2026, 1, 5)
    assert bar.available_at.date() == date(2026, 1, 6)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"symbol": ""}, ValueError),
        ({"symbol": " AAA"}, ValueError),
        ({"symbol": None}, TypeError),
        ({"session_date": "2026-01-05"}, TypeError),
        ({"session_date": datetime(2026, 1, 5)}, TypeError),
        ({"session_date": None}, TypeError),
        ({"available_at": datetime(2026, 1, 5, 21)}, ValueError),
        ({"available_at": None}, TypeError),
        ({"volume": -1}, ValueError),
        ({"volume": 1.5}, TypeError),
        ({"volume": True}, TypeError),
        ({"volume": None}, TypeError),
        ({"open": 100.0}, TypeError),
        ({"open": None}, TypeError),
        ({"open": Decimal(0)}, ValueError),
        ({"open": Decimal(-1)}, ValueError),
        ({"open": Decimal("NaN")}, ValueError),
        ({"open": Decimal("Infinity")}, ValueError),
        ({"open": Decimal("100.001")}, ValueError),
        ({"open": Decimal("103.00")}, ValueError),
        ({"close": Decimal("98.00")}, ValueError),
        ({"low": Decimal("103.00")}, ValueError),
        ({"high": Decimal("98.00")}, ValueError),
        ({"tick_size": Decimal(0)}, ValueError),
        ({"tick_size": Decimal("-0.01")}, ValueError),
        ({"tick_size": Decimal("NaN")}, ValueError),
        ({"tick_size": 0.01}, TypeError),
    ],
)
def test_invalid_bar_is_rejected(changes, error):
    with pytest.raises(error):
        make_bar(**changes)


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
def test_every_price_field_rejects_missing_values(field):
    with pytest.raises(TypeError):
        make_bar(**{field: None})


def test_custom_tick_accepts_exact_multiples():
    bar = make_bar(
        tick_size=Decimal("0.05"),
        close=Decimal("101.15"),
    )

    assert bar.close == Decimal("101.15")


def test_custom_tick_rejects_nonmultiple():
    with pytest.raises(ValueError, match="exact multiple"):
        make_bar(
            tick_size=Decimal("0.05"),
            close=Decimal("101.16"),
        )


def test_subcent_tick_is_supported():
    bar = make_bar(
        tick_size=Decimal("0.001"),
        close=Decimal("101.001"),
    )

    assert bar.close == Decimal("101.001")


def test_bar_is_immutable():
    bar = make_bar()

    with pytest.raises(FrozenInstanceError):
        bar.close = Decimal("100.00")
