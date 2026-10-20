"""Tests for chronological history and future-data isolation."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from market_data import MarketHistory, MarketSnapshot, OHLCVBar


def make_bar(day, symbol="AAA", available_at=None):
    if available_at is None:
        available_at = datetime(2026, 1, day, 21, tzinfo=UTC)

    return OHLCVBar(
        symbol=symbol,
        session_date=date(2026, 1, day),
        available_at=available_at,
        open=Decimal("100.00"),
        high=Decimal("102.00"),
        low=Decimal("99.00"),
        close=Decimal("101.00"),
        volume=1000,
    )


def test_unsorted_input_is_returned_in_session_order():
    first = make_bar(5)
    second = make_bar(6)
    third = make_bar(7)

    store = MarketHistory([third, first, second])
    snapshot = store.snapshot(third.available_at)

    assert snapshot.history("AAA") == (first, second, third)
    assert snapshot.latest("AAA") == third


def test_order_is_deterministic_across_symbols():
    aaa = make_bar(5, "AAA")
    bbb = make_bar(5, "BBB")

    snapshot = MarketHistory([bbb, aaa]).snapshot(aaa.available_at)

    assert snapshot.bars == (aaa, bbb)
    assert snapshot.symbols == ("AAA", "BBB")


def test_duplicate_symbol_session_is_rejected():
    bar = make_bar(5)
    delayed_copy = replace(
        bar,
        available_at=bar.available_at + timedelta(hours=1),
    )

    with pytest.raises(ValueError, match="duplicate"):
        MarketHistory([bar, delayed_copy])


def test_invalid_record_is_rejected():
    with pytest.raises(TypeError):
        MarketHistory([None])


def test_same_session_for_different_symbols_is_valid():
    store = MarketHistory([make_bar(5, "AAA"), make_bar(5, "BBB")])
    snapshot = store.snapshot(make_bar(5).available_at)

    assert len(snapshot.bars) == 2


def test_bar_is_hidden_before_availability():
    bar = make_bar(5)
    store = MarketHistory([bar])

    snapshot = store.snapshot(
        bar.available_at - timedelta(microseconds=1),
    )

    assert snapshot.bars == ()
    assert snapshot.symbols == ()
    assert snapshot.history("AAA") == ()
    assert snapshot.latest("AAA") is None


def test_bar_is_visible_at_exact_availability():
    bar = make_bar(5)
    snapshot = MarketHistory([bar]).snapshot(bar.available_at)

    assert snapshot.bars == (bar,)


def test_future_bars_and_future_only_symbols_are_hidden():
    current = make_bar(5)
    future = make_bar(6)
    future_symbol = make_bar(6, "FUTURE")

    snapshot = MarketHistory(
        [current, future, future_symbol],
    ).snapshot(current.available_at)

    assert snapshot.bars == (current,)
    assert snapshot.symbols == ("AAA",)
    assert snapshot.history("FUTURE") == ()
    assert snapshot.latest("FUTURE") is None
    assert all(
        bar.available_at <= snapshot.cutoff
        for bar in snapshot.bars
    )


def test_availability_controls_visibility_not_session_date():
    delayed = make_bar(
        5,
        available_at=datetime(2026, 1, 6, 1, tzinfo=UTC),
    )
    store = MarketHistory([delayed])

    before = store.snapshot(
        datetime(2026, 1, 5, 23, tzinfo=UTC),
    )
    after = store.snapshot(delayed.available_at)

    assert before.bars == ()
    assert after.bars == (delayed,)
    assert after.latest("AAA").session_date == date(2026, 1, 5)


def test_older_snapshot_does_not_change():
    first = make_bar(5)
    second = make_bar(6)
    store = MarketHistory([first, second])

    earlier = store.snapshot(first.available_at)
    later = store.snapshot(second.available_at)

    assert earlier.bars == (first,)
    assert later.bars == (first, second)


def test_mutating_input_list_does_not_change_store():
    first = make_bar(5)
    records = [first]
    store = MarketHistory(records)

    records.append(make_bar(6))

    snapshot = store.snapshot(make_bar(6).available_at)
    assert snapshot.bars == (first,)


def test_snapshot_is_immutable():
    bar = make_bar(5)
    snapshot = MarketHistory([bar]).snapshot(bar.available_at)

    with pytest.raises(FrozenInstanceError):
        snapshot.cutoff = bar.available_at + timedelta(days=1)

    with pytest.raises(FrozenInstanceError):
        snapshot.bars = ()


def test_direct_snapshot_construction_rejects_future_data():
    bar = make_bar(5)

    with pytest.raises(ValueError, match="unavailable"):
        MarketSnapshot(
            cutoff=bar.available_at - timedelta(seconds=1),
            bars=(bar,),
        )


def test_empty_store_is_valid():
    snapshot = MarketHistory([]).snapshot(
        datetime(2026, 1, 5, 21, tzinfo=UTC),
    )

    assert snapshot.bars == ()
    assert snapshot.latest("AAA") is None


def test_unknown_symbol_returns_empty_history():
    bar = make_bar(5)
    snapshot = MarketHistory([bar]).snapshot(bar.available_at)

    assert snapshot.history("UNKNOWN") == ()
    assert snapshot.latest("UNKNOWN") is None


def test_naive_cutoff_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketHistory([]).snapshot(datetime(2026, 1, 5, 21))


def test_cutoff_is_normalized_to_utc():
    bar = make_bar(5)
    offset = timezone(timedelta(hours=-5))

    snapshot = MarketHistory([bar]).snapshot(
        datetime(2026, 1, 5, 16, tzinfo=offset),
    )

    assert snapshot.cutoff == bar.available_at
    assert snapshot.bars == (bar,)
