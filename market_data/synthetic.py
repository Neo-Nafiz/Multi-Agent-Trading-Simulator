"""Deterministic synthetic daily OHLCV generation."""

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
from random import Random

from market_data.models import (
    DEFAULT_PRICE_TICK,
    OHLCVBar,
    validate_price,
    validate_session_date,
    validate_symbol,
)

_BASIS_POINTS = Decimal(10000)
_SYNTHETIC_CLOSE = time(21, 0, tzinfo=UTC)


def _round_to_tick(value: Decimal, tick_size: Decimal) -> Decimal:
    """Round to the nearest tick, with half ticks rounded upward."""
    tick_count = (value / tick_size).to_integral_value(
        rounding=ROUND_HALF_UP,
    )
    return max(tick_size, tick_count * tick_size)


def _session_dates(start_date: date, sessions: int) -> tuple[date, ...]:
    """Generate weekdays; exchange holidays are intentionally not modeled."""
    result: list[date] = []
    current = start_date

    while len(result) < sessions:
        if current.weekday() < 5:
            result.append(current)

        if len(result) < sessions:
            current += timedelta(days=1)

    return tuple(result)


def generate_synthetic_bars(
    starting_prices: Mapping[str, Decimal],
    *,
    start_date: date,
    sessions: int,
    seed: int,
    tick_sizes: Mapping[str, Decimal] | None = None,
) -> tuple[OHLCVBar, ...]:
    """Generate reproducible bars for one or more symbols.

    starting_prices are reference prices before the first generated session.

    Synthetic sessions are weekdays with complete bars available at
    21:00 UTC. This is not a real exchange calendar.

    Prices use bounded random percentage changes. This generator supports
    simulator testing; it is not a calibrated financial market model.
    """
    validate_session_date(start_date)

    if type(sessions) is not int:
        raise TypeError("sessions must be an integer")
    if sessions <= 0:
        raise ValueError("sessions must be greater than zero")

    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    prices = dict(starting_prices)
    if not prices:
        raise ValueError("starting_prices cannot be empty")

    for symbol in prices:
        validate_symbol(symbol)

    ticks = dict(tick_sizes) if tick_sizes is not None else {}

    if set(ticks) - set(prices):
        raise ValueError("tick_sizes contains an unknown symbol")

    for symbol, price in prices.items():
        ticks.setdefault(symbol, DEFAULT_PRICE_TICK)
        validate_price(price, ticks[symbol])

    rng = Random(seed)
    symbols = tuple(sorted(prices))
    dates = _session_dates(start_date, sessions)
    bars: list[OHLCVBar] = []

    with localcontext(Context(prec=28, rounding=ROUND_HALF_UP)):
        for session_date in dates:
            available_at = datetime.combine(
                session_date,
                _SYNTHETIC_CLOSE,
            )

            for symbol in symbols:
                tick = ticks[symbol]
                previous_close = prices[symbol]

                # Overnight move: -1% to +1%.
                open_price = _round_to_tick(
                    previous_close
                    * (
                        Decimal(1)
                        + Decimal(rng.randint(-100, 100)) / _BASIS_POINTS
                    ),
                    tick,
                )

                # Open-to-close move: -2% to +2%.
                close_price = _round_to_tick(
                    open_price
                    * (
                        Decimal(1)
                        + Decimal(rng.randint(-200, 200)) / _BASIS_POINTS
                    ),
                    tick,
                )

                # Extend the high and low by up to 1%.
                high_price = _round_to_tick(
                    max(open_price, close_price)
                    * (
                        Decimal(1)
                        + Decimal(rng.randint(0, 100)) / _BASIS_POINTS
                    ),
                    tick,
                )

                low_price = _round_to_tick(
                    min(open_price, close_price)
                    * (
                        Decimal(1)
                        - Decimal(rng.randint(0, 100)) / _BASIS_POINTS
                    ),
                    tick,
                )

                bar = OHLCVBar(
                    symbol=symbol,
                    session_date=session_date,
                    available_at=available_at,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=rng.randint(1000, 1_000_000),
                    tick_size=tick,
                )

                bars.append(bar)
                prices[symbol] = bar.close

    return tuple(bars)
