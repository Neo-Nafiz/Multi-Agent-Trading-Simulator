"""Historical market data and bounded observation snapshots."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

from market_data.models import (
    OHLCVBar,
    normalize_timestamp,
    validate_symbol,
)


def _ordered_bars(bars: Iterable[OHLCVBar]) -> tuple[OHLCVBar, ...]:
    """Validate records and sort by session date, then symbol."""
    records = tuple(bars)
    seen: set[tuple[str, date]] = set()

    for bar in records:
        if not isinstance(bar, OHLCVBar):
            raise TypeError("all observations must be OHLCVBar instances")

        key = (bar.symbol, bar.session_date)
        if key in seen:
            raise ValueError(
                f"duplicate bar for {bar.symbol} on {bar.session_date}"
            )
        seen.add(key)

    return tuple(
        sorted(
            records,
            key=lambda bar: (bar.session_date, bar.symbol),
        )
    )


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Immutable observations available at a fixed simulation timestamp.

    The snapshot holds no reference to the complete MarketHistory.
    It cannot advance itself or request a later cutoff.
    """

    cutoff: datetime
    bars: tuple[OHLCVBar, ...]

    def __post_init__(self) -> None:
        cutoff = normalize_timestamp(self.cutoff)
        bars = _ordered_bars(self.bars)

        if any(bar.available_at > cutoff for bar in bars):
            raise ValueError("snapshot cannot contain unavailable bars")

        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "bars", bars)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols present in the observations available so far."""
        return tuple(sorted({bar.symbol for bar in self.bars}))

    def history(self, symbol: str) -> tuple[OHLCVBar, ...]:
        """Return observed bars for a symbol in session-date order."""
        validate_symbol(symbol)
        return tuple(bar for bar in self.bars if bar.symbol == symbol)

    def latest(self, symbol: str) -> OHLCVBar | None:
        """Return the latest observed session, or None if none exists."""
        observations = self.history(symbol)
        return observations[-1] if observations else None


class MarketHistory:
    """Complete data store owned by simulation infrastructure.

    Do not give this object to an agent. The simulation selects the cutoff
    and passes only the resulting MarketSnapshot to the agent.
    """

    def __init__(self, bars: Iterable[OHLCVBar]) -> None:
        self._bars = _ordered_bars(bars)

    def snapshot(self, cutoff: datetime) -> MarketSnapshot:
        """Include only complete bars available at or before cutoff."""
        cutoff = normalize_timestamp(cutoff)

        available_bars = tuple(
            bar
            for bar in self._bars
            if bar.available_at <= cutoff
        )

        return MarketSnapshot(
            cutoff=cutoff,
            bars=available_bars,
        )
