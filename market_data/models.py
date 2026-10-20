"""Validated daily market data models."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Context, Decimal, InvalidOperation, localcontext

DEFAULT_PRICE_TICK = Decimal("0.01")


def validate_symbol(symbol: str) -> None:
    """Require a nonempty symbol without surrounding whitespace."""
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")

    if not symbol or symbol != symbol.strip():
        raise ValueError("symbol must be nonempty and have no outer whitespace")


def normalize_timestamp(value: datetime) -> datetime:
    """Validate an aware timestamp and normalize it to UTC."""
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")

    return value.astimezone(UTC)


def validate_session_date(value: date) -> None:
    """Require a date, excluding datetime objects."""
    if type(value) is not date:
        raise TypeError("session_date must be a date, not a datetime")


def _validate_positive_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")

    if not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def validate_price(
    value: Decimal,
    tick_size: Decimal = DEFAULT_PRICE_TICK,
) -> None:
    """Require a positive price that is an exact multiple of its tick."""
    _validate_positive_decimal(tick_size, "tick_size")
    _validate_positive_decimal(value, "price")

    try:
        with localcontext(Context(prec=28)):
            remainder = value % tick_size
    except InvalidOperation as exc:
        raise ValueError("price exceeds supported Decimal precision") from exc

    if remainder != 0:
        raise ValueError("price must be an exact multiple of tick_size")


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    """A completed daily bar.

    session_date:
        Trading-calendar label for the session.

    available_at:
        Earliest timestamp at which the complete bar may be observed.
        Stored in UTC.

    tick_size:
        Smallest allowed price increment for this instrument.

    No exchange calendar is inferred from session_date. The data provider
    is responsible for assigning the correct availability timestamp.
    """

    symbol: str
    session_date: date
    available_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    tick_size: Decimal = DEFAULT_PRICE_TICK

    def __post_init__(self) -> None:
        validate_symbol(self.symbol)
        validate_session_date(self.session_date)

        object.__setattr__(
            self,
            "available_at",
            normalize_timestamp(self.available_at),
        )

        for price in (self.open, self.high, self.low, self.close):
            validate_price(price, self.tick_size)

        if self.low > self.high:
            raise ValueError("low cannot exceed high")

        if not self.low <= self.open <= self.high:
            raise ValueError("open must be between low and high")

        if not self.low <= self.close <= self.high:
            raise ValueError("close must be between low and high")

        # bool is an int subclass, so use an exact type check.
        if type(self.volume) is not int:
            raise TypeError("volume must be an integer")

        if self.volume < 0:
            raise ValueError("volume cannot be negative")
