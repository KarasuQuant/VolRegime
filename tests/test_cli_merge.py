from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from volregime.cli import _merge_settings_with_overrides


@dataclass
class FakeSettings:
    symbol: str = "SPY"
    data_dir: Path = Path("data")
    start: date | None = None
    end: date | None = None
    source: str = "yfinance"
    interval: str = "1d"
    cache_mode: str = "use"


def test_merge_uses_settings_when_no_overrides():
    s = FakeSettings(
        symbol="SPY",
        data_dir=Path("data"),
        start=None,
        end=None,
        source="yfinance",
        interval="1d",
        cache_mode="use",
    )

    kwargs = _merge_settings_with_overrides(
        s,
        symbol=None,
        data_dir=None,
        start=None,
        end=None,
        source=None,
        interval=None,
        refresh=False,
    )

    assert kwargs["symbol"] == "SPY"
    assert kwargs["data_dir"] == Path("data")
    assert kwargs["start"] is None
    assert kwargs["end"] is None
    assert kwargs["source"] == "yfinance"
    assert kwargs["interval"] == "1d"
    assert kwargs["cache_mode"] == "use"


def test_merge_overrides_take_precedence_and_refresh_wins():
    s = FakeSettings(
        symbol="SPY",
        data_dir=Path("data"),
        start=date(2000, 1, 1),
        end=date(2020, 1, 1),
        source="yfinance",
        interval="1d",
        cache_mode="use",
    )

    kwargs = _merge_settings_with_overrides(
        s,
        symbol="QQQ",
        data_dir=Path("/tmp/vol"),
        start=date(2018, 1, 1),
        end=None,
        source="yfinance",
        interval="1d",
        refresh=True,
    )

    assert kwargs["symbol"] == "QQQ"
    assert kwargs["data_dir"] == Path("/tmp/vol")
    assert kwargs["start"] == date(2018, 1, 1)
    assert kwargs["end"] == date(
        2020, 1, 1
    )  # inherited from settings because override is None
    assert kwargs["cache_mode"] == "refresh"  # refresh flag wins
