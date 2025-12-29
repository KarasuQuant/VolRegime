from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer

from volregime.config import DataSettings
from volregime.data import get_ohlcv, read_cache_metadata

app = typer.Typer(
    help="Volatility regime project CLI.",
)


def _parse_iso_date(value: str | None, *, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise typer.BadParameter(f"{name} must be YYYY-MM-DD") from e


def _merge_settings_with_overrides(
    settings: DataSettings,
    *,
    symbol: Optional[str],
    data_dir: Optional[Path],
    start: Optional[date],
    end: Optional[date],
    source: Optional[str],
    interval: Optional[str],
    refresh: bool,
) -> dict:
    """
    Merge DataSettings defaults with explicit CLI overrides.
    Return kwargs compatible with load_or_fetch_ohlcv().
    """
    merged_symbol = symbol if symbol is not None else settings.symbol
    merged_data_dir = data_dir if data_dir is not None else settings.data_dir
    merged_start = start if start is not None else settings.start
    merged_end = end if end is not None else settings.end
    merged_source = source if source is not None else settings.source
    merged_interval = interval if interval is not None else settings.interval

    # If user passes --refresh, it should win. Otherwise use the configured cache_mode.
    merged_cache_mode = "refresh" if refresh else settings.cache_mode

    return {
        "symbol": merged_symbol,
        "data_dir": merged_data_dir,
        "source": merged_source,
        "interval": merged_interval,
        "cache_mode": merged_cache_mode,
        "start": merged_start,
        "end": merged_end,
    }


@app.command()
def health():
    print("OK")


@app.command()
def fetch(
    symbol: Optional[str] = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Ticker symbol (override config). Example: SPY",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Base data directory (override config). Example: ./data",
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (inclusive) YYYY-MM-DD (override config).",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (inclusive) YYYY-MM-DD (override config).",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Data source (override config). Currently: yfinance",
    ),
    interval: Optional[str] = typer.Option(
        None,
        "--interval",
        help="Interval (override config). Currently: 1d",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Force refetch and overwrite cache regardless of config cache_mode.",
    ),
    show_tail: int = typer.Option(
        5,
        "--show-tail",
        min=0,
        help="Print last N rows after loading (0 disables).",
    ),
    show_meta: bool = typer.Option(
        True,
        "--meta/--no-meta",
        help="Print cache metadata if available.",
    ),
) -> None:
    """
    Fetch and cache OHLCV data (full history cached; optional start/end filtering on return).
    Uses DataSettings defaults unless overridden by CLI flags.
    """
    start_d = _parse_iso_date(start, name="--start")
    end_d = _parse_iso_date(end, name="--end")

    settings = DataSettings()
    kwargs = _merge_settings_with_overrides(
        settings,
        symbol=symbol,
        data_dir=data_dir,
        start=start_d,
        end=end_d,
        source=source,
        interval=interval,
        refresh=refresh,
    )

    df = get_ohlcv(**kwargs)

    typer.echo(
        f"Loaded {len(df)} rows for {kwargs['symbol'].upper()} "
        f"(source={kwargs['source']}, interval={kwargs['interval']}, cache_mode={kwargs['cache_mode']})."
    )

    if show_tail > 0:
        typer.echo(df.tail(show_tail).to_string(index=False))

    if show_meta:
        meta = read_cache_metadata(
            symbol=kwargs["symbol"],
            data_dir=kwargs["data_dir"],
            source=kwargs["source"],
            interval=kwargs["interval"],
        )
        if meta is None:
            typer.echo("No cache metadata found yet.")
        else:
            typer.echo("Cache metadata:")
            # pretty print
            import json

            typer.echo(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
