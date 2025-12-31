from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import typer

from volregime.config import DataSettings
from volregime.data import get_ohlcv, read_cache_metadata
from volregime.dataset import build_dataset_frame, DatasetMeta
from volregime.features import FeatureConfig

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
    symbol: Optional[str] = None,
    data_dir: Optional[Path] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    source: Optional[str] = None,
    interval: Optional[str] = None,
    price_col: Optional[str] = None,
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
    merged_price_col = price_col if price_col is not None else settings.price_col_for_returns

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
        "price_col": merged_price_col,
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


def _parse_int_tuple(value: Optional[str], default: tuple[int, ...]) -> tuple[int, ...]:
    """
    Parse comma-separated ints: "5,10,20" -> (5,10,20).
    If value is None, return default.
    """
    if value is None:
        return default
    s = value.strip()
    if not s:
        return default
    parts = [p.strip() for p in s.split(",")]
    out: list[int] = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    if not out:
        return default
    return tuple(out)


def _default_dataset_out_path(*, data_dir: Path, symbol: str, horizon: int) -> Path:
    return data_dir / "processed" / f"{symbol.upper()}_dataset_h{horizon}.parquet"


def _write_meta_json(meta: DatasetMeta, out_parquet: Path) -> Path:
    meta_path = out_parquet.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta.__dict__, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


@app.command()
def build_dataset(
    # Data selection (defaults from settings unless overridden)
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Ticker symbol (override config)."),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Base data directory (override config)."),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (inclusive) YYYY-MM-DD (override config)."),
    end: Optional[str] = typer.Option(None, "--end", help="End date (inclusive) YYYY-MM-DD (override config)."),
    source: Optional[str] = typer.Option(None, "--source", help="Data source (override config). Currently: yfinance"),
    interval: Optional[str] = typer.Option(None, "--interval", help="Interval (override config). Currently: 1d"),
    refresh: bool = typer.Option(False, "--refresh", help="Force refetch and overwrite cache."),

    # Dataset build params
    horizon: int = typer.Option(5, "--horizon", "-h", min=1, help="Future volatility horizon in sessions."),
    price_col: Optional[str] = typer.Option(
        None,
        "--price-col",
        help="Price column for returns: adj_close or close (override config).",
    ),
    vol_windows: Optional[str] = typer.Option(
        None,
        "--vol-windows",
        help="Comma-separated vol windows, e.g. 5,10,20,60 (override defaults).",
    ),
    mom_windows: Optional[str] = typer.Option(
        None,
        "--mom-windows",
        help="Comma-separated momentum windows, e.g. 5,10,20 (override defaults).",
    ),
    dd_window: Optional[int] = typer.Option(None, "--dd-window", help="Drawdown rolling window (override default)."),
    vol_z_window: Optional[int] = typer.Option(None, "--vol-z-window", help="Volume z-score window (override default)."),
    ewm_span: Optional[int] = typer.Option(None, "--ewm-span", help="EWM vol span (override default)."),

    # Output
    out: Optional[Path] = typer.Option(None, "--out", help="Output parquet path (defaults to data/processed/...)."),
    write_meta: bool = typer.Option(True, "--meta/--no-meta", help="Write a .meta.json next to parquet."),
    show_tail: int = typer.Option(0, "--show-tail", min=0, help="Print last N rows (0 disables)."),
) -> None:
    """
    Build a features+label-prep dataset (features + future_vol + regime=<NA>).
    Thresholds are fit on train later (Day 3) to avoid leakage.
    """
    start_d = _parse_iso_date(start, name="--start")
    end_d = _parse_iso_date(end, name="--end")

    settings = DataSettings()

    data_kwargs = _merge_settings_with_overrides(
        settings,
        symbol=symbol,
        data_dir=data_dir,
        start=start_d,
        end=end_d,
        source=source,
        interval=interval,
        price_col=price_col,
        refresh=refresh,
    )

    if data_kwargs["price_col"] not in ("adj_close", "close"):
        raise typer.BadParameter("price-col must be 'adj_close' or 'close'.")

    # Feature config defaults + optional overrides
    base_cfg = FeatureConfig(price_col_for_returns=data_kwargs["price_col"])  # keep other defaults
    cfg = FeatureConfig(
        price_col_for_returns=base_cfg.price_col_for_returns,
        vol_windows=_parse_int_tuple(vol_windows, base_cfg.vol_windows),
        mom_windows=_parse_int_tuple(mom_windows, base_cfg.mom_windows),
        dd_window=int(dd_window) if dd_window is not None else base_cfg.dd_window,
        vol_z_window=int(vol_z_window) if vol_z_window is not None else base_cfg.vol_z_window,
        ewm_vol_span=int(ewm_span) if ewm_span is not None else base_cfg.ewm_vol_span,
    )

    # Load data
    df_ohlcv = get_ohlcv(
        symbol=data_kwargs["symbol"],
        data_dir=data_kwargs["data_dir"],
        source=data_kwargs["source"],
        interval=data_kwargs["interval"],
        cache_mode=data_kwargs["cache_mode"],
        start=data_kwargs["start"],
        end=data_kwargs["end"],
    )

    # Build dataset frame (no thresholds yet)
    df_dataset, meta = build_dataset_frame(
        df_ohlcv,
        symbol=data_kwargs["symbol"],
        horizon=int(horizon),
        feature_cfg=cfg,
        thresholds=None,
    )

    # Output path
    out_path = out if out is not None else _default_dataset_out_path(
        data_dir=data_kwargs["data_dir"], symbol=data_kwargs["symbol"], horizon=int(horizon)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_dataset.to_parquet(out_path, index=False)

    typer.echo(
        f"Built dataset for {data_kwargs['symbol'].upper()} h={horizon}. "
        f"Rows={len(df_dataset)} (dropped={meta.dropped_rows}). "
        f"Features={len(meta.feature_columns)}. Saved: {out_path}"
    )

    if write_meta:
        meta_path = _write_meta_json(meta, out_path)
        typer.echo(f"Wrote meta: {meta_path}")

    if show_tail > 0:
        typer.echo(df_dataset.tail(show_tail).to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
