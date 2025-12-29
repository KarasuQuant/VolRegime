from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

# Tipos simples para no acoplar a Settings
Source = Literal["yfinance"]
Interval = Literal["1d"]
CacheMode = Literal["use", "refresh"]


@dataclass(frozen=True)
class CachePaths:
    raw_parquet: Path
    raw_meta: Path


def _ensure_dirs(data_dir: Path) -> None:
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)


def _cache_paths(data_dir: Path, symbol: str, source: Source, interval: Interval) -> CachePaths:
    symbol_u = symbol.upper()
    stem = f"{symbol_u}_{source}_{interval}_full"
    raw_dir = data_dir / "raw"
    return CachePaths(
        raw_parquet=raw_dir / f"{stem}.parquet",
        raw_meta=raw_dir / f"{stem}.meta.json",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty dataframe received from data source.")

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["date"] = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df.reset_index(drop=True, inplace=True)
    elif "Date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["Date"], utc=False).dt.tz_localize(None)
    else:
        raise ValueError("Could not find a datetime index or 'Date' column in the fetched dataframe.")

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    missing = [k for k in rename_map.keys() if k not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from source: {missing}")

    df = df[["date"] + list(rename_map.keys())].rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "adj_close"]).reset_index(drop=True)

    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns after normalization. Got: {list(df.columns)}")

    return df


def _fetch_yfinance_full_history(symbol: str, interval: Interval = "1d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is not installed. Add it via `uv add yfinance`.") from e

    df = yf.download(
        tickers=symbol,
        period="max",
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        ticker = symbol.upper()
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.droplevel(-1, axis=1)

    return df


def read_cache_metadata(
    *,
    symbol: str,
    data_dir: Path = Path("data"),
    source: Source = "yfinance",
    interval: Interval = "1d",
) -> Optional[dict]:
    paths = _cache_paths(data_dir, symbol, source, interval)
    if not paths.raw_meta.exists():
        return None
    return _read_json(paths.raw_meta)


def get_ohlcv(
    *,
    symbol: str,
    data_dir: Path = Path("data"),
    source: Source = "yfinance",
    interval: Interval = "1d",
    cache_mode: CacheMode = "use",
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Core API: parameterized, no dependency on any Settings class.
    """
    _ensure_dirs(data_dir)
    paths = _cache_paths(data_dir, symbol, source, interval)

    use_cache = paths.raw_parquet.exists() and cache_mode == "use"

    if use_cache:
        df = pd.read_parquet(paths.raw_parquet)
    else:
        if source != "yfinance":
            raise ValueError(f"Unsupported source: {source}")

        raw = _fetch_yfinance_full_history(symbol=symbol, interval=interval)
        df = _normalize_ohlcv(raw)

        df.to_parquet(paths.raw_parquet, index=False)
        _write_json(
            paths.raw_meta,
            {
                "symbol": symbol.upper(),
                "source": source,
                "interval": interval,
                "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds") + "Z",
                "rows": int(df.shape[0]),
                "cols": list(df.columns),
                "cache_file": str(paths.raw_parquet),
            },
        )

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.reset_index(drop=True)
