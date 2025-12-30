from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PriceCol = Literal["adj_close", "close"]


@dataclass(frozen=True)
class FeatureConfig:
    """
    Feature configuration. Keep this small for the MVP.

    Windows are in trading sessions (daily bars).
    """
    price_col_for_returns: PriceCol = "adj_close"
    vol_windows: tuple[int, ...] = (5, 10, 20, 60)
    mom_windows: tuple[int, ...] = (5, 10, 20)
    dd_window: int = 20
    vol_z_window: int = 20
    ewm_vol_span: int = 20


def _validate_ohlcv_contract(df: pd.DataFrame) -> None:
    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    if list(df.columns) != expected:
        raise ValueError(
            "Unexpected OHLCV schema. Expected columns in exact order:\n"
            f"{expected}\nGot:\n{list(df.columns)}"
        )
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        raise ValueError("df['date'] must be datetime64[ns].")


def _safe_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window=window, min_periods=window).mean()
    sd = x.rolling(window=window, min_periods=window).std()
    sd = sd.replace(0.0, np.nan)
    return (x - mu) / sd


def build_features(df_ohlcv: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Build a feature frame from normalized OHLCV.

    Input contract: columns must be exactly:
      date, open, high, low, close, adj_close, volume

    Output: a dataframe with:
      date, ret_1, range_pct, gap_pct, vol_*, ewm_vol_*, mom_*, dd_*, vol_z_*
    """
    _validate_ohlcv_contract(df_ohlcv)

    df = df_ohlcv.copy()
    price = df[cfg.price_col_for_returns].astype(float)

    # 1) Base return series
    ret_1 = np.log(price / price.shift(1))
    df_feat = pd.DataFrame({"date": df["date"]})
    df_feat["ret_1"] = ret_1

    # 2) Intraday range and gap
    df_feat["range_pct"] = (df["high"].astype(float) - df["low"].astype(float)) / df["close"].astype(float)

    prev_close = df["close"].astype(float).shift(1)
    df_feat["gap_pct"] = (df["open"].astype(float) - prev_close) / prev_close

    # 3) Rolling realized vol of returns
    for w in cfg.vol_windows:
        df_feat[f"vol_{w}"] = df_feat["ret_1"].rolling(window=w, min_periods=w).std()

    # 4) Exponentially weighted vol (useful, cheap)
    df_feat[f"ewm_vol_{cfg.ewm_vol_span}"] = (
        df_feat["ret_1"].ewm(span=cfg.ewm_vol_span, min_periods=cfg.ewm_vol_span, adjust=False).std(bias=False)
    )

    # 5) Momentum
    for w in cfg.mom_windows:
        df_feat[f"mom_{w}"] = (price / price.shift(w)) - 1.0

    # 6) Drawdown vs rolling max
    roll_max = price.rolling(window=cfg.dd_window, min_periods=cfg.dd_window).max()
    df_feat[f"dd_{cfg.dd_window}"] = (price / roll_max) - 1.0

    # 7) Volume z-score
    df_feat[f"vol_z_{cfg.vol_z_window}"] = _safe_zscore(df["volume"].astype(float), window=cfg.vol_z_window)

    # Sanity: stable column ordering
    # Note: date first, then ret_1, then everything else sorted.
    base = ["date", "ret_1"]
    others = sorted([c for c in df_feat.columns if c not in base])
    df_feat = df_feat[base + others]

    return df_feat
