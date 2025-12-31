from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    """
    Volatility regime thresholds.

    By convention:
      vol < low  -> regime 0 (LOW)
      low <= vol < high -> regime 1 (MID)
      vol >= high -> regime 2 (HIGH)
    """

    low: float
    high: float

    def to_dict(self) -> dict:
        return {"low": float(self.low), "high": float(self.high)}

    @staticmethod
    def from_dict(d: dict) -> "Thresholds":
        return Thresholds(low=float(d["low"]), high=float(d["high"]))


def compute_future_vol(ret_1: pd.Series, horizon: int) -> pd.Series:
    """
    Compute realized future volatility for each t over returns t+1..t+horizon.

    Alignment: output at index t corresponds to std(ret[t+1], ..., ret[t+horizon]).
    The last `horizon` rows will be NaN (insufficient future data).
    """
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be > 0")

    # Step 1: shift so that at time t we start from t+1
    shifted = ret_1.shift(-1)

    # Step 2: rolling std computed ending at t (by default it aligns to the right),
    # so we shift back by (h-1) to place the value at original time t.
    future_vol = shifted.rolling(window=h, min_periods=h).std().shift(-(h - 1))

    return future_vol


def fit_thresholds(
    future_vol: pd.Series, q_low: float = 1 / 3, q_high: float = 2 / 3
) -> Thresholds:
    """
    Fit thresholds from given volatility series.
    """
    s = future_vol.dropna()
    if s.empty:
        raise ValueError(
            "future_vol_train is empty after dropping NaNs; cannot fit thresholds."
        )

    low = float(s.quantile(q_low))
    high = float(s.quantile(q_high))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Threshold quantiles produced non-finite values.")
    if low >= high:
        raise ValueError(f"Invalid thresholds: low ({low}) must be < high ({high}).")

    return Thresholds(low=low, high=high)


def make_regime_labels(future_vol: pd.Series, thresholds: Thresholds) -> pd.Series:
    """
    Convert future volatility into integer regimes: 0, 1, 2.
    Uses pandas nullable Int64 dtype to allow <NA>.
    """
    fv = future_vol.copy()

    out = pd.Series(pd.NA, index=fv.index, dtype="Int64")
    mask = fv.notna()

    out.loc[mask & (fv < thresholds.low)] = 0
    out.loc[mask & (fv >= thresholds.low) & (fv < thresholds.high)] = 1
    out.loc[mask & (fv >= thresholds.high)] = 2

    return out
