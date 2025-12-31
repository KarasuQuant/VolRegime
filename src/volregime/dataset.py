from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .features import FeatureConfig, build_features
from .labels import Thresholds, compute_future_vol, make_regime_labels


@dataclass(frozen=True)
class DatasetMeta:
    symbol: str
    horizon: int
    dropped_rows: int
    feature_columns: list[str]
    thresholds: Optional[dict] = None


def build_feature_frame(
    df_ohlcv: pd.DataFrame, feature_cfg: FeatureConfig
) -> pd.DataFrame:
    """
    Thin wrapper around build_features to keep naming consistent.
    """
    return build_features(df_ohlcv, feature_cfg)


def build_dataset_frame(
    df_ohlcv: pd.DataFrame,
    *,
    symbol: str,
    horizon: int,
    feature_cfg: FeatureConfig,
    thresholds: Optional[Thresholds] = None,
) -> tuple[pd.DataFrame, DatasetMeta]:
    """
    Build a single dataframe containing:
      - date
      - features
      - future_vol
      - regime (if thresholds provided)
    """
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be > 0")

    df_feat = build_feature_frame(df_ohlcv, feature_cfg)

    # future vol uses ret_1 (computed in features)
    df_feat["future_vol"] = compute_future_vol(df_feat["ret_1"], horizon=h)

    if thresholds is not None:
        df_feat["regime"] = make_regime_labels(df_feat["future_vol"], thresholds)
    else:
        # Keep as nullable Int64 with NA to preserve schema expectations downstream
        df_feat["regime"] = pd.Series(pd.NA, index=df_feat.index, dtype="Int64")

    # Identify feature columns (exclude targets and non-features)
    non_features = {"date", "future_vol", "regime"}
    feature_cols = [c for c in df_feat.columns if c not in non_features]

    # Drop rows with any NaNs in features or target-related columns we need
    before = len(df_feat)
    df_clean = df_feat.dropna(subset=feature_cols + ["future_vol"]).reset_index(
        drop=True
    )
    dropped = before - len(df_clean)

    meta = DatasetMeta(
        symbol=symbol.upper(),
        horizon=h,
        dropped_rows=int(dropped),
        feature_columns=feature_cols,
        thresholds=thresholds.to_dict() if thresholds is not None else None,
    )
    return df_clean, meta


def to_xy(
    df_dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    require_labels: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert dataset frame into X, y.
    """
    if require_labels:
        if "regime" not in df_dataset.columns:
            raise ValueError("df_dataset missing 'regime' column.")
        if df_dataset["regime"].isna().any():
            raise ValueError(
                "Labels contain NA. Ensure thresholds are fit and labels are generated before training."
            )

    X = df_dataset[feature_columns].copy()
    y = df_dataset["regime"].astype("int64") if require_labels else df_dataset["regime"]
    return X, y
