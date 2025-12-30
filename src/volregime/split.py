from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """
    Temporal split indices as integer positions in the dataframe.
    """
    train_end: int
    val_end: int

    def as_slices(self) -> tuple[slice, slice, slice]:
        return slice(0, self.train_end), slice(self.train_end, self.val_end), slice(self.val_end, None)


def temporal_split_by_ratio(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> TemporalSplit:
    """
    Split by ratios over time (no shuffling).
    """
    n = len(df)
    if n < 10:
        raise ValueError("Not enough rows to split reliably (need >= 10).")
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or (train_ratio + val_ratio) >= 1:
        raise ValueError("Invalid ratios. Need 0<train<1, 0<val<1, and train+val<1.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Ensure non-empty splits
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError("Split produced empty partition(s). Adjust ratios or provide more data.")

    return TemporalSplit(train_end=train_end, val_end=val_end)


def temporal_split_by_date(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    train_end_date: pd.Timestamp,
    val_end_date: pd.Timestamp,
) -> TemporalSplit:
    """
    Split using cut-off dates (inclusive on the left partitions).
    Expects df sorted ascending by date.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    d = pd.to_datetime(df[date_col])
    if not d.is_monotonic_increasing:
        raise ValueError("df must be sorted ascending by date for temporal_split_by_date.")

    train_end = int((d <= train_end_date).sum())
    val_end = int((d <= val_end_date).sum())

    n = len(df)
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError("Date split produced empty partition(s). Adjust cut-off dates.")

    return TemporalSplit(train_end=train_end, val_end=val_end)
