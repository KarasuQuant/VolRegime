from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """
    Configuration for data ingestion and caching.

    Values can be overridden via environment variables using the prefix VOLREGIME_.
    Example:
      export VOLREGIME_SYMBOL=QQQ
      export VOLREGIME_DATA_DIR=/tmp/volregime
    """

    model_config = SettingsConfigDict(
        env_prefix="VOLREGIME_",
        case_sensitive=False,
        extra="ignore",
    )

    # What to fetch
    symbol: str = Field(default="SPY", description="Ticker symbol to fetch.")
    source: Literal["yfinance"] = Field(default="yfinance", description="Data source.")
    interval: Literal["1d"] = Field(default="1d", description="Data interval (daily).")

    # Optional filtering when returning data (cache can store full history)
    start: Optional[date] = Field(
        default=None, description="Filter start date (inclusive)."
    )
    end: Optional[date] = Field(
        default=None, description="Filter end date (inclusive)."
    )

    # Where to store files
    data_dir: Path = Field(default=Path("data"), description="Base data directory.")
    raw_dirname: str = Field(default="raw", description="Subdir for raw cache.")
    processed_dirname: str = Field(
        default="processed", description="Subdir for processed outputs."
    )

    # Cache behavior
    cache_mode: Literal["use", "refresh"] = Field(
        default="use",
        description="use: load cache if present; refresh: always refetch and overwrite.",
    )

    # Convention for downstream feature/label building
    price_col_for_returns: Literal["adj_close", "close"] = Field(
        default="adj_close",
        description="Column used for returns computations downstream.",
    )

    def raw_dir(self) -> Path:
        return self.data_dir / self.raw_dirname

    def processed_dir(self) -> Path:
        return self.data_dir / self.processed_dirname
