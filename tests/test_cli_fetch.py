from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import volregime.cli as cli_module


runner = CliRunner()


@dataclass
class FakeSettings:
    symbol: str = "SPY"
    data_dir: Path = Path("data")
    start: date | None = None
    end: date | None = None
    source: str = "yfinance"
    interval: str = "1d"
    cache_mode: str = "use"


def test_cli_fetch_calls_data_layer_with_defaults(monkeypatch):
    # Arrange: replace DataSettings() constructor to return our fake settings
    monkeypatch.setattr(cli_module, "DataSettings", lambda: FakeSettings())

    called = {}

    def fake_get_ohlcv(**kwargs):
        called["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.1, 1.2],
                "adj_close": [1.1, 1.2],
                "volume": [100, 200],
            }
        )

    def fake_read_cache_metadata(*, symbol, data_dir, source, interval):
        return {
            "symbol": symbol,
            "cache_file": str(Path(data_dir) / "raw" / "dummy.parquet"),
        }

    monkeypatch.setattr(cli_module, "get_ohlcv", fake_get_ohlcv)
    monkeypatch.setattr(cli_module, "read_cache_metadata", fake_read_cache_metadata)

    # Act
    result = runner.invoke(cli_module.app, ["fetch", "--show-tail", "0"])

    # Assert
    assert result.exit_code == 0, result.stdout
    assert "Loaded 2 rows for SPY" in result.stdout

    kwargs = called["kwargs"]
    assert kwargs["symbol"] == "SPY"
    assert kwargs["data_dir"] == Path("data")
    assert kwargs["source"] == "yfinance"
    assert kwargs["interval"] == "1d"
    assert kwargs["cache_mode"] == "use"


def test_cli_fetch_overrides_and_no_meta(monkeypatch):
    monkeypatch.setattr(cli_module, "DataSettings", lambda: FakeSettings())

    called = {}

    def fake_get_ohlcv(**kwargs):
        called["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "open": [1.0],
                "high": [1.2],
                "low": [0.9],
                "close": [1.1],
                "adj_close": [1.1],
                "volume": [100],
            }
        )

    # If --no-meta is passed, this must NOT be called.
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "read_cache_metadata should not be called when --no-meta is used"
        )

    monkeypatch.setattr(cli_module, "get_ohlcv", fake_get_ohlcv)
    monkeypatch.setattr(cli_module, "read_cache_metadata", fail_if_called)

    result = runner.invoke(
        cli_module.app,
        [
            "fetch",
            "--symbol",
            "QQQ",
            "--start",
            "2018-01-01",
            "--refresh",
            "--no-meta",
            "--show-tail",
            "0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Loaded 1 rows for QQQ" in result.stdout

    kwargs = called["kwargs"]
    assert kwargs["symbol"] == "QQQ"
    assert kwargs["start"] == date(2018, 1, 1)
    assert kwargs["cache_mode"] == "refresh"
