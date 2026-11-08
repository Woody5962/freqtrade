"""Verify the V9_3 live-path minute ATR matches the backtest path."""
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from freqtrade.enums import RunMode

from user_data.strategies import TurtleStrategyOptV9_3 as v93_module
from user_data.strategies.TurtleStrategyOptV9_3 import TurtleStrategyOptV9_3


def _candle_frame(dates):
    return pd.DataFrame({
        "date": dates,
        "open": np.arange(len(dates), dtype=float) + 100,
        "high": np.arange(len(dates), dtype=float) + 102,
        "low": np.arange(len(dates), dtype=float) + 99,
        "close": np.arange(len(dates), dtype=float) + 101,
        "volume": np.ones(len(dates)),
    })


def _bars_for_atr(bucket, minute_count=5):
    previous_dates = [bucket - timedelta(hours=4 * n) for n in range(14, 0, -1)]
    base_df = _candle_frame(previous_dates)
    minute_dates = [bucket + timedelta(minutes=n) for n in range(minute_count)]
    return _candle_frame(minute_dates), base_df


def _strategy(minute_df, base_df):
    s = object.__new__(TurtleStrategyOptV9_3)
    s.dp = v93_module.Mock() if hasattr(v93_module, "Mock") else None
    from unittest.mock import Mock
    s.dp = Mock()
    s.dp.get_pair_dataframe.side_effect = lambda pair, tf: (
        minute_df if tf == "1m" else base_df
    )
    s.atr_period = 14
    s.atr_base_timeframe = "4h"
    s.informative_timeframe = "1m"
    return s


def test_live_atr_matches_backtest_atr(monkeypatch):
    """The live path and the historical path must agree on the ATR value."""
    bucket = pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    minute_df, base_df = _bars_for_atr(bucket, minute_count=60)
    current_time = datetime(2026, 10, 17, 13, 0, tzinfo=UTC)

    # Live path.
    live = _strategy(minute_df, base_df)
    live.dp.runmode = RunMode.DRY_RUN
    live_atr = live._latest_updated_4h_atr("BTC/USDT", current_time)
    assert live_atr is not None
    assert live_atr > 0

    # Historical path over the same window.
    hist = _strategy(minute_df, base_df)
    hist.dp.runmode = RunMode.BACKTEST
    hist.dp.historic_ohlcv.side_effect = lambda pair, tf: (
        minute_df if tf == "1m" else base_df
    )
    hist_atr = hist._latest_updated_4h_atr("BTC/USDT", current_time)
    assert hist_atr is not None
    assert hist_atr == pytest.approx(live_atr, rel=0.02)


def test_stale_minute_data_is_rejected():
    bucket = pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    minute_df, base_df = _bars_for_atr(bucket, minute_count=5)
    s = _strategy(minute_df, base_df)
    s.dp.runmode = RunMode.DRY_RUN
    # 10 minutes after the last closed minute: data is stale.
    assert s._latest_updated_4h_atr("BTC/USDT", datetime(2026, 10, 17, 12, 15, tzinfo=UTC)) is None
