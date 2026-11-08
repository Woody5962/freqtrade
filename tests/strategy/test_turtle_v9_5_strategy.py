"""Tests for the V9_5 regime filter and staged ATR stoploss."""

from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.TurtleStrategyOptV9_5 import TurtleStrategyOptV9_5


def _strategy(atr: float = 4.0):
    strategy = object.__new__(TurtleStrategyOptV9_5)
    strategy.dp = Mock()
    # V9_5 sizes and stops off the minute-resolution ATR, not the 15m frame.
    strategy._latest_updated_4h_atr = Mock(return_value=atr)
    strategy.sl_atr_init = Mock(value=3.0)
    strategy.sl_atr_profit_trigger = Mock(value=2.0)
    strategy.sl_atr_trail_dist = Mock(value=1.5)
    strategy.sl_breakeven_atr = Mock(value=1.0)
    strategy.stoploss = -0.15
    return strategy


def _trade(open_rate: float = 100.0, max_rate: float | None = None):
    trade = Mock()
    trade.open_rate = open_rate
    trade.max_rate = max_rate
    trade.is_short = False
    trade.leverage = 1.0
    return trade


def test_initial_stop_uses_wide_atr_distance():
    strategy = _strategy(atr=4.0)
    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=_trade(),
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=100.0,
        current_profit=0.0,
        after_fill=False,
    )
    # open_rate - 3.0 ATR = 88.0
    assert result == pytest.approx(1 - 88.0 / 100.0)


def test_trailing_stage_tracks_the_high_watermark():
    strategy = _strategy(atr=4.0)
    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=_trade(max_rate=120.0),
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=120.0,
        current_profit=0.20,
        after_fill=False,
    )
    # high watermark 120 - 1.5 ATR = 114
    assert result == pytest.approx(1 - 114.0 / 120.0)


def test_breakeven_does_not_preempt_the_trailing_stage():
    """Between breakeven and trail trigger the stop must stay at the wide stop."""
    strategy = _strategy(atr=4.0)
    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=_trade(max_rate=104.0),
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=104.0,
        current_profit=0.04,
        after_fill=False,
    )
    # Still the initial ATR stop, not the entry price.
    assert result == pytest.approx(1 - 88.0 / 104.0)


def test_regime_filter_blocks_entries_below_the_long_term_ema():
    """A close below the regime EMA must block the entry."""
    strategy = object.__new__(TurtleStrategyOptV9_5)
    strategy.dp = Mock()
    strategy.entry_dc_period = Mock(value=20)
    strategy.ema_fast_period = Mock(value=19)
    strategy.ema_slow_period = Mock(value=37)
    strategy.adx_period = Mock(value=21)
    strategy.adx_threshold = Mock(value=25)
    strategy.regime_enabled = Mock(value=1)
    strategy.regime_ema_period = Mock(value=200)

    rows = 2
    dataframe = pd.DataFrame(
        {
            "date_4h": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 04:00"], utc=True
            ),
            "close_4h": [100.0, 100.0],
            "dc_upper_20_4h": [90.0, 90.0],
            "ema_fast_19_4h": [95.0, 95.0],
            "ema_slow_37_4h": [85.0, 85.0],
            "adx_21_4h": [30.0, 30.0],
            "regime_ema_200_4h": [120.0, 120.0],
            "volume": [1.0, 1.0],
        }
    )
    out = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})
    assert out["enter_long"].tolist() == [0, 0]


def test_regime_filter_allows_entries_above_the_long_term_ema():
    strategy = object.__new__(TurtleStrategyOptV9_5)
    strategy.dp = Mock()
    strategy.entry_dc_period = Mock(value=20)
    strategy.ema_fast_period = Mock(value=19)
    strategy.ema_slow_period = Mock(value=37)
    strategy.adx_period = Mock(value=21)
    strategy.adx_threshold = Mock(value=25)
    strategy.regime_enabled = Mock(value=1)
    strategy.regime_ema_period = Mock(value=200)

    dataframe = pd.DataFrame(
        {
            "date_4h": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 04:00"], utc=True
            ),
            "close_4h": [100.0, 100.0],
            "dc_upper_20_4h": [90.0, 90.0],
            "ema_fast_19_4h": [95.0, 95.0],
            "ema_slow_37_4h": [85.0, 85.0],
            "adx_21_4h": [30.0, 30.0],
            "regime_ema_200_4h": [80.0, 80.0],
            "volume": [1.0, 1.0],
        }
    )
    out = strategy.populate_entry_trend(dataframe, {"pair": "BTC/USDT"})
    # Both rows are distinct 4h candles, so both are fresh signals.
    assert out["enter_long"].tolist() == [1, 1]
    assert out["enter_tag"].tolist() == ["v9_5_4h_signal", "v9_5_4h_signal"]
