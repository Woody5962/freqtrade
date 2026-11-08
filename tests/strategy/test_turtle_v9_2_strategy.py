"""Tests for the V9_2 staged ATR stoploss and risk-based position sizing."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from user_data.strategies import TurtleStrategyOptV9_2 as v9_2_module
from user_data.strategies.TurtleStrategyOptV9_2 import TurtleStrategyOptV9_2


def _strategy(atr: float = 4.0):
    strategy = object.__new__(TurtleStrategyOptV9_2)
    strategy.dp = Mock()
    strategy.dp.get_analyzed_dataframe.return_value = (
        pd.DataFrame({"atr_4h": [atr]}),
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    strategy.sl_atr_init.value = 3.0
    strategy.sl_atr_profit_trigger.value = 2.0
    strategy.sl_atr_trail_dist.value = 1.5
    strategy.sl_breakeven_atr.value = 1.0
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
    trade = _trade()

    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=100.0,
        current_profit=0.0,
        after_fill=False,
    )

    # open_rate - 3.0 ATR = 88.0 -> stoploss is 12% below current rate
    assert result == pytest.approx(0.12)


def test_breakeven_does_not_preempt_the_trailing_stage():
    """Between breakeven and trail trigger the stop must not jump to flat."""
    strategy = _strategy(atr=4.0)
    trade = _trade(max_rate=104.0)

    # 4% profit: above breakeven (1 ATR = 3.8%) but below trail trigger
    # (2 ATR = 7.7%).
    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=104.0,
        current_profit=0.04,
        after_fill=False,
    )

    # The stop stays at the wide initial ATR distance, not at the entry price.
    assert result == pytest.approx(1 - 88.0 / 104.0)


def test_trailing_stage_tracks_the_high_watermark():
    strategy = _strategy(atr=4.0)
    trade = _trade(max_rate=120.0)

    # 20% profit, well past the 2 ATR trail trigger.
    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=120.0,
        current_profit=0.20,
        after_fill=False,
    )

    # high watermark 120 - 1.5 ATR = 114 -> 5% below current rate
    assert result == pytest.approx(1 - 114.0 / 120.0)


def test_trailing_stage_keeps_breakeven_floor():
    """The trail must never sit below the entry price once it activates."""
    strategy = _strategy(atr=4.0)
    strategy.sl_atr_trail_dist.value = 5.0
    strategy.sl_breakeven_atr.value = 2.0
    trade = _trade(max_rate=120.0)

    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=120.0,
        current_profit=0.20,
        after_fill=False,
    )

    # trail would be 120 - 20 = 100, but the breakeven floor lifts it to
    # open_rate + (2.0 - 1.0) * ATR = 104.
    assert result == pytest.approx(1 - 104.0 / 120.0)


def test_catastrophic_stop_caps_the_initial_distance():
    """A huge ATR must not widen the stop beyond the hard stoploss."""
    strategy = _strategy(atr=40.0)
    trade = _trade()

    result = strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=100.0,
        current_profit=0.0,
        after_fill=False,
    )

    # The -15% catastrophic stop is tighter than open - 3 ATR.
    assert result == pytest.approx(0.15)
