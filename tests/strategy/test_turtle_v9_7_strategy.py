"""Tests for the V9_7 run-proportional trail and the pyramiding (scale-in) logic."""

from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np
import pytest

from user_data.strategies.TurtleStrategyOptV9_7 import TurtleStrategyOptV9_7


def _strategy(atr: float = 4.0, **overrides):
    strategy = object.__new__(TurtleStrategyOptV9_7)
    strategy.dp = Mock()
    strategy._latest_updated_4h_atr = Mock(return_value=atr)
    strategy.sl_atr_init = Mock(value=3.0)
    strategy.sl_atr_profit_trigger = Mock(value=2.0)
    strategy.sl_atr_trail_dist = Mock(value=1.5)
    strategy.trail_giveback_frac = Mock(value=overrides.get("giveback", 0.62))
    strategy.sl_min_profit_pct = Mock(value=overrides.get("min_profit", 0.0))
    strategy.pyramid_add_ratio = Mock(value=overrides.get("ratio", 1.0))
    strategy.pyramid_trigger_atr = Mock(value=overrides.get("trigger", 2.0))
    strategy.pyramid_max_adds = Mock(value=overrides.get("max_adds", 2))
    strategy.risk_per_trade = 0.008
    strategy.max_portfolio_stop_risk = 0.03
    strategy.stoploss = -0.15
    return strategy


def _trade(open_rate: float = 100.0, max_rate: float | None = None, adds: int = 1):
    trade = Mock()
    trade.open_rate = open_rate
    trade.max_rate = max_rate
    trade.is_short = False
    trade.leverage = 1.0
    trade.stake_amount = 1000.0
    trade.nr_of_successful_entries = adds
    return trade


def _call(strategy, trade, current_rate, current_profit):
    return strategy.custom_stoploss(
        pair="BTC/USDT",
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=current_rate,
        current_profit=current_profit,
        after_fill=False,
    )


def test_initial_stop_is_atr_based():
    strategy = _strategy(atr=4.0)
    # 3.0 ATR initial stop => 100 - 12 = 88, i.e. 12% below the current price.
    # stoploss_from_absolute returns the distance as a positive ratio.
    assert _call(strategy, _trade(), 100.0, 0.0) == pytest.approx(0.12, abs=1e-6)


def test_trail_uses_the_wider_of_atr_and_giveback_distance():
    strategy = _strategy(atr=4.0, giveback=0.62)
    trade = _trade(open_rate=100.0, max_rate=200.0)
    # run = 100; giveback distance = 62 -> trail at 138; ATR distance would be 6.
    # The giveback distance must win, otherwise big runs get cut to small ones.
    # trail at 138 vs current 200 -> 62/200 = 31% away.
    assert _call(strategy, trade, 200.0, 1.0) == pytest.approx(0.31, abs=1e-6)


def test_small_run_keeps_the_atr_distance():
    # atr = 1.0 so the 2.0 ATR profit trigger (1.9%) is cleared by a 5% gain.
    strategy = _strategy(atr=1.0, giveback=0.62)
    trade = _trade(open_rate=100.0, max_rate=105.0)
    # run = 5; giveback distance = 3.1 > ATR distance 1.5 -> giveback wins here.
    # trail at 101.9 vs current 105 -> 3.1/105 = 2.95% away.
    assert _call(strategy, trade, 105.0, 0.05) == pytest.approx(0.02952, abs=1e-4)


def test_atr_distance_wins_when_the_run_is_small():
    strategy = _strategy(atr=1.0, giveback=0.1)
    trade = _trade(open_rate=100.0, max_rate=105.0)
    # run = 5; giveback distance = 0.5 < ATR distance 1.5 -> ATR distance wins.
    # trail at 103.5 vs current 105 -> 1.5/105 = 1.43% away.
    assert _call(strategy, trade, 105.0, 0.05) == pytest.approx(0.01429, abs=1e-4)


def test_giveback_scales_with_the_run():
    """A larger giveback fraction must keep more of the run."""
    tight = _strategy(atr=4.0, giveback=0.3)
    loose = _strategy(atr=4.0, giveback=0.8)
    trade = _trade(open_rate=100.0, max_rate=200.0)
    # run = 100. tight: distance = max(6, 30) = 30 -> trail 170.
    # loose: distance = max(6, 80) = 80 -> trail 120.
    # The returned value is the distance from the current price, so the looser
    # trail must return a LARGER distance.
    assert _call(tight, trade, 200.0, 1.0) == pytest.approx(0.15, abs=1e-6)
    assert _call(loose, trade, 200.0, 1.0) == pytest.approx(0.40, abs=1e-6)


def test_profit_floor_is_applied_when_enabled():
    strategy = _strategy(atr=1.0, giveback=0.62, min_profit=8.0)
    trade = _trade(open_rate=100.0, max_rate=105.0)
    # trail would sit at 101.9; the 8% floor is at 108, which is above the
    # current price, so the stop must be rejected (None).
    assert _call(strategy, trade, 105.0, 0.05) is None


def test_profit_floor_lifts_the_stop_when_it_is_reachable():
    strategy = _strategy(atr=1.0, giveback=0.62, min_profit=3.0)
    trade = _trade(open_rate=100.0, max_rate=105.0)
    # trail at 101.9 vs the 3% floor at 103 -> floor wins: 2/105 = 1.90% away.
    assert _call(strategy, trade, 105.0, 0.05) == pytest.approx(0.01905, abs=1e-4)


def _pyramid_call(strategy, trade, current_rate, current_profit):
    return strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=current_rate,
        current_profit=current_profit,
        min_stake=10.0,
        max_stake=100000.0,
        current_entry_rate=current_rate,
        current_exit_rate=current_rate,
        current_entry_profit=current_profit,
        current_exit_profit=current_profit,
    )


def _with_wallet(strategy, equity: float = 100000.0, open_trades=()):
    strategy.wallets = Mock()
    strategy.wallets.get_total_stake_amount = Mock(return_value=equity)
    # The strategy reads the global open-trade table; the tests have no DB.
    strategy._open_trades_for_risk = Mock(return_value=list(open_trades))
    return strategy


def test_pyramid_adds_when_the_trigger_is_cleared():
    strategy = _with_wallet(_strategy(atr=4.0))
    # ATR is 4% of price, trigger at 2.0 ATR => need >8% profit.
    assert _pyramid_call(strategy, _trade(), 100.0, 0.10) is not None


def test_pyramid_does_not_add_below_the_trigger():
    strategy = _with_wallet(_strategy(atr=4.0))
    assert _pyramid_call(strategy, _trade(), 100.0, 0.05) is None


def test_pyramid_respects_the_max_number_of_adds():
    strategy = _with_wallet(_strategy(atr=4.0, max_adds=2))
    # nr_of_successful_entries == 3 means two adds already happened.
    assert _pyramid_call(strategy, _trade(adds=3), 100.0, 0.10) is None


def test_pyramid_is_disabled_when_max_adds_is_zero():
    strategy = _with_wallet(_strategy(atr=4.0, max_adds=0))
    assert _pyramid_call(strategy, _trade(), 100.0, 0.10) is None


def test_pyramid_size_is_capped_by_the_portfolio_risk_budget():
    strategy = _with_wallet(_strategy(atr=4.0), equity=100000.0)
    trade = _trade()
    stake = _pyramid_call(strategy, trade, 100.0, 0.10)
    # risk budget: min(equity*0.008, equity*0.03) = 800 USDT of risk.
    # stop distance = min(3.0*ATR=12, 15) = 12 -> max stake by risk = 800*100/12.
    assert stake <= 800 * 100 / 12 + 1e-6
    assert stake <= trade.stake_amount * 1.0 + 1e-6


def test_no_pyramid_without_fresh_atr():
    strategy = _with_wallet(_strategy())
    strategy._latest_updated_4h_atr = Mock(return_value=None)
    assert _pyramid_call(strategy, _trade(), 100.0, 0.10) is None
