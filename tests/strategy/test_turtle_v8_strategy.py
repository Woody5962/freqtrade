from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from freqtrade.enums import RunMode

from user_data.strategies import TurtleStrategyOptV8 as v8_module
from user_data.strategies.TurtleStrategyOptV8 import TurtleStrategyOptV8


def _candle_frame(dates):
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.arange(len(dates), dtype=float) + 100,
            "high": np.arange(len(dates), dtype=float) + 102,
            "low": np.arange(len(dates), dtype=float) + 99,
            "close": np.arange(len(dates), dtype=float) + 101,
            "volume": np.ones(len(dates)),
        }
    )


def _strategy_with_frames(minute_df, base_df):
    strategy = object.__new__(TurtleStrategyOptV8)
    strategy.dp = Mock()
    strategy.dp.get_pair_dataframe.side_effect = lambda pair, timeframe: (
        minute_df if timeframe == "1m" else base_df
    )
    return strategy


def _bars_for_atr(bucket, minute_count=5):
    previous_dates = [bucket - timedelta(hours=4 * n) for n in range(14, 0, -1)]
    base_df = _candle_frame(previous_dates)
    minute_dates = [bucket + timedelta(minutes=n) for n in range(minute_count)]
    minute_df = _candle_frame(minute_dates)
    return minute_df, base_df


def test_minute_bars_build_current_4h_atr_without_using_open_minute(monkeypatch):
    bucket = pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    minute_df, base_df = _bars_for_atr(bucket)
    # This row is still forming at 12:05 and must not be included.
    minute_df = pd.concat(
        [minute_df, _candle_frame([bucket + pd.Timedelta(minutes=5)])], ignore_index=True
    )
    strategy = _strategy_with_frames(minute_df, base_df)
    captured = {}

    def fake_atr(dataframe, timeperiod):
        captured["dataframe"] = dataframe.copy()
        captured["timeperiod"] = timeperiod
        values = [np.nan] * (len(dataframe) - 1) + [123.0]
        return pd.Series(values, index=dataframe.index)

    monkeypatch.setattr(v8_module.ta, "ATR", fake_atr)
    current_time = datetime(2026, 10, 17, 12, 5, tzinfo=UTC)

    atr = strategy._latest_updated_4h_atr("BTC/USDT", current_time)

    assert atr == 123.0
    assert captured["timeperiod"] == 14
    bars = captured["dataframe"]
    assert len(bars) == 15
    assert bars.iloc[-1]["date"] == bucket
    assert bars.iloc[-1]["open"] == minute_df.iloc[0]["open"]
    assert bars.iloc[-1]["high"] == minute_df.iloc[:5]["high"].max()
    assert bars.iloc[-1]["low"] == minute_df.iloc[:5]["low"].min()
    assert bars.iloc[-1]["close"] == minute_df.iloc[4]["close"]


def test_backtest_minute_atr_lookup_matches_live_calculation_and_is_cached(monkeypatch):
    bucket = pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    minute_df, base_df = _bars_for_atr(bucket, minute_count=5)
    previous_extra = _candle_frame([bucket - pd.Timedelta(hours=4 * 15)])
    base_df = pd.concat([previous_extra, base_df], ignore_index=True)
    strategy = _strategy_with_frames(minute_df, base_df)
    strategy.dp.runmode = RunMode.BACKTEST
    strategy.dp.historic_ohlcv.side_effect = lambda pair, timeframe: (
        minute_df if timeframe == "1m" else base_df
    )
    current_time = datetime(2026, 10, 17, 12, 5, tzinfo=UTC)
    expected_partial = {
        "date": bucket,
        "open": minute_df["open"].iloc[0],
        "high": minute_df["high"].max(),
        "low": minute_df["low"].min(),
        "close": minute_df["close"].iloc[-1],
        "volume": minute_df["volume"].sum(),
    }
    expected_atr = v8_module.ta.ATR(
        pd.concat([base_df.tail(14), pd.DataFrame([expected_partial])], ignore_index=True),
        timeperiod=14,
    ).iloc[-1]

    actual = strategy._latest_updated_4h_atr("BTC/USDT", current_time)
    assert actual == pytest.approx(expected_atr)
    assert strategy.dp.historic_ohlcv.call_count == 2
    strategy._latest_updated_4h_atr("BTC/USDT", current_time)
    assert strategy.dp.historic_ohlcv.call_count == 2


@pytest.mark.parametrize("failure", ["gap", "stale"])
def test_minute_atr_rejects_gaps_and_stale_candles(monkeypatch, failure):
    bucket = pd.Timestamp("2026-10-17 12:00:00", tz="UTC")
    minute_df, base_df = _bars_for_atr(bucket)
    if failure == "gap":
        minute_df = minute_df.drop(index=2).reset_index(drop=True)
        current_time = datetime(2026, 10, 17, 12, 5, tzinfo=UTC)
    else:
        current_time = datetime(2026, 10, 17, 12, 8, tzinfo=UTC)
    strategy = _strategy_with_frames(minute_df, base_df)
    monkeypatch.setattr(v8_module.ta, "ATR", Mock())

    assert strategy._latest_updated_4h_atr("BTC/USDT", current_time) is None
    v8_module.ta.ATR.assert_not_called()


def test_custom_stoploss_uses_latest_atr_and_trailing_high(monkeypatch):
    strategy = object.__new__(TurtleStrategyOptV8)
    strategy._latest_updated_4h_atr = Mock(return_value=100.0)
    # Pin the trailing parameters explicitly: object.__new__ skips __init__, so
    # these would otherwise fall back to the class defaults.
    strategy.sl_atr_init = SimpleNamespace(value=3.6)
    strategy.sl_atr_profit_trigger = SimpleNamespace(value=2.5)
    strategy.sl_atr_trail_dist = SimpleNamespace(value=2.1)
    strategy.stoploss = -0.1
    trade = SimpleNamespace(
        open_rate=1000.0,
        max_rate=1300.0,
        is_short=False,
        leverage=1.0,
    )

    result = strategy.custom_stoploss(
        "BTC/USDT",
        trade,
        datetime(2026, 10, 17, 12, 5, tzinfo=UTC),
        current_rate=1250.0,
        current_profit=0.25,
        after_fill=False,
    )

    # Highest-price trailing stop: 1300 - 2.1 * 100 = 1090.
    assert result == pytest.approx(1 - 1090 / 1250)


def test_custom_stoploss_keeps_initial_stop_below_the_trailing_trigger(monkeypatch):
    """Before the profit trigger the trailing stop must not engage."""
    strategy = object.__new__(TurtleStrategyOptV8)
    strategy._latest_updated_4h_atr = Mock(return_value=100.0)
    strategy.sl_atr_init = SimpleNamespace(value=3.6)
    strategy.sl_atr_profit_trigger = SimpleNamespace(value=3.3)
    strategy.sl_atr_trail_dist = SimpleNamespace(value=2.1)
    strategy.stoploss = -0.1
    trade = SimpleNamespace(
        open_rate=1000.0,
        max_rate=1300.0,
        is_short=False,
        leverage=1.0,
    )

    # 25% profit is below the 3.3 ATR (33%) trigger, so only the initial
    # stop applies. At ATR=100 the 3.6 ATR distance (640) is wider than the
    # 10% catastrophic stop (900), so the catastrophic stop wins.
    result = strategy.custom_stoploss(
        "BTC/USDT",
        trade,
        datetime(2026, 10, 17, 12, 5, tzinfo=UTC),
        current_rate=1250.0,
        current_profit=0.25,
        after_fill=False,
    )

    assert result == pytest.approx(1 - 900 / 1250)


def test_custom_stoploss_uses_atr_stop_when_it_is_tighter_than_catastrophic():
    """With a tight ATR the ATR distance must win over the hard stoploss."""
    strategy = object.__new__(TurtleStrategyOptV8)
    strategy._latest_updated_4h_atr = Mock(return_value=20.0)
    strategy.sl_atr_init = SimpleNamespace(value=3.6)
    strategy.sl_atr_profit_trigger = SimpleNamespace(value=3.3)
    strategy.sl_atr_trail_dist = SimpleNamespace(value=2.1)
    strategy.stoploss = -0.1
    trade = SimpleNamespace(
        open_rate=1000.0,
        max_rate=1010.0,
        is_short=False,
        leverage=1.0,
    )

    result = strategy.custom_stoploss(
        "BTC/USDT",
        trade,
        datetime(2026, 10, 17, 12, 5, tzinfo=UTC),
        current_rate=1000.0,
        current_profit=0.0,
        after_fill=False,
    )

    # 1000 - 3.6 * 20 = 928 is tighter than the 900 catastrophic stop.
    assert result == pytest.approx(1 - 928 / 1000)


def test_custom_stoploss_keeps_existing_stop_when_minute_atr_is_unavailable(caplog):
    strategy = object.__new__(TurtleStrategyOptV8)
    strategy._latest_updated_4h_atr = Mock(return_value=None)
    trade = SimpleNamespace(open_rate=1000.0, max_rate=1000.0, is_short=False, leverage=1.0)

    assert (
        strategy.custom_stoploss(
            "BTC/USDT",
            trade,
            datetime(2026, 10, 17, 12, 5, tzinfo=UTC),
            current_rate=1000.0,
            current_profit=0.0,
            after_fill=False,
        )
        is None
    )
    assert "No fresh 1m ATR" in caplog.text


def test_stake_sizing_obeys_per_trade_portfolio_and_minimum_limits(monkeypatch):
    strategy = object.__new__(TurtleStrategyOptV8)
    strategy._latest_updated_4h_atr = Mock(return_value=50.0)
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10_000.0)
    open_trades = [
        SimpleNamespace(open_rate=1000.0, stop_loss=910.0, stake_amount=1000.0),
        SimpleNamespace(open_rate=1000.0, stop_loss=910.0, stake_amount=1000.0),
    ]
    monkeypatch.setattr(v8_module.Trade, "get_open_trades", lambda: open_trades)
    params = {
        "pair": "BTC/USDT",
        "current_time": datetime(2026, 10, 17, 12, 5, tzinfo=UTC),
        "current_rate": 1000.0,
        "proposed_stake": 5000.0,
        "min_stake": 10.0,
        "max_stake": 5000.0,
        "leverage": 1.0,
        "entry_tag": "dc_ema_adx_breakout",
        "side": "long",
    }

    # Existing stops risk 180 USDT; the 2% portfolio cap leaves 20 USDT.
    assert strategy.custom_stake_amount(**params) == pytest.approx(200.0)
    params["min_stake"] = 201.0
    assert strategy.custom_stake_amount(**params) == 0.0
