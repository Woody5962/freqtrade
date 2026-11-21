"""Backtest ATR path vs live ATR path must agree.

V9_7 computes the 4h ATR two different ways depending on the run mode:

* backtest / hyperopt: ``_historical_minute_atr`` - a vectorised pre-computation
  over the whole 1m history;
* live / dry-run: ``_latest_updated_4h_atr`` - rebuilds the *currently forming*
  4h candle from closed 1m bars and feeds it to TA-Lib together with the last
  13 completed 4h candles.

If these two disagree, the backtest is measuring a strategy that does not exist
in production: the whole backtest/live parity of the strategy depends on them
being the same number.  These tests build synthetic 1m + 4h data and compare
both paths minute by minute.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from freqtrade.enums import RunMode
from user_data.strategies.TurtleStrategyOptV9_7 import TurtleStrategyOptV9_7


class _FakeDP:
    """Minimal DataProvider stand-in for both ATR code paths."""

    def __init__(self, minute_df: pd.DataFrame, base_df: pd.DataFrame, pair: str):
        self._minute = minute_df
        self._base = base_df
        self._pair = pair
        self.runmode = RunMode.DRY_RUN

    def historic_ohlcv(self, pair: str, timeframe: str) -> pd.DataFrame:
        return self._minute if timeframe == "1m" else self._base

    def get_pair_dataframe(self, pair: str, timeframe: str) -> pd.DataFrame:
        return self._minute if timeframe == "1m" else self._base

    def current_whitelist(self) -> list[str]:
        return [self._pair]


def _synthetic_pair(pair: str = "TEST/USDT", seed: int = 7):
    """Build 1m and 4h candles that are consistent with each other."""
    rng = np.random.default_rng(seed)
    # 4h buckets, 12 days -> enough for ATR(14) plus history
    start = pd.Timestamp("2025-03-01 00:00", tz=UTC)
    n_buckets = 60
    bucket_hours = 4

    base_rows = []
    minute_rows = []
    price = 100.0
    for b in range(n_buckets):
        bstart = start + pd.Timedelta(hours=bucket_hours * b)
        o = price
        # random walk over the bucket
        steps = rng.normal(0, 0.25, size=bucket_hours * 60)
        path = o + np.cumsum(steps)
        high = float(path.max())
        low = float(path.min())
        close = float(path[-1])
        base_rows.append(
            {
                "date": bstart,
                "open": o,
                "high": max(high, o, close),
                "low": min(low, o, close),
                "close": close,
                "volume": 1000.0,
            }
        )
        for i, p in enumerate(path):
            minute_rows.append(
                {
                    "date": bstart + pd.Timedelta(minutes=i),
                    "open": float(path[i - 1]) if i else o,
                    "high": float(max(p, path[i - 1] if i else o)),
                    "low": float(min(p, path[i - 1] if i else o)),
                    "close": float(p),
                    "volume": 10.0,
                }
            )
        price = close

    minute_df = pd.DataFrame(minute_rows)
    base_df = pd.DataFrame(base_rows)
    return minute_df, base_df


def _strategy_for(minute_df: pd.DataFrame, base_df: pd.DataFrame, pair: str):
    strategy = object.__new__(TurtleStrategyOptV9_7)
    strategy.dp = _FakeDP(minute_df, base_df, pair)
    strategy._minute_atr_cache = {}
    strategy._historical_minute_atr_cache = {}
    return strategy


def _live_atr_at(minute_df, base_df, pair, when: datetime):
    """Run the live path with only the data that was known at ``when``."""
    known_minutes = minute_df[minute_df["date"] < when].copy()
    known_base = base_df[base_df["date"] < when].copy()
    if len(known_minutes) < 30 or len(known_base) < 20:
        return None
    strategy = _strategy_for(known_minutes, known_base, pair)
    return strategy._latest_updated_4h_atr(pair, when)


def _backtest_atr_at(minute_df, base_df, pair, when: datetime):
    strategy = _strategy_for(minute_df, base_df, pair)
    return strategy._historical_minute_atr(pair, when)


def _sample_times(minute_df: pd.DataFrame):
    """Times spread across several 4h buckets, including bucket boundaries."""
    times = []
    # skip the first buckets so ATR(14) has enough history
    for bucket in (20, 21, 25, 30, 40, 55):
        bucket_start = minute_df["date"].iloc[bucket * 240]
        for offset in (0, 1, 37, 119, 238):
            times.append(bucket_start + pd.Timedelta(minutes=offset))
    return [t.to_pydatetime() for t in times]


def test_backtest_and_live_atr_paths_agree():
    """The two ATR implementations must return the same number.

    A mismatch would mean the backtest measures a strategy that cannot exist
    in production.
    """
    pair = "TEST/USDT"
    minute_df, base_df = _synthetic_pair(pair)
    times = _sample_times(minute_df)

    compared = 0
    for when in times:
        backtest = _backtest_atr_at(minute_df, base_df, pair, when)
        live = _live_atr_at(minute_df, base_df, pair, when)
        assert backtest is not None, f"backtest ATR missing at {when}"
        assert live is not None, f"live ATR missing at {when}"
        assert backtest == pytest.approx(live, rel=1e-9), (
            f"ATR paths diverge at {when}: backtest={backtest} live={live}"
        )
        compared += 1

    assert compared >= 20


def test_neither_atr_path_becomes_silently_missing():
    """A path returning None while the other works would silently widen stops."""
    pair = "TEST/USDT"
    minute_df, base_df = _synthetic_pair(pair)
    for when in _sample_times(minute_df):
        backtest = _backtest_atr_at(minute_df, base_df, pair, when)
        live = _live_atr_at(minute_df, base_df, pair, when)
        assert (backtest is None) == (live is None), (
            f"one path is missing while the other is not at {when}: "
            f"backtest={backtest} live={live}"
        )


def test_stale_minute_logged_with_age_once(caplog):
    pair = "TEST/USDT"
    minutes, base = _synthetic_pair(pair)
    strategy = _strategy_for(minutes, base, pair)
    now = minutes["date"].iloc[-1] + pd.Timedelta(minutes=5)
    with caplog.at_level("INFO"):
        for _ in range(2):
            assert strategy._latest_updated_4h_atr(pair, now) is None
    messages = [
        record.message for record in caplog.records if "reason=stale_minute" in record.message
    ]
    assert len(messages) == 1
    assert "age_seconds=240.0" in messages[0]
