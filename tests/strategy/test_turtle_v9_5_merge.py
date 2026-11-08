"""Guard against lookahead in the 4h -> 15m informative merge."""

import pandas as pd
import pytest

from freqtrade.strategy import merge_informative_pair


def test_informative_merge_never_exposes_an_unclosed_4h_candle():
    """A 15m row may only see a 4h candle that has already closed."""
    base = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC"),
            "open": [100.0] * 6,
            "high": [100.0] * 6,
            "low": [100.0] * 6,
            "close": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "volume": [1.0] * 6,
        }
    )
    rows = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    dataframe = pd.DataFrame(
        {
            "date": rows,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )

    merged = merge_informative_pair(dataframe, base, "15m", "4h", ffill=True)

    for _, row in merged.iterrows():
        row_close = row["date"] + pd.Timedelta(minutes=15)
        informative_date = row["date_4h"]
        if pd.isna(informative_date):
            continue
        # The 4h candle is only usable once its own close time has passed.
        assert row_close >= informative_date + pd.Timedelta(hours=4), (
            f"lookahead: 15m row closing {row_close} sees 4h candle "
            f"{informative_date} that had not closed yet"
        )


def test_informative_merge_exposes_the_previous_candle_not_the_current_one():
    """During a 4h candle, the merged value must come from the prior candle."""
    base = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"),
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [10.0, 20.0, 30.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    # A 15m row inside the second 4h candle. The frame needs several rows
    # because merge_ordered only backfills the leading rows when it has more
    # than one row to work with.
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=24, freq="15min", tz="UTC"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )

    merged = merge_informative_pair(dataframe, base, "15m", "4h", ffill=True)

    row = merged[merged["date"] == pd.Timestamp("2024-01-01 04:30", tz="UTC")].iloc[0]
    assert row["date_4h"] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert row["close_4h"] == 10.0
