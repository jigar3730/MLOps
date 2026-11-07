"""Regression tests for weekly-bar completeness, calendar alignment, and manifest parsing."""

import pandas as pd
import pytest

from finance_vibe import data_ingestor as di
from finance_vibe import ticker_provider as tp


def _weekly(mondays):
    idx = pd.DatetimeIndex(pd.to_datetime(mondays), name="Date")
    n = len(idx)
    return pd.DataFrame(
        {"Open": [1.0] * n, "High": [2.0] * n, "Low": [0.5] * n, "Close": [1.5] * n, "Volume": [10] * n},
        index=idx,
    )


# ---------------------------------------------------------------------------
# F-01: last weekly bar is kept once its week has closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "now, expect_last",
    [
        ("2026-11-07 10:00", "2026-11-02"),  # Saturday: Nov 2-6 week closed -> keep
        ("2026-11-06 16:00", "2026-11-02"),  # exactly at Friday close -> keep
        ("2026-11-06 15:59", "2026-10-26"),  # Friday before close -> drop partial
        ("2026-11-04 12:00", "2026-10-26"),  # mid-week -> drop partial
    ],
)
def test_drop_incomplete_weekly_bar_uses_friday_close(now, expect_last):
    df = _weekly(["2026-10-19", "2026-10-26", "2026-11-02"])
    out = di.drop_incomplete_weekly_bar(df, now=pd.Timestamp(now))
    assert out.index[-1] == pd.Timestamp(expect_last)


def test_drop_incomplete_weekly_bar_handles_tz_aware_now():
    df = _weekly(["2026-10-26", "2026-11-02"])
    # 2026-11-06 21:30 UTC == 16:30 New York (EST) -> week closed.
    out = di.drop_incomplete_weekly_bar(df, now=pd.Timestamp("2026-11-06 21:30", tz="UTC"))
    assert out.index[-1] == pd.Timestamp("2026-11-02")


def test_drop_incomplete_weekly_bar_empty_frame():
    empty = _weekly([])
    assert di.drop_incomplete_weekly_bar(empty, now=pd.Timestamp("2026-11-07")).empty


# ---------------------------------------------------------------------------
# F-13: weekly calendar alignment
# ---------------------------------------------------------------------------

def test_week_start_and_monday_anchor_detection():
    assert di.week_start("2026-10-27") == pd.Timestamp("2026-10-26")  # Tue -> Mon
    assert di.week_start("2026-10-26") == pd.Timestamp("2026-10-26")
    assert di.is_monday_anchored(pd.to_datetime(["2026-10-19", "2026-10-26"]))
    assert not di.is_monday_anchored(pd.to_datetime(["2026-10-20", "2026-10-27"]))


def test_resample_daily_to_weekly_monday_labels_and_ohlcv():
    days = pd.bdate_range("2026-10-19", "2026-10-30")  # two full Mon-Fri weeks
    daily = pd.DataFrame(
        {
            "Open": range(1, 11),
            "High": [x + 0.5 for x in range(1, 11)],
            "Low": [x - 0.5 for x in range(1, 11)],
            "Close": [x + 0.25 for x in range(1, 11)],
            "Volume": [100] * 10,
        },
        index=days.tz_localize("America/New_York"),
    )
    w = di.resample_daily_to_weekly(daily)
    assert list(w.index) == [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-26")]
    assert w.index.tz is None
    first = w.iloc[0]
    assert first["Open"] == 1 and first["Close"] == 5.25
    assert first["High"] == 5.5 and first["Low"] == 0.5
    assert first["Volume"] == 500


# ---------------------------------------------------------------------------
# F-05: manifest parsing keeps the first ticker and manifest order
# ---------------------------------------------------------------------------

def test_load_manifest_without_header_keeps_first_row(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("AAPL\nMSFT\nnvda\n\nBRK.B\n^VIX\nMSFT\n")
    assert tp.load_manifest(str(p)) == ["AAPL", "MSFT", "NVDA", "BRK-B"]


def test_load_manifest_with_header_skips_header(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("Symbol\nAAPL\nMSFT\n")
    assert tp.load_manifest(str(p)) == ["AAPL", "MSFT"]


def test_shipped_manifest_includes_its_first_line():
    with open(tp.MANIFEST_PATH, encoding="utf-8") as fh:
        first = fh.readline().strip().upper()
    assert tp.load_manifest()[0] == first
