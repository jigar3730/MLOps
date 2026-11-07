"""Phase 2 data-contract tests: plausibility, freshness, snapshots, loud failures."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe import data_ingestor as di
from finance_vibe import run_vibe


def _frame(n=120, close=50.0, volume=1_000_000, start="2024-01-01", freq="W-MON"):
    dates = pd.date_range(start, periods=n, freq=freq)
    c = np.full(n, close, dtype=float)
    return pd.DataFrame({
        "Date": dates, "Open": c, "High": c * 1.02, "Low": c * 0.98, "Close": c,
        "Volume": np.full(n, volume, dtype=float),
    })


# ---------------------------------------------------------------------------
# F-07 plausibility
# ---------------------------------------------------------------------------

def test_plausibility_accepts_liquid_normal_series():
    v = config.assess_plausibility(_frame(), "1wk")
    assert v == {"reject": [], "warn": []}


def test_plausibility_rejects_penny_stock():
    v = config.assess_plausibility(_frame(close=2.5, volume=50_000_000), "1wk")
    assert any(r.startswith("price_below_min") for r in v["reject"])


def test_plausibility_rejects_illiquid():
    # $50 × 100k shares / 5 days = $1M/day < $4M/day
    v = config.assess_plausibility(_frame(volume=100_000), "1wk")
    assert any(r.startswith("illiquid") for r in v["reject"])


def test_plausibility_liquidity_is_per_day_for_daily_bars():
    # $50 × 100k shares = $5M/day on daily bars -> passes (same volume fails weekly).
    v = config.assess_plausibility(_frame(volume=100_000, freq="B"), "1d")
    assert not any(r.startswith("illiquid") for r in v["reject"])


def test_plausibility_rejects_zero_volume_share():
    df = _frame(volume=10_000_000)
    df.loc[df.index[-10:], "Volume"] = 0  # 10 of last 52 weekly bars
    v = config.assess_plausibility(df, "1wk")
    assert any(r.startswith("zero_volume_share") for r in v["reject"])


def test_plausibility_rejects_corrupt_rows():
    df = _frame(volume=10_000_000)
    df.loc[5, "High"] = df.loc[5, "Low"] - 1
    v = config.assess_plausibility(df, "1wk")
    assert any(r.startswith("corrupt_ohlc_rows:1") for r in v["reject"])


def test_plausibility_warns_but_keeps_extreme_move():
    df = _frame(volume=10_000_000)
    df.loc[60:, ["Open", "High", "Low", "Close"]] *= 15  # one ×15 jump
    v = config.assess_plausibility(df, "1wk")
    assert v["reject"] == []
    assert v["warn"] == ["extreme_moves:1"]


# ---------------------------------------------------------------------------
# F-23 freshness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("now, expected", [
    ("2026-11-07 10:00", "2026-11-02"),   # Saturday
    ("2026-11-06 16:00", "2026-11-02"),   # Friday at close
    ("2026-11-06 15:00", "2026-10-26"),   # Friday before close
    ("2026-11-02 09:00", "2026-10-26"),   # Monday morning
])
def test_expected_last_bar_weekly(now, expected):
    assert config.expected_last_bar("1wk", pd.Timestamp(now)) == pd.Timestamp(expected)


@pytest.mark.parametrize("now, expected", [
    ("2026-11-06 17:00", "2026-11-06"),   # Friday after close
    ("2026-11-06 10:00", "2026-11-05"),   # Friday before close
    ("2026-11-08 12:00", "2026-11-06"),   # Sunday -> Friday
    ("2026-11-09 09:00", "2026-11-06"),   # Monday pre-close -> Friday
])
def test_expected_last_bar_daily(now, expected):
    assert config.expected_last_bar("1d", pd.Timestamp(now)) == pd.Timestamp(expected)


def test_staleness_bars():
    now = pd.Timestamp("2026-11-07 10:00")
    assert config.staleness_bars("2026-11-02", "1wk", now) == 0
    assert config.staleness_bars("2026-10-26", "1wk", now) == 1
    assert config.staleness_bars("2026-11-06", "1d", now) == 0
    assert config.staleness_bars("2026-11-04", "1d", now) == 2


# ---------------------------------------------------------------------------
# Ingestor: plausibility gate + benchmark freshness gate (no network)
# ---------------------------------------------------------------------------

def _yf_frame(n, close, volume, last_monday):
    idx = pd.date_range(end=last_monday, periods=n, freq="W-MON", name="Date")
    c = np.full(n, close, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.02, "Low": c * 0.98, "Close": c,
                         "Volume": np.full(n, volume, dtype=float)}, index=idx)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    cfg = di.config  # the module object the ingestor actually uses
    raw, logs = tmp_path / "raw", tmp_path / "logs"
    raw.mkdir(); logs.mkdir()
    tickers = tmp_path / "active_tickers.csv"
    monkeypatch.setitem(cfg.TIMEFRAME_PROFILES, "weekly",
                        {**cfg.TIMEFRAME_PROFILES["weekly"], "raw_dir": str(raw), "logs_dir": str(logs)})
    monkeypatch.setattr(cfg, "TICKER_LIST_PATH", str(tickers))
    return tmp_path, tickers, raw, logs


def _run_ingest(monkeypatch, frames, now="2026-11-07 10:00"):
    monkeypatch.setattr(di.yf, "download", lambda t, **kw: frames[t].copy())
    real_now = pd.Timestamp(now, tz="America/New_York").to_pydatetime()

    class _FixedDT(di.datetime):
        @classmethod
        def now(cls, tz=None):
            return real_now if tz else real_now.replace(tzinfo=None)

    monkeypatch.setattr(di, "datetime", _FixedDT)
    return di.ingest_market_data("weekly")


def test_ingest_rejects_penny_and_keeps_good(sandbox, monkeypatch):
    _, tickers, raw, logs = sandbox
    pd.Series(["QQQ", "GOOD", "PENY"], name="Ticker").to_csv(tickers, index=False)
    frames = {
        "QQQ": _yf_frame(120, 500.0, 5e7, "2026-11-02"),
        "GOOD": _yf_frame(120, 80.0, 5e6, "2026-11-02"),
        "PENY": _yf_frame(120, 0.5, 5e9, "2026-11-02"),
    }
    assert _run_ingest(monkeypatch, frames) is True
    saved = sorted(p.name.split("_")[0] for p in raw.glob("*.csv"))
    assert saved == ["GOOD", "QQQ"]
    errs = pd.read_csv(next(logs.glob("ingest_errors_*.csv")))
    assert errs.loc[errs.Ticker == "PENY", "Error"].str.startswith("reject:price_below_min").all()


def test_ingest_fails_when_benchmark_stale(sandbox, monkeypatch):
    _, tickers, raw, logs = sandbox
    pd.Series(["QQQ"], name="Ticker").to_csv(tickers, index=False)
    frames = {"QQQ": _yf_frame(120, 500.0, 5e7, "2026-10-19")}  # 2 weeks behind
    assert _run_ingest(monkeypatch, frames) is False
    errs = pd.read_csv(next(logs.glob("ingest_errors_*.csv")))
    assert errs.Error.str.startswith("warn:stale_last_bar").any()


def test_ingest_fails_when_benchmark_missing(sandbox, monkeypatch):
    _, tickers, *_ = sandbox
    pd.Series(["GOOD"], name="Ticker").to_csv(tickers, index=False)
    frames = {"GOOD": _yf_frame(120, 80.0, 5e6, "2026-11-02")}
    assert _run_ingest(monkeypatch, frames) is False


# ---------------------------------------------------------------------------
# F-12 raw snapshots
# ---------------------------------------------------------------------------

def test_snapshot_raw_folder_archives_and_prunes(tmp_path):
    raw = tmp_path / "data" / "raw" / "weekly"
    raw.mkdir(parents=True)
    (raw / "AAA_10y_1wk.csv").write_text("Date,Close\n2026-11-02,1\n")
    (raw / "BBB_10y_1wk.csv").write_text("Date,Close\n2026-11-02,2\n")

    snap = run_vibe.snapshot_raw_folder(tmp_path, "weekly", keep=2)
    assert snap is not None and snap.exists()
    with zipfile.ZipFile(snap) as zf:
        assert sorted(zf.namelist()) == ["AAA_10y_1wk.csv", "BBB_10y_1wk.csv"]

    snap_dir = snap.parent
    for stamp in ("2020-01-01_0000", "2020-01-02_0000", "2020-01-03_0000"):
        (snap_dir / f"raw_weekly_{stamp}.zip").write_bytes(b"")
    run_vibe.snapshot_raw_folder(tmp_path, "weekly", keep=2)
    remaining = sorted(p.name for p in snap_dir.glob("raw_weekly_*.zip"))
    assert len(remaining) == 2 and snap.name in remaining


def test_snapshot_raw_folder_noop_when_empty(tmp_path):
    assert run_vibe.snapshot_raw_folder(tmp_path, "weekly") is None


# ---------------------------------------------------------------------------
# F-15 loud failures
# ---------------------------------------------------------------------------

def test_backtest_worker_reports_load_error(tmp_path):
    from finance_vibe import coiled_cobra_backtest as cb
    missing = tmp_path / "NOPE_10y_1wk.csv"
    symbol, rows, error = cb._backfill_ticker_worker(str(missing))
    assert symbol == "NOPE" and rows == []
    assert error and error.startswith("load:")


def test_write_failures_creates_audit_file(tmp_path):
    from finance_vibe import coiled_cobra_backtest as cb
    path = cb._write_failures(str(tmp_path), "backtest", [{"Symbol": "X", "Error": "boom"}])
    assert pd.read_csv(path).to_dict("records") == [{"Symbol": "X", "Error": "boom"}]
    assert cb._write_failures(str(tmp_path), "backtest", []) is None
