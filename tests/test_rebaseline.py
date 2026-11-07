"""Phase 3 tests: realistic fills, one position per ticker, null baselines, universe pinning."""

import numpy as np
import pandas as pd
import pytest

from finance_vibe import backtest_baselines as bb
from finance_vibe import coiled_cobra_backtest as cb
from finance_vibe.pipeline_backtest import simulate_trade_gap_aware


def _bars(rows):
    """(open, high, low, close) tuples with a leading pad bar."""
    rows = [(100.0, 100.0, 100.0, 100.0)] + rows
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=len(rows), freq="W-MON"),
        "Open": [r[0] for r in rows], "High": [r[1] for r in rows],
        "Low": [r[2] for r in rows], "Close": [r[3] for r in rows],
    })


def _sim(rows, **kw):
    return simulate_trade_gap_aware(
        _bars(rows), 1, is_long=True, entry=100.0, stop=96.0, target1=104.0, target2=108.0,
        entry_valid_bars=kw.get("entry_valid", 2), max_hold_bars=kw.get("max_hold", 10),
        slippage_pct=kw.get("slippage", 0.0),
    )


# ---------------------------------------------------------------------------
# F-08 gap-aware fills
# ---------------------------------------------------------------------------

def test_gap_through_stop_exits_at_open_beyond_minus_one_r():
    out, _, px, r, _ = _sim([(101, 102, 99.5, 101),   # fill at 100
                             (90, 92, 88, 91)])        # gaps below the 96 stop
    assert out == "stopped" and px == 90 and r == pytest.approx(-2.5)


def test_gap_down_entry_fills_at_open():
    out, _, px, r, _ = _sim([(98, 99, 97, 98.5),       # opens below entry -> fill at 98
                             (99, 110, 98, 109)])      # target2 next bar
    assert out == "target2"
    assert r == pytest.approx((108 - 98) / 4.0)


def test_no_target_exit_on_fill_bar():
    # Fill bar touches entry and also prints above T2: target must NOT count on that bar.
    out, _, _, _, exit_idx = _sim([(105, 110, 99, 103),
                                   (103, 103.5, 101, 102)], max_hold=2)
    assert out == "expired" and exit_idx == 2


def test_stop_on_fill_bar_is_pessimistic():
    out, _, px, r, _ = _sim([(101, 110, 95, 100)])
    assert out == "stopped" and px == 96 and r == pytest.approx(-1.0)


def test_slippage_worsens_fills():
    _, _, _, r0, _ = _sim([(101, 102, 99.5, 101), (99, 99, 95, 96)])
    _, _, _, r1, _ = _sim([(101, 102, 99.5, 101), (99, 99, 95, 96)], slippage=0.001)
    assert r1 < r0


# ---------------------------------------------------------------------------
# F-09 one position per ticker
# ---------------------------------------------------------------------------

def _setup_row():
    return {"Symbol": "TEST", "Setup Type": "SETUP_LONG", "Close": 100.0, "EMA20": 98.0,
            "EMA50": 95.0, "ATR": 4.0, "Swing Low": 94.0, "Fib 78.6%": 96.0, "Score": 85,
            "Grade": "A", "Checks Met": "5/6", "Source": "coiled_cobra", "Pct_From_EMA20": 0.02,
            "Pct_From_EMA50": 0.05, "Pct_From_Fib618": 0.01, "Pct_From_Fib786": 0.04, "ATR_Pct": 0.04}


def _flat_csv(tmp_path, n=80):
    path = tmp_path / "TEST_10y_1wk.csv"
    pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="W-MON"),
        "Open": [100.0] * n, "High": [101.0] * n, "Low": [99.0] * n,
        "Close": [100.0] * n, "Volume": [1e6] * n,
    }).to_csv(path, index=False)
    return path


def test_one_position_blocks_overlapping_signals(tmp_path, monkeypatch):
    path = _flat_csv(tmp_path)
    monkeypatch.setattr(cb, "detect_cobra_setup_at_bar", lambda w, s, benchmark_df=None: _setup_row())

    overlap, _ = cb.backtest_ticker(str(path), entry_valid=2, max_hold=4, one_position=False)
    single, _ = cb.backtest_ticker(str(path), entry_valid=2, max_hold=4, one_position=True)
    assert len(single) < len(overlap)

    t = pd.DataFrame(single)
    sig = pd.to_datetime(t["Signal Date"])
    ex = pd.to_datetime(t["Exit Date"])
    assert (sig.iloc[1:].to_numpy() > ex.iloc[:-1].to_numpy()).all()


# ---------------------------------------------------------------------------
# Random-entry null
# ---------------------------------------------------------------------------

def _random_walk_csv(tmp_path, n=300, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.04, n)))
    path = tmp_path / "RW_10y_1wk.csv"
    pd.DataFrame({
        "Date": pd.date_range("2018-01-01", periods=n, freq="W-MON"),
        "Open": close * (1 + rng.normal(0, 0.005, n)), "High": close * 1.03,
        "Low": close * 0.97, "Close": close, "Volume": np.full(n, 1e6),
    }).to_csv(path, index=False)
    return path


def test_random_entry_trades_deterministic_and_sized(tmp_path):
    path = _random_walk_csv(tmp_path)
    a = cb.random_entry_trades(str(path), 20, entry_valid=4, max_hold=12, seed=3)
    b = cb.random_entry_trades(str(path), 20, entry_valid=4, max_hold=12, seed=3)
    c = cb.random_entry_trades(str(path), 20, entry_valid=4, max_hold=12, seed=4)
    assert len(a) == 20
    assert [r["Signal Date"] for r in a] == [r["Signal Date"] for r in b]
    assert [r["Signal Date"] for r in a] != [r["Signal Date"] for r in c]
    first_allowed = pd.read_csv(path)["Date"].iloc[cb.LOOKBACK // 2 + 15]
    assert min(r["Signal Date"] for r in a) >= first_allowed


def test_random_entry_levels_match_planner_logic(tmp_path):
    path = _random_walk_csv(tmp_path)
    for r in cb.random_entry_trades(str(path), 10, entry_valid=4, max_hold=12):
        assert r["Stock Stop"] < r["Stock Entry"] < r["Target 1"] < r["Target 2"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_mean_ci_brackets_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(0.5, 1, 600)
    c = np.repeat(np.arange(60), 10)
    out = bb.cluster_bootstrap_mean(v, c)
    assert out["lo"] < out["mean"] < out["hi"] and out["clusters"] == 60


def test_cluster_bootstrap_diff_null_contains_zero_and_detects_edge():
    rng = np.random.default_rng(1)
    c = np.repeat(np.arange(80), 20)
    a, b = rng.normal(0, 1, 1600), rng.normal(0, 1, 1600)
    null = bb.cluster_bootstrap_diff(a, c, b, c)
    assert null["lo"] < 0 < null["hi"]
    edge = bb.cluster_bootstrap_diff(a + 0.5, c, b, c)
    assert edge["lo"] > 0


def test_forward_return_baselines_detects_selective_signals():
    dates = pd.date_range("2020-01-06", periods=200, freq="W-MON")
    rng = np.random.default_rng(2)
    closes = {s: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 200))), index=dates)
              for s in ("A", "B", "C", "D")}
    rows = []
    for s, c in closes.items():
        fr = bb.forward_returns(c, (5,))[5].dropna()
        for d in fr.nlargest(15).index:  # "signals" that pick the best weeks
            rows.append({"Symbol": s, "Signal Date": d, "Forward_Return_5w": fr[d]})
    out = bb.forward_return_baselines(pd.DataFrame(rows), closes.get, None, horizons=(5,))
    assert out.loc[0, "excess_vs_random"] > 0 and out.loc[0, "excess_vs_random_lo"] > 0


def test_r_per_signal_counts_no_fill_as_zero():
    df = pd.DataFrame({"Outcome": ["no_fill", "stopped", "target1"], "R Multiple": [None, -1.0, 2.0]})
    assert list(bb.r_per_signal(df)) == [0.0, -1.0, 2.0]


def test_promotion_gate():
    fwd = pd.DataFrame({"horizon_bars": [13], "excess_vs_random_lo": [0.01]})
    assert bb.promotion_gate({"lo": 0.05}, fwd, 13)["pass"] is True
    bad = bb.promotion_gate({"lo": -0.02}, fwd.assign(excess_vs_random_lo=[-0.01]), 13)
    assert bad["pass"] is False and len(bad["reasons"]) == 2


# ---------------------------------------------------------------------------
# F-04 universe pinning
# ---------------------------------------------------------------------------

def test_resolve_universe_variants(tmp_path):
    assert cb.resolve_universe(None)[0] is None
    manifest, desc = cb.resolve_universe("manifest")
    assert "AAPL" in manifest and "QQQ" in manifest and "manifest" in desc
    f = tmp_path / "u.csv"
    pd.Series(["aapl", "MSFT"], name="Ticker").to_csv(f, index=False)
    assert cb.resolve_universe(str(f))[0] == {"AAPL", "MSFT"}


def test_save_universe_writes_dated_copy(tmp_path, monkeypatch):
    from finance_vibe import ticker_provider as tp
    monkeypatch.setattr(tp.config, "TICKER_LIST_PATH", str(tmp_path / "active_tickers.csv"))
    dated = tp.save_universe(["AAPL", "MSFT"], today="2026-11-07")
    assert dated.endswith("universes/active_tickers_2026-11-07.csv")
    assert pd.read_csv(dated)["Ticker"].tolist() == ["AAPL", "MSFT"]
    assert pd.read_csv(tmp_path / "active_tickers.csv")["Ticker"].tolist() == ["AAPL", "MSFT"]
