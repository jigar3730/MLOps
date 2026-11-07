"""Coiled Cobra historical backfill and walk-forward backtest.

This module evaluates the Coiled Cobra strategy across historical raw OHLC data,
exports signal archives, and simulates stock trade outcomes using the existing
trade-planner stock level calculator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # run as a script: make `finance_vibe` importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finance_vibe import config
from finance_vibe.analysis_engine import load_benchmark_frame, load_ohlc_csv, ticker_from_filename
from finance_vibe.backtest_baselines import (
    cluster_bootstrap_diff,
    cluster_bootstrap_mean,
    forward_return_baselines,
    promotion_gate,
    r_per_signal,
)
from finance_vibe.coiled_cobra import (
    BENCHMARK,
    LOOKBACK,
    add_macro_indicators,
    evaluate_coiled_cobra,
    local_swing_low,
)
from finance_vibe.pipeline_backtest import _fmt_date, simulate_trade_gap_aware
from finance_vibe.trade_planner import calculate_stock_levels


# Data-shape errors expected from short/degenerate windows (e.g. pandas_ta
# returns None for too-short series). Anything else is a bug and must surface.
_DATA_ERRORS = (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError)


def _fmt_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _write_failures(logs_dir: str, kind: str, failures: list[dict]) -> Optional[str]:
    """Persist per-ticker worker failures so dropped tickers are auditable."""
    if not failures:
        return None
    path = os.path.join(logs_dir, f"coiled_cobra_{kind}_errors_{datetime.now():%Y-%m-%d}.csv")
    pd.DataFrame(failures).to_csv(path, index=False)
    return path


def detect_cobra_setup_at_bar(
    df: pd.DataFrame,
    symbol: str,
    benchmark_df=None,
) -> Optional[dict]:
    """Evaluate the latest bar in a history window for a Coiled Cobra coil setup."""
    if len(df) < LOOKBACK // 2 + 15:
        return None

    window = df.copy()
    try:
        window = add_macro_indicators(window)
    except _DATA_ERRORS:
        return None

    if len(window) < 2:
        return None

    setup = evaluate_coiled_cobra(window, benchmark_df)
    if not setup:
        return None

    latest = window.iloc[-1]
    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    atr = float(latest["ATR"])
    fib_618 = float(latest["Fib_618"]) if pd.notna(latest.get("Fib_618")) else None
    fib_786 = float(latest["Fib_786"]) if pd.notna(latest.get("Fib_786")) else None

    return {
        "Symbol": symbol.upper(),
        "Date": latest["Date"],
        "Setup Type": "SETUP_LONG",
        "Close": round(close, 2),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "ATR": round(atr, 2),
        "Swing Low": round(local_swing_low(window), 2),
        "Fib 61.8%": round(fib_618, 2) if fib_618 is not None else None,
        "Fib 78.6%": round(fib_786, 2) if fib_786 is not None else None,
        "Pct_From_EMA20": round((close - ema20) / ema20, 4),
        "Pct_From_EMA50": round((close - ema50) / ema50, 4),
        "Pct_From_Fib618": round((close - fib_618) / fib_618, 4) if fib_618 is not None else None,
        "Pct_From_Fib786": round((close - fib_786) / fib_786, 4) if fib_786 is not None else None,
        "ATR_Pct": round(atr / close, 4),
        "Score": setup["Score"],
        "Grade": setup["Grade"],
        "Checks Met": setup["Checks Met"],
        "Source": "coiled_cobra",
        "RS 63d": setup.get("RS 63d"),
    }


def generate_backfill(mode: str = "weekly", tickers: Optional[str] = None) -> pd.DataFrame:
    """Scan historical raw CSVs to produce a Coiled Cobra signal archive."""
    mode_cfg = config.get_mode_config(mode)
    raw_dir = mode_cfg["raw_dir"]
    logs_dir = mode_cfg["logs_dir"]
    ticker_filter = None
    if tickers:
        ticker_filter = {t.strip().upper() for t in tickers.split(",") if t.strip()}

    file_paths = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    )
    work_paths = [
        path
        for path in file_paths
        if not ticker_filter or ticker_from_filename(path) in ticker_filter
    ]

    max_workers = os.cpu_count() or 1
    print(f"--- Coiled Cobra Backfill [{mode.upper()}] ---")
    print(f"Raw dir: {raw_dir}")
    print(f"Workers: {max_workers}  Tickers: {len(work_paths)}\n")

    results_by_path: dict[str, list[dict]] = {}
    failures: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_cobra_worker,
        initargs=(mode,),
    ) as ex:
        futures = {ex.submit(_backfill_ticker_worker, path): path for path in work_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            symbol = ticker_from_filename(path)
            try:
                symbol, rows, error = fut.result()
            except Exception as exc:  # noqa: BLE001 - process crash / pickling boundary
                rows, error = [], f"worker: {_fmt_exc(exc)}"
            if error:
                print(f"{symbol}: error — {error}", file=sys.stderr)
                failures.append({"Symbol": symbol, "Error": error})
            results_by_path[path] = rows
            if rows:
                print(f"{symbol}: {len(rows)} setup(s)")

    err_path = _write_failures(logs_dir, "backfill", failures)
    if err_path:
        print(f"⚠️ {len(failures)} ticker(s) failed; details: {err_path}")

    # Reassemble in sorted path order so CSV row sequence matches single-threaded runs.
    all_rows: list[dict] = []
    for path in work_paths:
        all_rows.extend(results_by_path.get(path, []))

    out_df = pd.DataFrame(all_rows)
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(logs_dir, f"coiled_cobra_backfill_{stamp}.csv")
    os.makedirs(logs_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved Coiled Cobra backfill archive: {out_path}")
    return out_df


FORWARD_HORIZONS = (2, 5, 13, 26)
# Random-entry null draws per real signal (per ticker); more draws = tighter null.
RANDOM_ENTRY_MULT = 3


def _simulate(df, start_idx, entry, stop, t1, t2, entry_valid, max_hold, slippage_pct):
    """Run one long trade; returns (outcome, exit_date, exit_price, r_mult, exit_idx)."""
    return simulate_trade_gap_aware(
        df, start_idx, is_long=True, entry=entry, stop=stop, target1=t1, target2=t2,
        entry_valid_bars=entry_valid, max_hold_bars=max_hold, slippage_pct=slippage_pct,
    )


def _trade_record(symbol, df, idx, setup_row, levels, sim) -> dict:
    """Build one output row (shared by real signals and random-entry nulls)."""
    entry, stop, t1, t2 = levels
    outcome, exit_date, exit_price, r_mult, _ = sim
    setup_close = float(setup_row["Close"])
    n_bars = len(df)

    def _fwd(h: int):
        j = idx + h
        if j >= n_bars:
            return None
        return round((float(df.iloc[j]["Close"]) - setup_close) / setup_close, 4)

    r_multiple = round(r_mult, 2) if r_mult is not None else None
    if outcome == "no_fill":
        target_label = None
    elif outcome in ("target1", "target2"):
        target_label = 1
    else:
        target_label = 0

    row = {
        "Symbol": symbol,
        "Signal Date": _fmt_date(df.iloc[idx]["Date"]),
        "Setup Type": setup_row["Setup Type"],
        "Score": setup_row.get("Score"),
        "Grade": setup_row.get("Grade"),
        "Pct_From_EMA20": setup_row.get("Pct_From_EMA20"),
        "Pct_From_EMA50": setup_row.get("Pct_From_EMA50"),
        "Pct_From_Fib618": setup_row.get("Pct_From_Fib618"),
        "Pct_From_Fib786": setup_row.get("Pct_From_Fib786"),
        "ATR_Pct": setup_row.get("ATR_Pct"),
        "Stock Entry": round(entry, 2),
        "Stock Stop": round(stop, 2),
        "Target 1": round(t1, 2),
        "Target 2": round(t2, 2),
        "Outcome": outcome,
        "Exit Date": _fmt_date(exit_date),
        "Exit Price": round(exit_price, 2) if exit_price is not None else None,
        "R Multiple": r_multiple,
        "Target_Label": target_label,
        "Target_R_Mult": r_multiple,
    }
    for h in FORWARD_HORIZONS:
        row[f"Forward_Return_{h}w"] = _fwd(h)
    return row


def _next_free_idx(idx: int, sim, entry_valid: int) -> int:
    """First bar index at which a new position may be opened (one position per ticker)."""
    outcome, _, _, _, exit_idx = sim
    if outcome == "no_fill" or exit_idx is None:
        return idx + 1 + entry_valid  # pending limit order occupied the slot
    return exit_idx + 1


def backtest_ticker(
    path: str,
    entry_valid: int,
    max_hold: int,
    benchmark_df=None,
    *,
    one_position: bool = True,
    slippage_pct: float = config.BACKTEST_SLIPPAGE_PCT,
) -> tuple[list[dict], dict]:
    """Walk-forward simulate one ticker using Coiled Cobra signals (gap/slippage-aware fills).

    ``one_position`` ignores new signals while a trade or pending entry is open,
    so trades are not double-counted exposure. Set False to record every
    signal (e.g. to build an every-signal ML dataset).
    """
    symbol = ticker_from_filename(path)
    df = load_ohlc_csv(path)
    trades: list[dict] = []
    counts = {
        "signals": 0, "filled": 0, "no_fill": 0, "stopped": 0,
        "target1": 0, "target2": 0, "expired": 0, "errors": 0,
    }

    min_bars = LOOKBACK // 2 + 15
    next_free = 0
    for idx in range(min_bars, len(df) - 1):
        if one_position and idx < next_free:
            continue
        window = df.iloc[: idx + 1]
        setup_row = detect_cobra_setup_at_bar(window, symbol, benchmark_df=benchmark_df)
        if not setup_row:
            continue

        counts["signals"] += 1
        try:
            entry, stop, t1, t2, _, _ = calculate_stock_levels(setup_row)
            sim = _simulate(df, idx + 1, entry, stop, t1, t2, entry_valid, max_hold, slippage_pct)
        except _DATA_ERRORS:
            counts["errors"] += 1
            continue

        outcome = sim[0]
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome != "no_fill":
            counts["filled"] += 1
        trades.append(_trade_record(symbol, df, idx, setup_row, (entry, stop, t1, t2), sim))
        next_free = _next_free_idx(idx, sim, entry_valid)

    return trades, counts


def random_entry_trades(
    path: str,
    n_draws: int,
    entry_valid: int,
    max_hold: int,
    *,
    slippage_pct: float = config.BACKTEST_SLIPPAGE_PCT,
    seed: int = 0,
) -> list[dict]:
    """Null model: the strategy's level and exit logic applied at random bars.

    Indicators are computed once on the full series; every indicator used is
    causal (EMA/ATR/rolling max-min), so the value at bar *i* equals what a
    window ending at *i* would produce. Bars are drawn uniformly (without
    replacement) from the same eligible range the strategy scans.
    """
    if n_draws <= 0:
        return []
    symbol = ticker_from_filename(path)
    df = load_ohlc_csv(path)
    try:
        ind = add_macro_indicators(df)
    except _DATA_ERRORS:
        return []

    min_bars = LOOKBACK // 2 + 15
    eligible = ind.index[(ind.index >= min_bars) & (ind.index < len(df) - 1)].to_numpy()
    if len(eligible) == 0:
        return []
    rng = np.random.default_rng((zlib.crc32(symbol.encode()) ^ seed) & 0xFFFFFFFF)
    picks = np.sort(rng.choice(eligible, size=min(n_draws, len(eligible)), replace=False))

    rows: list[dict] = []
    for i in picks:
        i = int(i)
        bar = ind.loc[i]
        close, ema20, ema50, atr = (float(bar[k]) for k in ("Close", "EMA20", "EMA50", "ATR"))
        fib618 = float(bar["Fib_618"]) if pd.notna(bar.get("Fib_618")) else None
        fib786 = float(bar["Fib_786"]) if pd.notna(bar.get("Fib_786")) else None
        setup_row = {
            "Symbol": symbol, "Setup Type": "SETUP_LONG", "Source": "coiled_cobra",
            "Close": close, "EMA20": ema20, "EMA50": ema50, "ATR": atr,
            "Swing Low": local_swing_low(ind.loc[:i]),
            "Fib 61.8%": fib618, "Fib 78.6%": fib786,
            "Pct_From_EMA20": round((close - ema20) / ema20, 4),
            "Pct_From_EMA50": round((close - ema50) / ema50, 4),
            "ATR_Pct": round(atr / close, 4),
        }
        try:
            entry, stop, t1, t2, _, _ = calculate_stock_levels(setup_row)
            sim = _simulate(df, i + 1, entry, stop, t1, t2, entry_valid, max_hold, slippage_pct)
        except _DATA_ERRORS:
            continue
        rows.append(_trade_record(symbol, df, i, setup_row, (entry, stop, t1, t2), sim))
    return rows


# Per-process cache: loaded once in the worker initializer, never pickled from parent.
_WORKER_BENCHMARK_DF = None


def _init_cobra_worker(mode: str) -> None:
    """Load benchmark OHLC once per worker process."""
    global _WORKER_BENCHMARK_DF
    _WORKER_BENCHMARK_DF = load_benchmark_frame(BENCHMARK, mode)


def _backfill_ticker_worker(path: str) -> tuple[str, list[dict], Optional[str]]:
    """Process-pool worker: load ticker OHLC locally and collect Coiled Cobra setups.

    Returns ``(symbol, rows, error)``; ``error`` is None on success.
    """
    symbol = ticker_from_filename(path)
    try:
        df = load_ohlc_csv(path)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        return symbol, [], f"load: {_fmt_exc(exc)}"

    rows: list[dict] = []
    min_bars = LOOKBACK // 2 + 15
    try:
        for idx in range(min_bars, len(df)):
            setup = detect_cobra_setup_at_bar(
                df.iloc[: idx + 1], symbol, benchmark_df=_WORKER_BENCHMARK_DF
            )
            if setup:
                rows.append(setup)
    except Exception as exc:  # noqa: BLE001 - worker boundary; returned to parent and logged
        return symbol, [], f"scan: {_fmt_exc(exc)}"

    return symbol, rows, None


def _backtest_ticker_worker(
    path: str,
    entry_valid: int,
    max_hold: int,
    opts: Optional[dict] = None,
) -> tuple[str, list[dict], dict, list[dict], Optional[str]]:
    """Process-pool worker: walk-forward backtest plus its random-entry null.

    Returns ``(symbol, trades, counts, random_trades, error)``; ``error`` is None on success.
    """
    opts = opts or {}
    symbol = ticker_from_filename(path)
    slippage = opts.get("slippage_pct", config.BACKTEST_SLIPPAGE_PCT)
    try:
        trades, counts = backtest_ticker(
            path, entry_valid, max_hold, benchmark_df=_WORKER_BENCHMARK_DF,
            one_position=opts.get("one_position", True), slippage_pct=slippage,
        )
        n_draws = len(trades) * opts.get("random_mult", RANDOM_ENTRY_MULT)
        rand = random_entry_trades(
            path, n_draws, entry_valid, max_hold,
            slippage_pct=slippage, seed=opts.get("seed", 0),
        )
        return symbol, trades, counts, rand, None
    except Exception as exc:  # noqa: BLE001 - worker boundary; returned to parent and logged
        return symbol, [], {}, [], _fmt_exc(exc)


def resolve_universe(spec: Optional[str]) -> tuple[Optional[set[str]], str]:
    """Resolve ``--universe``: None (all raw files), ``manifest``, or a CSV path.

    ``manifest`` uses only the curated manifest + STATIC_TICKERS, excluding the
    day-specific screener names that bias a historical backtest toward today's
    most-active (often recently-run-up) stocks.
    """
    if not spec:
        return None, "all raw files (includes screener-discovered names)"
    if spec == "manifest":
        from finance_vibe.ticker_provider import load_manifest, _normalize_symbols
        syms = set(_normalize_symbols(load_manifest() + config.STATIC_TICKERS))
        return syms, f"manifest + STATIC_TICKERS ({len(syms)} symbols)"
    frame = pd.read_csv(spec)
    col = "Ticker" if "Ticker" in frame.columns else frame.columns[0]
    syms = set(frame[col].dropna().astype(str).str.upper().str.strip())
    return syms, f"file {spec} ({len(syms)} symbols)"


def _strategy_stats(df: pd.DataFrame) -> dict:
    """Headline stats for a trades frame (real or random-entry)."""
    if df.empty:
        return {"signals": 0}
    filled = df[df["Outcome"] != "no_fill"]
    rps = r_per_signal(df)
    ci = cluster_bootstrap_mean(rps, df["Symbol"].to_numpy())
    years = pd.to_datetime(df["Signal Date"]).dt.year
    by_year = pd.Series(rps).groupby(years.to_numpy()).agg(["count", "mean"]).round(3)
    return {
        "signals": int(len(df)),
        "tickers": int(df["Symbol"].nunique()),
        "fill_rate": round(len(filled) / len(df), 4),
        "win_rate_filled": round(float(filled["Outcome"].isin(["target1", "target2"]).mean()), 4)
        if len(filled) else None,
        "mean_r_filled": round(float(pd.to_numeric(filled["R Multiple"]).mean()), 4) if len(filled) else None,
        "r_per_signal": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ci.items()},
        "worst_r": round(float(pd.to_numeric(filled["R Multiple"]).min()), 2) if len(filled) else None,
        "by_year_r_per_signal": {int(y): {"n": int(r["count"]), "mean": float(r["mean"])}
                                 for y, r in by_year.iterrows()},
    }


def summarize_backtest(
    trades_df: pd.DataFrame,
    random_df: pd.DataFrame,
    load_close,
    benchmark_close: Optional[pd.Series],
    primary_horizon: int,
    start_bar: int,
) -> dict:
    """Strategy vs random-entry null vs benchmark, with ticker-clustered CIs and a gate."""
    r_diff = cluster_bootstrap_diff(
        r_per_signal(trades_df), trades_df["Symbol"].to_numpy(),
        r_per_signal(random_df), random_df["Symbol"].to_numpy(),
    ) if len(trades_df) and len(random_df) else {"diff": np.nan, "lo": np.nan, "hi": np.nan}
    fwd = forward_return_baselines(
        trades_df, load_close, benchmark_close,
        horizons=(2, 5, 13), start_bar=start_bar,
    ) if len(trades_df) else pd.DataFrame(columns=["horizon_bars", "excess_vs_random_lo"])
    gate = promotion_gate(r_diff, fwd, primary_horizon)
    return {
        "strategy": _strategy_stats(trades_df),
        "random_entry": _strategy_stats(random_df),
        "r_per_signal_vs_random": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_diff.items()},
        "forward_return_baselines": fwd.round(5).to_dict("records"),
        "primary_horizon_bars": primary_horizon,
        "gate": gate,
    }


def _print_summary(summary: dict) -> None:
    s, r, d = summary["strategy"], summary["random_entry"], summary["r_per_signal_vs_random"]
    print("\n=== Coiled Cobra vs null baselines ===")
    print(f"Universe: {summary['universe']}")
    print(f"Fills: gap/slippage-aware; one position per ticker: {summary['settings']['one_position']}")
    if not s.get("signals"):
        print("No signals.")
        return
    for name, x in (("Strategy", s), ("Random entry", r)):
        if not x.get("signals"):
            print(f"{name:13s}: no trades")
            continue
        rp = x["r_per_signal"]
        print(f"{name:13s}: signals={x['signals']:5d} fill={x['fill_rate']:.1%} "
              f"win(filled)={x['win_rate_filled']:.1%} R/fill={x['mean_r_filled']:+.3f} "
              f"R/signal={rp['mean']:+.3f} [{rp['lo']:+.3f}, {rp['hi']:+.3f}]")
    print(f"Edge R/signal vs random: {d['diff']:+.3f}  95% CI [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    for row in summary["forward_return_baselines"]:
        line = (f"  fwd {row['horizon_bars']:>2} bars: signal {row['signal_mean']:+.4f} | random "
                f"{row['random_mean']:+.4f} | excess {row['excess_vs_random']:+.4f} "
                f"[{row['excess_vs_random_lo']:+.4f}, {row['excess_vs_random_hi']:+.4f}]")
        if "excess_vs_benchmark" in row:
            line += (f" | vs QQQ {row['excess_vs_benchmark']:+.4f} "
                     f"[{row['excess_vs_benchmark_lo']:+.4f}, {row['excess_vs_benchmark_hi']:+.4f}]")
        print(line)
    g = summary["gate"]
    print(f"PROMOTION GATE: {'PASS' if g['pass'] else 'FAIL'}")
    for reason in g["reasons"]:
        print(f"  - {reason}")
    print("Caveat: universe membership is today's; survivorship bias is reduced, not removed.")


def run_backtest(
    mode: str = "weekly",
    tickers: Optional[str] = None,
    entry_valid: int = config.BACKTEST_ENTRY_VALID_BARS,
    max_hold: int = config.BACKTEST_MAX_HOLD_BARS,
    *,
    universe: Optional[str] = None,
    one_position: bool = True,
    random_mult: int = RANDOM_ENTRY_MULT,
    seed: int = 0,
    primary_horizon: int = 13,
) -> pd.DataFrame:
    """Run Coiled Cobra backtest + random-entry null across raw CSVs using all CPU cores.

    Writes ``coiled_cobra_backtest_trades_<date>.csv``, ``coiled_cobra_random_trades_<date>.csv``
    and ``coiled_cobra_backtest_summary_<date>.json`` (with the promotion gate).
    """
    mode_cfg = config.get_mode_config(mode)
    raw_dir = mode_cfg["raw_dir"]
    logs_dir = mode_cfg["logs_dir"]
    ticker_filter = None
    if tickers:
        ticker_filter = {t.strip().upper() for t in tickers.split(",") if t.strip()}
    universe_set, universe_desc = resolve_universe(universe)

    paths = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    )
    work_paths = [
        path for path in paths
        if (not ticker_filter or ticker_from_filename(path) in ticker_filter)
        and (universe_set is None or ticker_from_filename(path) in universe_set)
    ]

    max_workers = os.cpu_count() or 1
    opts = {"one_position": one_position,
            "slippage_pct": config.BACKTEST_SLIPPAGE_PCT, "random_mult": random_mult, "seed": seed}
    all_trades: list[dict] = []
    all_random: list[dict] = []
    all_counts: dict = {}

    print(f"--- Coiled Cobra Backtest [{mode.upper()}] ---")
    print(f"Raw dir: {raw_dir}")
    print(f"Universe: {universe_desc}")
    print(f"Workers: {max_workers}  Tickers: {len(work_paths)}\n")

    results_by_path: dict[str, tuple[list[dict], dict, list[dict]]] = {}
    failures: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_cobra_worker,
        initargs=(mode,),
    ) as ex:
        futures = {
            ex.submit(_backtest_ticker_worker, path, entry_valid, max_hold, opts): path
            for path in work_paths
        }
        for fut in as_completed(futures):
            path = futures[fut]
            symbol = ticker_from_filename(path)
            try:
                symbol, trades, counts, rand, error = fut.result()
            except Exception as exc:  # noqa: BLE001 - process crash / pickling boundary
                trades, counts, rand, error = [], {}, [], f"worker: {_fmt_exc(exc)}"
            if error:
                print(f"{symbol}: error — {error}", file=sys.stderr)
                failures.append({"Symbol": symbol, "Error": error})
            results_by_path[path] = (trades, counts, rand)
            if trades:
                filled = sum(1 for t in trades if t["Outcome"] != "no_fill")
                print(f"{symbol}: {len(trades)} signal(s), {filled} filled")

    # Reassemble in sorted path order so CSV row sequence matches single-threaded runs.
    for path in work_paths:
        trades, counts, rand = results_by_path.get(path, ([], {}, []))
        all_trades.extend(trades)
        all_random.extend(rand)
        for key, value in counts.items():
            all_counts[key] = all_counts.get(key, 0) + value

    trades_df = pd.DataFrame(all_trades)
    random_df = pd.DataFrame(all_random)
    stamp = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"coiled_cobra_backtest_trades_{stamp}.csv")
    trades_df.to_csv(out_path, index=False)
    rand_path = os.path.join(logs_dir, f"coiled_cobra_random_trades_{stamp}.csv")
    random_df.to_csv(rand_path, index=False)
    print(f"Saved backtest output: {out_path}")
    print(f"Saved random-entry null: {rand_path}")

    print("Outcome counts: " + ", ".join(f"{k}={v}" for k, v in sorted(all_counts.items())))
    if all_counts.get("errors"):
        print(f"⚠️ {all_counts['errors']} signal(s) skipped due to level/simulation errors.")
    err_path = _write_failures(logs_dir, "backtest", failures)
    if err_path:
        print(f"⚠️ {len(failures)} ticker(s) failed entirely; details: {err_path}")

    path_by_symbol = {ticker_from_filename(p): p for p in paths}

    def load_close(sym: str) -> Optional[pd.Series]:
        p = path_by_symbol.get(sym)
        if p is None:
            return None
        f = load_ohlc_csv(p)
        return pd.Series(f["Close"].astype(float).to_numpy(), index=pd.to_datetime(f["Date"]))

    summary = summarize_backtest(
        trades_df, random_df, load_close, load_close(BENCHMARK),
        primary_horizon=primary_horizon, start_bar=LOOKBACK // 2 + 15,
    )
    newest = max((os.path.getmtime(p) for p in work_paths), default=None)
    summary.update({
        "mode": mode,
        "universe": universe_desc,
        "raw_data_as_of": datetime.fromtimestamp(newest).isoformat(timespec="minutes") if newest else None,
        "settings": {"fills": "gap_aware", "one_position": one_position,
                     "slippage_pct": config.BACKTEST_SLIPPAGE_PCT, "entry_valid_bars": entry_valid,
                     "max_hold_bars": max_hold, "random_mult": random_mult, "seed": seed},
        "outcome_counts": all_counts,
        "failed_tickers": len(failures),
    })
    summary_path = os.path.join(logs_dir, f"coiled_cobra_backtest_summary_{stamp}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _print_summary(summary)
    print(f"\nSaved summary: {summary_path}")

    return trades_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coiled Cobra historical backfill and backtest")
    parser.add_argument("mode", nargs="?", default=config.DEFAULT_MODE, choices=list(config.TIMEFRAME_PROFILES))
    parser.add_argument("--tickers", help="Comma-separated tickers to include")
    parser.add_argument("--backfill", action="store_true", help="Export historical Coiled Cobra signal archive")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward Coiled Cobra backtest")
    parser.add_argument("--entry-valid", type=int, default=config.BACKTEST_ENTRY_VALID_BARS)
    parser.add_argument("--max-hold", type=int, default=config.BACKTEST_MAX_HOLD_BARS)
    parser.add_argument("--universe", default=None,
                        help="'manifest' (curated list only) or a CSV of tickers; default: all raw files")
    parser.add_argument("--all-signals", action="store_true",
                        help="Allow overlapping positions on the same ticker (e.g. for ML datasets)")
    parser.add_argument("--random-mult", type=int, default=RANDOM_ENTRY_MULT,
                        help="Random-entry null draws per real signal (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primary-horizon", type=int, default=13,
                        help="Forward-return horizon (bars) used by the promotion gate")
    args = parser.parse_args(argv)

    if not args.backfill and not args.backtest:
        args.backtest = True

    if args.backfill:
        generate_backfill(args.mode, args.tickers)

    if args.backtest:
        run_backtest(
            args.mode, args.tickers, args.entry_valid, args.max_hold,
            universe=args.universe, one_position=not args.all_signals, random_mult=args.random_mult,
            seed=args.seed, primary_horizon=args.primary_horizon,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
