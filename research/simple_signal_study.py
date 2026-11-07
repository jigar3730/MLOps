"""
Simple signal study: does "new high + volume spike + beating QQQ" beat random picks?

    python research/simple_signal_study.py           # BUILD period only (2016-2022)
    python research/simple_signal_study.py --test    # the one-time HOLDOUT (2023-today)

Everything is written to data/research/ (git-ignored).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # draw to files; no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "research")
os.makedirs(OUT, exist_ok=True)
PRICES_CSV = os.path.join(OUT, "weekly_prices.csv")

START = "2016-01-04"        # a Monday, so Yahoo's weekly bars are labelled on Mondays
BUILD_END = "2022-12-31"    # we may only LOOK at results up to here while building
TEST_START = "2023-01-01"   # untouched holdout, run once at the very end
HOLD = 13                   # hold each trade 13 weeks
N_RANDOM = 1000             # how many random portfolios to compare against
rng = np.random.default_rng(42)   # fixed seed -> same "random" results every run


# ---------------------------------------------------------------------------
# STEP 1 - Download weekly, split/dividend-adjusted prices and save them.
# ---------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    if os.path.exists(PRICES_CSV):        # re-use the saved file so reruns are identical
        print(f"Step 1: using saved prices {PRICES_CSV} (delete it to re-download)")
        return pd.read_csv(PRICES_CSV, parse_dates=["Date"])

    tickers = pd.read_csv(os.path.join(HERE, "universe.csv"))["Ticker"].tolist() + ["QQQ"]
    print(f"Step 1: downloading {len(tickers)} tickers from Yahoo ...")
    raw = yf.download(tickers, start=START, interval="1wk", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    frames = []
    for t in tickers:                     # one table per ticker -> one long table
        if t in raw.columns.get_level_values(0):
            f = raw[t].dropna(how="all").reset_index()
            frames.append(f.assign(Ticker=t))
    px = pd.concat(frames)[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].dropna()
    px["Date"] = pd.to_datetime(px["Date"])
    if px["Date"].dt.tz is not None:
        px["Date"] = px["Date"].dt.tz_localize(None)

    # The current week isn't finished until Friday's 4pm close - drop it if still open.
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    px = px[px["Date"] + pd.Timedelta(days=4, hours=16) <= now]

    # Every bar must be a Monday-labelled week, or tickers won't line up with QQQ.
    off = px.loc[px["Date"].dt.day_of_week != 0, "Ticker"].unique()
    if len(off):
        print(f"   dropping {len(off)} ticker(s) with non-Monday weeks: {list(off)}")
        px = px[~px["Ticker"].isin(off)]

    px.to_csv(PRICES_CSV, index=False)
    print(f"   saved {px['Ticker'].nunique()} tickers, {len(px):,} weekly rows -> {PRICES_CSV}")
    return px


# ---------------------------------------------------------------------------
# STEP 2 - Features. Each one uses ONLY this week and earlier weeks.
#   .shift(1)   = "look at last week's value"  (backwards in time: safe)
#   .rolling(n) = "the n rows ending here"      (backwards in time: safe)
# ---------------------------------------------------------------------------
def add_features(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values(["Ticker", "Date"]).copy()
    g = px.groupby("Ticker")
    # New 26-week high: this week's close is above the highest high of the PREVIOUS 26 weeks.
    prior_high = g["High"].transform(lambda s: s.shift(1).rolling(26).max())
    px["new_high"] = px["Close"] > prior_high
    # Volume spike: this week's volume > 1.5x the average of the PREVIOUS 10 weeks.
    prior_vol = g["Volume"].transform(lambda s: s.shift(1).rolling(10).mean())
    px["vol_spike"] = px["Volume"] > 1.5 * prior_vol
    # 13-week return (this close vs the close 13 weeks ago) minus QQQ's over the same weeks.
    px["ret13"] = g["Close"].transform(lambda s: s / s.shift(13) - 1)
    qqq = px.loc[px["Ticker"] == "QQQ", ["Date", "ret13"]].rename(columns={"ret13": "qqq_ret13"})
    px = px.merge(qqq, on="Date", how="left")
    px["rel13"] = px["ret13"] - px["qqq_ret13"]
    # (Comparisons with missing history are False, so no signal until 26 weeks of data exist.)
    return px


FEATURES = ["new_high", "vol_spike", "rel13"]


# ---------------------------------------------------------------------------
# STEP 4 - Outcome: buy at NEXT week's open, sell at the close 13 weeks after
# the signal week, and subtract what QQQ did over exactly the same window.
# This is the ONLY place we look forward (.shift(-n)); it's the answer we grade
# against, and it is never used to decide a signal.
# ---------------------------------------------------------------------------
def add_outcome(px: pd.DataFrame) -> pd.DataFrame:
    g = px.groupby("Ticker")
    px["entry"] = g["Open"].shift(-1)
    px["exit"] = g["Close"].shift(-HOLD)
    px["exit_date"] = g["Date"].shift(-HOLD)
    px["fwd"] = px["exit"] / px["entry"] - 1
    qqq = px.loc[px["Ticker"] == "QQQ", ["Date", "fwd"]].rename(columns={"fwd": "qqq_fwd"})
    px = px.merge(qqq, on="Date", how="left")
    px["excess"] = px["fwd"] - px["qqq_fwd"]
    return px


# ---------------------------------------------------------------------------
# LOOK-AHEAD CHECK - If a feature only uses the past, then computing it on data
# that STOPS at some date must give exactly the same values (up to that date)
# as computing it on all the data. A feature that peeks ahead would change.
# ---------------------------------------------------------------------------
def check_no_lookahead(px: pd.DataFrame, n_cuts: int = 15) -> None:
    full = add_features(px).set_index(["Ticker", "Date"])[FEATURES]
    dates = np.sort(px["Date"].unique())
    for cut in rng.choice(dates[60:-5], size=n_cuts, replace=False):
        part = add_features(px[px["Date"] <= cut]).set_index(["Ticker", "Date"])[FEATURES]
        same = full.loc[part.index].fillna(-999).eq(part.fillna(-999)).all().all()
        assert same, f"look-ahead detected: features changed when data was cut at {cut}"
    print(f"Look-ahead check: passed ({n_cuts} random cut-off dates, all {len(FEATURES)} features).")

    # Show the test works: a deliberately cheating feature (next week's close) must FAIL it.
    cheat = lambda d: d.sort_values(["Ticker", "Date"]).assign(
        peek=lambda x: x.groupby("Ticker")["Close"].shift(-1)).set_index(["Ticker", "Date"])["peek"]
    cut = dates[len(dates) // 2]
    a, b = cheat(px), cheat(px[px["Date"] <= cut])
    caught = not a.loc[b.index].fillna(-999).eq(b.fillna(-999)).all()
    print(f"   sanity: a cheating 'next week's close' feature {'IS caught' if caught else 'was NOT caught'} by this test.")


# ---------------------------------------------------------------------------
# STEP 3 - Signal (fixed rules, no tuning) + one position per stock at a time.
# ---------------------------------------------------------------------------
def pick_signals(rows: pd.DataFrame) -> pd.DataFrame:
    sig = rows[rows["new_high"] & rows["vol_spike"] & (rows["rel13"] > 0)].sort_values(["Ticker", "Date"])
    # If a stock signals again while we'd still be holding it, skip that repeat -
    # otherwise one strong run counts as many "independent" wins.
    keep, last = [], {}
    for i, r in sig.iterrows():
        if r["Ticker"] not in last or r["Date"] > last[r["Ticker"]]:
            keep.append(i)
            last[r["Ticker"]] = r["exit_date"]
    return sig.loc[keep]


# ---------------------------------------------------------------------------
# STEP 5 - Random baseline: for each stock with k signals, pick k random weeks
# of that same stock from the same period. Repeat 1,000 times.
# ---------------------------------------------------------------------------
def random_means(signals: pd.DataFrame, pool: pd.DataFrame) -> np.ndarray:
    counts = signals["Ticker"].value_counts()
    by_ticker = {t: pool.loc[pool["Ticker"] == t, "excess"].to_numpy() for t in counts.index}
    means = np.empty(N_RANDOM)
    for i in range(N_RANDOM):
        draws = [rng.choice(by_ticker[t], size=min(k, len(by_ticker[t])), replace=False)
                 for t, k in counts.items()]
        means[i] = np.concatenate(draws).mean()
    return means


def report(name: str, signals: pd.DataFrame, pool: pd.DataFrame) -> None:
    sig_mean = signals["excess"].mean()
    rand = random_means(signals, pool)
    pct = (rand < sig_mean).mean() * 100            # % of random portfolios the signal beats
    print(f"\n=== {name}: {len(signals)} signals in {signals['Ticker'].nunique()} stocks ===")
    print(f"Signal: average 13-week return vs QQQ = {sig_mean:+.2%}")
    print(f"Random: average = {rand.mean():+.2%} (middle 95%: {np.percentile(rand, 2.5):+.2%} "
          f"to {np.percentile(rand, 97.5):+.2%})")
    print(f"The signal beats {pct:.1f}% of 1,000 random portfolios "
          f"(chance of doing this well by luck ≈ {(rand >= sig_mean).mean():.3f}).")

    plt.figure(figsize=(7, 4))
    plt.hist(rand * 100, bins=40, color="#9aa7b8", label="1,000 random portfolios")
    plt.axvline(sig_mean * 100, color="#d1495b", lw=2.5, label=f"signal ({pct:.0f}th percentile)")
    plt.xlabel("average 13-week return minus QQQ (%)")
    plt.ylabel("count")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT, f"hist_{name.split()[0].lower()}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Histogram saved: {path}")

    # STEP 7 - Tier breakdown (AFTER the fact; tier never enters the signal).
    print("By tier (median dollar volume, 2016-2022):")
    for tier in ["large", "mid", "small"]:
        s, p = signals[signals["tier"] == tier], pool[pool["tier"] == tier]
        if len(s) < 5:
            print(f"   {tier:5s}: too few signals ({len(s)})")
            continue
        r = random_means(s, p)
        print(f"   {tier:5s}: {len(s):4d} signals | signal {s['excess'].mean():+.2%} | "
              f"random {r.mean():+.2%} | beats {(r < s['excess'].mean()).mean():.0%} of random")


def main() -> None:
    final_test = "--test" in sys.argv
    px = load_prices()
    check_no_lookahead(px)
    data = add_outcome(add_features(px))
    data = data[data["Ticker"] != "QQQ"]           # QQQ is the yardstick, not a candidate

    # Tier = size by median weekly dollar volume, measured on BUILD years only.
    build_px = data[data["Date"] <= BUILD_END]
    dollar_vol = (build_px["Close"] * build_px["Volume"]).groupby(build_px["Ticker"]).median()
    tiers = pd.qcut(dollar_vol, 3, labels=["small", "mid", "large"]).astype(str)
    data["tier"] = data["Ticker"].map(tiers)

    # STEP 6 - Keep the periods apart. A build-period trade must also FINISH by the end
    # of 2022, so no 2023 price ever leaks into what we look at while building.
    valid = data["excess"].notna()
    if final_test:
        marker = os.path.join(OUT, "holdout_already_used.txt")
        if os.path.exists(marker):
            print("WARNING: the holdout has been run before; it is no longer an unseen test.")
        open(marker, "a").write(f"{pd.Timestamp.now()}\n")
        rows = data[valid & (data["Date"] >= TEST_START)]
        name = "HOLDOUT 2023-today"
    else:
        rows = data[valid & (data["exit_date"] <= BUILD_END)]
        name = "BUILD 2016-2022"
    report(name, pick_signals(rows), rows)


if __name__ == "__main__":
    main()
