import os
import pandas as pd
from yahooquery import Screener

# --- 1. PACKAGE IMPORT ---
try:
    from finance_vibe import config
except ImportError:
    import config  # Fallback for local testing

MANIFEST_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'ticker_manifest.csv')


_HEADER_NAMES = {"SYMBOL", "TICKER"}


def _normalize_symbols(symbols) -> list[str]:
    """Uppercase, strip, drop blanks / indices; map share-class dots to Yahoo dashes.

    ``BRK.B`` -> ``BRK-B`` (Yahoo's convention) instead of silently dropping it.
    Order is preserved and duplicates removed.
    """
    out = []
    for t in symbols:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            continue
        s = str(t).upper().strip().replace(".", "-")
        if not s or "^" in s:
            continue
        out.append(s)
    return list(dict.fromkeys(out))


def load_manifest(path: str = None) -> list[str]:
    """Read the ticker manifest, with or without a header row.

    The shipped manifest has no header; reading it with pandas' default
    ``header=0`` silently turned the first ticker into the column name.
    """
    path = path or MANIFEST_PATH
    raw = pd.read_csv(path, header=None, usecols=[0], dtype=str, skip_blank_lines=True)
    values = raw.iloc[:, 0].dropna().astype(str).str.strip()
    values = values[~values.str.upper().isin(_HEADER_NAMES)]
    return _normalize_symbols(values.tolist())


def refresh_active_tickers():
    print("--- STEP 1: Discovering Tickers (Manifest + Active) ---")
    cap = int(getattr(config, "ACTIVE_TICKER_CAP", 1000))
    screener_ids = list(getattr(config, "SCREENER_IDS", ["most_actives", "day_gainers"]))
    screener_count = int(getattr(config, "SCREENER_COUNT", 250))

    # 1. Load the Static Manifest
    static_tickers = []
    if os.path.exists(MANIFEST_PATH):
        try:
            static_tickers = load_manifest(MANIFEST_PATH)
            print(f"📦 Loaded {len(static_tickers)} static tickers from manifest.")
        except (OSError, ValueError, pd.errors.ParserError) as e:
            print(f"⚠️ Could not read manifest: {e}")
    else:
        print(f"⚠️ Manifest not found at {MANIFEST_PATH}")

    # 2. Add Static Tickers from config.py (priority baseline, manifest order kept)
    static_tickers = _normalize_symbols(static_tickers + config.STATIC_TICKERS)
    print(f"📌 Priority baseline (manifest + STATIC_TICKERS): {len(static_tickers)}")

    # 3. Discover Active Tickers via Screener
    discovered_tickers: list[str] = []
    try:
        s = Screener()
        data = s.get_screeners(screener_ids, count=screener_count)
        for screen_id in screener_ids:
            quotes = data.get(screen_id, {}).get("quotes") if isinstance(data, dict) else None
            if not quotes:
                print(f"⚠️ Screener '{screen_id}' returned no quotes")
                continue
            batch = _normalize_symbols(q["symbol"] for q in quotes if "symbol" in q)
            discovered_tickers.extend(batch)
            print(f"🔎 Screener '{screen_id}': {len(batch)} symbols")

        # Preserve discovery order while deduping
        seen = set()
        unique_discovered = []
        for t in discovered_tickers:
            if t not in seen:
                seen.add(t)
                unique_discovered.append(t)

        # 4. Manifest/static first, then fill with screener names up to ACTIVE_TICKER_CAP
        final_list = list(dict.fromkeys(static_tickers))  # stable unique
        final_set = set(final_list)
        for ticker in unique_discovered:
            if len(final_list) >= cap:
                break
            if ticker not in final_set:
                final_list.append(ticker)
                final_set.add(ticker)

        # If static alone already exceeds cap, trim (manifest order preserved via dict.fromkeys)
        if len(final_list) > cap:
            final_list = final_list[:cap]

        # 5. Save using config path
        pd.Series(final_list, name='Ticker').to_csv(
            config.TICKER_LIST_PATH, index=False)

        print(
            f"✅ Success! Saved {len(final_list)} total tickers "
            f"(cap={cap}) to {config.TICKER_LIST_PATH}"
        )

    except Exception as e:
        print(f"❌ Error during ticker discovery: {e}")
        # Still persist the priority baseline so the pipeline is not blocked.
        if static_tickers:
            fallback = static_tickers[:cap]
            pd.Series(fallback, name='Ticker').to_csv(
                config.TICKER_LIST_PATH, index=False)
            print(
                f"⚠️ Fell back to {len(fallback)} static tickers "
                f"at {config.TICKER_LIST_PATH}"
            )


if __name__ == "__main__":
    refresh_active_tickers()
