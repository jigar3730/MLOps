"""Finance Vibe pipeline orchestrator.

Runs ingestion, macro scoring, tactical scanning, and trade plan generation
in sequence for a given timeframe profile (weekly or daily).
"""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import os
import zipfile

# How many previous raw-data snapshots to keep per mode (see snapshot_raw_folder).
RAW_SNAPSHOT_KEEP = 8


def snapshot_raw_folder(root_dir, mode, keep: int = RAW_SNAPSHOT_KEEP):
    """Zip the current data/raw/{mode}/ CSVs into data/snapshots/{mode}/ before they are replaced.

    The archive is named after the newest file's modification time (i.e. when
    that data was ingested), so any past report can be re-derived from the
    exact inputs it used. Only the newest *keep* archives are retained.
    Returns the archive path, or None when there was nothing to snapshot.
    """
    raw_dir = Path(root_dir) / "data" / "raw" / mode
    files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not files:
        return None

    newest = max(f.stat().st_mtime for f in files)
    stamp = datetime.fromtimestamp(newest).strftime("%Y-%m-%d_%H%M")
    snap_dir = Path(root_dir) / "data" / "snapshots" / mode
    snap_dir.mkdir(parents=True, exist_ok=True)
    archive = snap_dir / f"raw_{mode}_{stamp}.zip"

    if not archive.exists():
        tmp = archive.with_suffix(".zip.partial")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        tmp.replace(archive)  # atomic: never leave a half-written snapshot

    archives = sorted(snap_dir.glob(f"raw_{mode}_*.zip"))
    for old in archives[:-keep] if keep > 0 else []:
        old.unlink()
    return archive


def clean_raw_folder(root_dir, mode):
    """Remove all files in data/raw/{mode}/ before a fresh ingestion run."""
    raw_dir = Path(root_dir) / "data" / "raw" / mode
    if not raw_dir.exists():
        print(f"⚠️ Raw '{mode}' folder does not exist. Skipping cleanup.")
        return

    failed = 0
    for item in raw_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except OSError as e:
            failed += 1
            print(f"❌ Failed to delete {item}: {e}")

    if failed:
        # Stale leftovers would be scanned as if they were fresh; stop here.
        raise SystemExit(f"❌ {failed} raw file(s) could not be removed; aborting run.")
    print(f"🧹 Raw '{mode}' folder cleaned.\n")

def run_workflow():
    """Parse CLI args and execute each pipeline stage as a subprocess."""
    parser = argparse.ArgumentParser(description="Finance-Vibe Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["weekly", "daily", "high_beta"],
        default="weekly",
        help="Execution profile (weekly, daily, or high_beta long-only single names)",
    )
    args = parser.parse_args()
    mode = args.mode.lower()

    # high_beta reads daily OHLCV but keeps its own swing profile + log silo.
    data_mode = "daily" if mode == "high_beta" else mode

    # 2. CLIMB TO ROOT
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    SRC_DIR = os.path.join(ROOT_DIR, "src")

    # 3. ENVIRONMENT SETUP
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")

    # 4. SCRIPT CONFIGURATION
    # "scope" selects the argument each stage receives:
    #   data    -> data timeframe (weekly/daily); shares raw data silo
    #   profile -> swing profile (weekly/daily/high_beta); drives geometry + logs
    # high_beta skips Coiled Cobra (LEAPS-oriented, not part of the long-only swing).
    # The macro vibe report (analysis_engine.py) is not a pipeline stage; run it
    # standalone if needed. Its scoring is still used by the swing scanner's gate.
    scripts_config = [
        {"path": "src/finance_vibe/ticker_provider.py", "pass_mode": False, "scope": "data"},
        {"path": "src/finance_vibe/data_ingestor.py", "pass_mode": True, "scope": "data"},
        {"path": "src/finance_vibe/swing_scanner.py", "pass_mode": True, "scope": "profile"},
        {"path": "src/finance_vibe/coiled_cobra.py", "pass_mode": True, "scope": "data",
         "skip_modes": ["high_beta"]},
        {"path": "src/finance_vibe/trade_planner.py", "pass_mode": True, "scope": "profile"},
        {"path": "src/finance_vibe/trade_plan_helper.py", "pass_mode": True, "scope": "profile"},
    ]

    print(f"🚀 Starting Finance-Vibe Pipeline [{mode.upper()} MODE]...")
    print(f"📍 Project Root: {ROOT_DIR}")
    if mode != data_mode:
        print(f"🧬 Data timeframe: {data_mode} | Swing profile: {mode}")
    print()

    # Snapshot, then clean the shared raw silo for the data timeframe.
    archive = snapshot_raw_folder(ROOT_DIR, data_mode)
    if archive:
        print(f"🗄️  Previous raw data archived: {archive}")
    clean_raw_folder(ROOT_DIR, data_mode)

    for script in scripts_config:
        if mode in script.get("skip_modes", []):
            print(f"⏭️  Skipping {script['path']} for {mode} mode.\n")
            continue

        script_path = os.path.join(ROOT_DIR, script["path"])
        print(f"🔹 Running: {script['path']}...")

        # Data-scope stages receive the data timeframe; profile-scope stages
        # receive the swing profile so high_beta geometry/logs stay isolated.
        arg_mode = data_mode if script.get("scope") == "data" else mode

        cmd = [sys.executable, script_path]
        if script["pass_mode"]:
            cmd.append(arg_mode)

        try:
            subprocess.run(cmd, check=True, env=env, cwd=ROOT_DIR)
            print(f"✅ Finished: {script['path']}\n")
        except subprocess.CalledProcessError:
            print(f"❌ Error in {script['path']}. Pipeline halted.")
            sys.exit(1)

    print("🏁 Workflow Complete!")
    print(f"📁 Reports saved to: {os.path.join(ROOT_DIR, 'data', 'logs', mode)}")

if __name__ == "__main__":
    run_workflow()