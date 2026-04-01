"""
master_backtest.py
==================
Unified backtest runner for all SpotMyChart patterns.

Produces standardized 25-combination output (5 SL variants × 5 hold periods)
for every pattern, seeds results to Supabase, and prints tier summaries.

Usage:
    python3 scripts/master_backtest.py --pattern vcp
    python3 scripts/master_backtest.py --all

SL variants  : 5, 8, 10, 12, 15  (% from entry)
Hold periods : 15, 30, 45, 60, 90 (trading days)
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client, Client

warnings.filterwarnings("ignore")

# ── Environment ───────────────────────────────────────────────────────────────
# Load from API .env first, then bridge the NEXT_PUBLIC_ alias that the
# individual pattern scripts reference at module level during import.
load_dotenv()
os.environ.setdefault(
    "NEXT_PUBLIC_SUPABASE_URL",
    os.environ.get("SUPABASE_URL", ""),
)

# ── Import detection functions from pattern scripts ───────────────────────────
# The individual scripts live in /Users/sahib/spotmychart/scripts/. We add that
# directory to sys.path and import only the detection functions; each script
# also creates its own supabase client at import time (which is harmless — we
# never call their seeding/run_backtest functions here).
_PATTERN_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "spotmychart", "scripts")
)
sys.path.insert(0, _PATTERN_SCRIPTS_DIR)

from backtest_vcp          import find_vcp_signals           # noqa: E402
from backtest_bull_flag    import find_bull_flag_signals      # noqa: E402
from backtest_bear_flag    import find_bear_flag_signals      # noqa: E402
from backtest_bull_pennant import find_bull_pennant_signals   # noqa: E402
from backtest_bear_pennant import find_bear_pennant_signals   # noqa: E402

# ── Supabase client ───────────────────────────────────────────────────────────
sb: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

# ── Standard config ───────────────────────────────────────────────────────────
SL_VARIANTS    = [5, 8, 10, 12, 15]    # % from entry price
HOLD_PERIODS   = [15, 30, 45, 60, 90]  # approximate trading days
MIN_BARS_APART = 15                     # minimum bars between same-stock signals
BATCH_SIZE     = 500

# CSV output directory (same folder as all individual backtest CSVs)
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "spotmychart")
)

# ── Pattern registry ──────────────────────────────────────────────────────────
PATTERN_REGISTRY: dict[str, dict] = {
    "vcp":          {"fn": find_vcp_signals,          "bias": "bull"},
    "bull_flag":    {"fn": find_bull_flag_signals,    "bias": "bull"},
    "bear_flag":    {"fn": find_bear_flag_signals,    "bias": "bear"},
    "bull_pennant": {"fn": find_bull_pennant_signals, "bias": "bull"},
    "bear_pennant": {"fn": find_bear_pennant_signals, "bias": "bear"},
    # Add more patterns here as they are built:
    # "cup_handle": {"fn": find_cup_handle_signals, "bias": "bull"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def assign_tier(total_signals: int, win_rate: float) -> int:
    if total_signals < 3:  return 5
    if win_rate >= 75:     return 1
    if win_rate >= 50:     return 2
    if win_rate >= 25:     return 3
    return 4


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def clean(v):
    """Return None for NaN/None, else the value."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


# ── Standardized return computation ──────────────────────────────────────────

def compute_returns(df_full: pd.DataFrame, signal: dict, bias: str) -> dict:
    """
    Compute forward returns for all 25 SL × hold combinations.

    Column naming: ret_{sl_pct}pct_{days}d  /  sl_hit_{sl_pct}pct_{days}d
    e.g. ret_5pct_15d, sl_hit_8pct_90d

    Gap logic (applied before each bar):
      Bull: open <= sl  → exit at open (gap-down)
            close <= sl → exit at sl   (intraday)
      Bear: open >= sl  → exit at open (gap-up)
            close >= sl → exit at sl   (intraday)

    Returns are always sign-positive for winning trades:
      Bull: (exit - entry) / entry * 100
      Bear: (entry - exit) / entry * 100
    """
    idx   = signal["bar_idx"]
    entry = signal["entry_price"]
    n     = len(df_full)
    result: dict = {}

    for sl_pct in SL_VARIANTS:
        sl_price = (
            entry * (1 - sl_pct / 100) if bias == "bull"
            else entry * (1 + sl_pct / 100)
        )

        for days in HOLD_PERIODS:
            target_idx = idx + days
            col_ret = f"ret_{sl_pct}pct_{days}d"
            col_sl  = f"sl_hit_{sl_pct}pct_{days}d"

            if target_idx >= n:
                result[col_ret] = None
                result[col_sl]  = None
                continue

            sl_hit: bool = False
            exit_price: float | None = None

            for i in range(idx + 1, target_idx + 1):
                o = float(df_full["Open"].iloc[i])
                c = float(df_full["Close"].iloc[i])

                if bias == "bull":
                    if o <= sl_price:
                        exit_price = o
                        sl_hit = True
                        break
                    if c <= sl_price:
                        exit_price = sl_price
                        sl_hit = True
                        break
                else:
                    if o >= sl_price:
                        exit_price = o
                        sl_hit = True
                        break
                    if c >= sl_price:
                        exit_price = sl_price
                        sl_hit = True
                        break

            if not sl_hit:
                exit_price = float(df_full["Close"].iloc[target_idx])

            ret = (
                (exit_price - entry) / entry * 100 if bias == "bull"
                else (entry - exit_price) / entry * 100
            )
            result[col_ret] = round(ret, 2)
            result[col_sl]  = sl_hit

    # max_gain_90d — best price movement in forward 90 bars
    end_90 = min(idx + 91, n)
    fwd_90 = df_full.iloc[idx:end_90]
    if bias == "bull":
        max_h = float(fwd_90["High"].max()) if len(fwd_90) > 0 else entry
        result["max_gain_90d"] = round((max_h - entry) / entry * 100, 2)
    else:
        min_l = float(fwd_90["Low"].min()) if len(fwd_90) > 0 else entry
        result["max_gain_90d"] = round((entry - min_l) / entry * 100, 2)

    return result


# ── Tier builder ──────────────────────────────────────────────────────────────

def build_tiers(df_raw: pd.DataFrame, pattern_slug: str) -> pd.DataFrame:
    """
    Compute tier rows for every (symbol, sl_pct, hold_days) combination.
    Returns DataFrame with 25 rows per stock (or fewer if data missing).
    """
    rows: list[dict] = []
    for symbol, grp in df_raw.groupby("symbol"):
        for sl in SL_VARIANTS:
            for days in HOLD_PERIODS:
                col = f"ret_{sl}pct_{days}d"
                if col not in grp.columns:
                    continue
                valid = grp[grp[col].notna()]
                if len(valid) == 0:
                    continue

                wins     = valid[valid[col] > 0]
                losses   = valid[valid[col] <= 0]
                win_rate = len(wins) / len(valid) * 100
                avg_win  = float(wins[col].mean())  if len(wins)   > 0 else 0.0
                avg_loss = float(losses[col].mean()) if len(losses) > 0 else 0.0
                rr       = round(avg_win / abs(avg_loss), 2) if avg_loss < 0 else None
                exp      = round(
                    (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss), 2
                )

                rows.append({
                    "symbol":        symbol,
                    "pattern_slug":  pattern_slug,
                    "sl_pct":        sl,
                    "hold_days":     days,
                    "tier":          assign_tier(len(valid), win_rate),
                    "total_signals": len(valid),
                    "win_rate":      round(win_rate, 1),
                    "avg_win":       round(avg_win, 2),
                    "avg_loss":      round(avg_loss, 2),
                    "rr":            rr,
                    "expectancy":    exp,
                })

    return pd.DataFrame(rows)


# ── Schema migration helper ───────────────────────────────────────────────────

def ensure_stock_tiers_schema() -> bool:
    """
    Check if stock_tiers has sl_pct and hold_days columns.
    Returns True if the schema is ready for 25-combo inserts.
    If not, prints the required SQL and returns False (falls back to primary-only insert).
    """
    probe = sb.table("stock_tiers").select("*").limit(1).execute().data
    cols  = set(probe[0].keys()) if probe else set()

    if "sl_pct" in cols and "hold_days" in cols:
        print("  ✅ stock_tiers schema OK (sl_pct + hold_days present)")
        return True

    missing = [c for c in ("sl_pct", "hold_days") if c not in cols]
    print(f"  ⚠️  stock_tiers missing columns: {missing}")
    print("  Run the following SQL in the Supabase SQL editor, then re-run this script:\n")
    print("    ALTER TABLE stock_tiers ADD COLUMN IF NOT EXISTS sl_pct    FLOAT;")
    print("    ALTER TABLE stock_tiers ADD COLUMN IF NOT EXISTS hold_days INTEGER;")
    print()
    print("    -- Update unique constraint to allow all 25 combos per stock:")
    print("    ALTER TABLE stock_tiers")
    print("      DROP CONSTRAINT IF EXISTS stock_tiers_symbol_pattern_slug_key;")
    print("    ALTER TABLE stock_tiers")
    print("      ADD CONSTRAINT stock_tiers_symbol_pattern_slug_sl_hold_key")
    print("      UNIQUE (symbol, pattern_slug, sl_pct, hold_days);")
    print()
    print("  Falling back to primary-only insert (sl_pct=5, hold_days=90).")
    return False


# ── Supabase seeding ──────────────────────────────────────────────────────────

def _seed_backtest_signals(df_raw: pd.DataFrame, pattern_id: str, bias: str) -> None:
    """
    Seed backtest_signals table.
    Primary return columns (ret_15d etc.) use the 5% SL variant.
    The DB has no ret_45d column so 45d is skipped.
    """
    print(f"  Seeding backtest_signals ({pattern_id})...")
    sb.table("backtest_signals").delete().eq("pattern_id", pattern_id).execute()

    records: list[dict] = []
    for _, r in df_raw.iterrows():
        ep = float(r["entry_price"])
        records.append({
            "symbol":       r["symbol"],
            "name":         clean(r.get("name"))  or None,
            "sector":       clean(r.get("sector")) or None,
            "signal_date":  r["signal_date"],
            "stage":        int(r["stage"]) if pd.notna(r.get("stage")) else None,
            "pattern_id":   pattern_id,
            "entry_price":  ep,
            "sl_price":     round(ep * 0.95, 2) if bias == "bull" else round(ep * 1.05, 2),
            "sl_pct":       5.0,
            "confidence":   int(r["confidence"]) if pd.notna(r.get("confidence")) else None,
            # pattern-specific metadata
            "resistance":   clean(r.get("resistance")),
            "pct_from_res": clean(r.get("pct_from_res")),
            "pole_move":    clean(r.get("pole_move_pct")),
            "flag_len":     int(r["flag_len"])    if pd.notna(r.get("flag_len"))    else None,
            "contractions": int(r["contractions"]) if pd.notna(r.get("contractions")) else None,
            "final_depth":  clean(r.get("final_depth")),
            # primary returns — 5% SL variant (DB has no ret_45d column)
            "ret_15d":      clean(r.get("ret_5pct_15d")),
            "sl_hit_15d":   bool(r["sl_hit_5pct_15d"])  if pd.notna(r.get("sl_hit_5pct_15d"))  else None,
            "ret_30d":      clean(r.get("ret_5pct_30d")),
            "sl_hit_30d":   bool(r["sl_hit_5pct_30d"])  if pd.notna(r.get("sl_hit_5pct_30d"))  else None,
            "ret_60d":      clean(r.get("ret_5pct_60d")),
            "sl_hit_60d":   bool(r["sl_hit_5pct_60d"])  if pd.notna(r.get("sl_hit_5pct_60d"))  else None,
            "ret_90d":      clean(r.get("ret_5pct_90d")),
            "sl_hit_90d":   bool(r["sl_hit_5pct_90d"])  if pd.notna(r.get("sl_hit_5pct_90d"))  else None,
            "max_gain_90d": clean(r.get("max_gain_90d")),
        })

    inserted = 0
    for batch in chunks(records, BATCH_SIZE):
        sb.table("backtest_signals").insert(batch).execute()
        inserted += len(batch)
    print(f"    Inserted {inserted} rows.")


def _seed_stock_tiers(
    df_tiers: pd.DataFrame, pattern_slug: str, has_sl_hold_cols: bool
) -> None:
    """
    Seed stock_tiers table.
    - has_sl_hold_cols=True:  insert all 25 (sl_pct × hold_days) combinations.
    - has_sl_hold_cols=False: insert only primary view (sl_pct=5, hold_days=90).
    """
    print(f"  Seeding stock_tiers ({pattern_slug})...")
    sb.table("stock_tiers").delete().eq("pattern_slug", pattern_slug).execute()

    now_iso = datetime.now(timezone.utc).isoformat()

    if has_sl_hold_cols:
        subset = df_tiers
    else:
        subset = df_tiers[
            (df_tiers["sl_pct"] == 5) & (df_tiers["hold_days"] == 90)
        ].copy()
        if len(subset) == 0:
            print("    No primary-view rows found — skipping.")
            return
        print(f"    (schema migration pending — inserting primary view only: {len(subset)} rows)")

    records: list[dict] = []
    for _, r in subset.iterrows():
        rec: dict = {
            "symbol":        r["symbol"],
            "pattern_slug":  pattern_slug,
            "tier":          int(r["tier"]),
            "total_signals": int(r["total_signals"]),
            "win_rate_30d":  clean(r["win_rate"]),
            "avg_win":       clean(r["avg_win"]),
            "avg_loss":      clean(r["avg_loss"]),
            "rr":            clean(r["rr"]) if r["rr"] is not None and not (
                                 isinstance(r["rr"], float) and pd.isna(r["rr"])
                             ) else None,
            "expectancy":    clean(r["expectancy"]),
            "updated_at":    now_iso,
        }
        if has_sl_hold_cols:
            rec["sl_pct"]    = int(r["sl_pct"])
            rec["hold_days"] = int(r["hold_days"])
        records.append(rec)

    inserted = 0
    for batch in chunks(records, BATCH_SIZE):
        sb.table("stock_tiers").insert(batch).execute()
        inserted += len(batch)
    print(f"    Inserted {inserted} rows.")


# ── Tier summary printer ──────────────────────────────────────────────────────

def print_tier_summary(df_tiers: pd.DataFrame, pattern_id: str) -> None:
    """Print tier table for SL=5% across all hold periods, plus T1 detail at 90d."""
    div = "=" * 60
    print(f"\n  {div}")
    print(f"  TIER SUMMARY — {pattern_id.upper()}  (SL = 5%)")
    print(f"  {div}")

    sub5 = df_tiers[df_tiers["sl_pct"] == 5]
    for days in HOLD_PERIODS:
        sub = sub5[sub5["hold_days"] == days]
        if sub.empty:
            continue
        print(f"\n  --- {days}D HOLD ---")
        for t in [1, 2, 3, 4, 5]:
            ts = sub[sub["tier"] == t]
            if ts.empty:
                continue
            wr  = ts["win_rate"].mean()
            ev  = ts["expectancy"].mean()
            cnt = len(ts)
            print(f"    T{t}: {cnt:3d} stocks | WR: {wr:.1f}% | EV: {ev:+.2f}%")

    # T1 deep-dive at 5pct / 90d
    t1 = sub5[(sub5["hold_days"] == 90) & (sub5["tier"] == 1)]
    if not t1.empty:
        rr_vals = t1["rr"].dropna()
        print(f"\n  --- T1 DETAIL @ 5% SL / 90D ---")
        print(f"    Stocks  : {len(t1)}")
        print(f"    Win Rate: {t1['win_rate'].mean():.1f}%")
        print(f"    Avg Win : {t1['avg_win'].mean():.2f}%")
        print(f"    Avg Loss: {t1['avg_loss'].mean():.2f}%")
        print(f"    R:R     : {rr_vals.mean():.2f}" if not rr_vals.empty else "    R:R     : N/A")
        print(f"    EV      : {t1['expectancy'].mean():+.2f}%")


# ── Per-pattern runner ────────────────────────────────────────────────────────

def run_backtest_for_pattern(pattern_id: str, has_sl_hold_cols: bool) -> None:
    cfg  = PATTERN_REGISTRY[pattern_id]
    fn   = cfg["fn"]
    bias = cfg["bias"]

    raw_path   = os.path.join(OUTPUT_DIR, f"{pattern_id}_backtest_raw.csv")
    tiers_path = os.path.join(OUTPUT_DIR, f"{pattern_id}_stock_tiers.csv")

    stocks = (
        sb.table("stocks")
        .select("id,symbol,name,sector")
        .eq("is_active", True)
        .execute()
        .data
    )

    print(f"\n{'='*65}")
    print(f"  PATTERN : {pattern_id.upper()}")
    print(f"  BIAS    : {bias}")
    print(f"  STOCKS  : {len(stocks)}")
    print(f"{'='*65}")

    all_rows: list[dict] = []
    errors   = 0

    for i, stock in enumerate(stocks):
        ticker = stock["symbol"] + ".NS"
        try:
            df = yf.download(
                ticker, period="5y", interval="1d",
                progress=False, auto_adjust=True,
            )
            if df.empty or len(df) < 300:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            signals = fn(df)

            for sig in signals:
                fwd  = compute_returns(df, sig, bias)
                row  = {
                    "symbol":       stock["symbol"],
                    "name":         stock.get("name", ""),
                    "sector":       stock.get("sector", ""),
                    "signal_date":  sig["date"],
                    "stage":        sig["stage"],
                    "entry_price":  sig["entry_price"],
                    "confidence":   sig.get("confidence"),
                    "bias":         bias,
                    # pattern-specific fields (None when not applicable)
                    "resistance":    sig.get("resistance"),
                    "pct_from_res":  sig.get("pct_from_res"),
                    "pole_move_pct": sig.get("pole_move_pct"),
                    "flag_len":      sig.get("flag_len"),
                    "contractions":  sig.get("contractions"),
                    "final_depth":   sig.get("final_depth"),
                }
                row.update(fwd)
                all_rows.append(row)

            if signals:
                n_sig = len(signals)
                print(f"  ✅ {stock['symbol']:<14} {n_sig} signal{'s' if n_sig > 1 else ' '}")

        except Exception as exc:
            errors += 1
            # Uncomment for verbose error logging:
            # print(f"  ❌ {stock['symbol']}: {exc}")

        if (i + 1) % 50 == 0:
            print(f"\n  [{i + 1:>3}/{len(stocks)}] checkpt — {len(all_rows)} signals so far\n")

        time.sleep(0.2)

    if not all_rows:
        print(f"\n  No signals found for {pattern_id}. Skipping CSV/DB output.")
        return

    # Write raw CSV
    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(raw_path, index=False)
    print(f"\n  Raw CSV  → {raw_path} ({len(df_raw)} rows)")

    # Build tiers and write tier CSV
    df_tiers = build_tiers(df_raw, pattern_id)
    df_tiers.to_csv(tiers_path, index=False)
    print(f"  Tier CSV → {tiers_path} ({len(df_tiers)} rows)")

    # Print tier summary
    print_tier_summary(df_tiers, pattern_id)

    # Seed to Supabase
    print(f"\n  Seeding to Supabase...")
    _seed_backtest_signals(df_raw, pattern_id, bias)
    _seed_stock_tiers(df_tiers, pattern_id, has_sl_hold_cols)

    print(f"\n  ✔ {pattern_id} complete — errors: {errors}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpotMyChart Master Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/master_backtest.py --pattern vcp\n"
            "  python3 scripts/master_backtest.py --all\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pattern",
        choices=list(PATTERN_REGISTRY.keys()),
        metavar="PATTERN",
        help=f"Single pattern to run. Choices: {', '.join(PATTERN_REGISTRY)}",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all registered patterns sequentially",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  SpotMyChart Master Backtest")
    print(f"  SL variants  : {SL_VARIANTS}")
    print(f"  Hold periods : {HOLD_PERIODS}")
    print("=" * 65)

    # Schema migration check
    print("\nChecking stock_tiers schema...")
    has_sl_hold_cols = ensure_stock_tiers_schema()

    patterns_to_run = list(PATTERN_REGISTRY.keys()) if args.all else [args.pattern]
    print(f"\nPatterns queued: {patterns_to_run}")

    t_start = time.time()
    for pattern_id in patterns_to_run:
        run_backtest_for_pattern(pattern_id, has_sl_hold_cols)

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  ALL DONE  —  {len(patterns_to_run)} pattern(s)  —  {elapsed/60:.1f} min")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
