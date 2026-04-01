"""
seed_bull_flag.py

Seeds backtest_signals and stock_tiers for bull_flag pattern.
"""

import csv
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BATCH_SIZE = 500
PATTERN_ID = "bull_flag"

BACKTEST_CSV = "/Users/sahib/spotmychart/bull_flag_backtest_raw.csv"
TIERS_CSV    = "/Users/sahib/spotmychart/bull_flag_stock_tiers.csv"


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def num(v):
    if v is None or str(v).strip() in ("", "None", "nan", "NaN"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def boolval(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


# ─── STEP 1: backtest_signals ────────────────────────────────────────────────

def seed_backtest_signals():
    print("\n── STEP 1: backtest_signals ─────────────────────────────────────")

    # DELETE existing
    print(f"  Deleting existing rows with pattern_id = '{PATTERN_ID}'...")
    sb.table("backtest_signals").delete().eq("pattern_id", PATTERN_ID).execute()
    print("  Deleted.")

    # Read CSV
    rows = []
    with open(BACKTEST_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Read {len(rows)} rows from CSV.")

    # Map columns
    records = []
    for r in rows:
        records.append({
            "symbol":       r["symbol"].strip(),
            "name":         r["name"].strip() if r.get("name") else None,
            "sector":       r["sector"].strip() if r.get("sector") else None,
            "signal_date":  r["signal_date"].strip(),
            "stage":        int(r["stage"]) if r.get("stage") and r["stage"].strip() else None,
            "pattern_id":   PATTERN_ID,
            "entry_price":  num(r.get("entry_price")),
            "sl_price":     num(r.get("sl_price")),
            "sl_pct":       num(r.get("sl_pct")),
            "resistance":   num(r.get("resistance")),
            "pct_from_res": num(r.get("pct_from_res")),
            "confidence":   int(float(r["confidence"])) if r.get("confidence") and r["confidence"].strip() else None,
            "pole_move":    num(r.get("pole_move_pct")),
            "flag_len":     int(float(r["flag_len"])) if r.get("flag_len") and r["flag_len"].strip() else None,
            "ret_15d":      num(r.get("ret_15d")),
            "sl_hit_15d":   boolval(r.get("sl_hit_15d")),
            "ret_30d":      num(r.get("ret_30d")),
            "sl_hit_30d":   boolval(r.get("sl_hit_30d")),
            "ret_60d":      num(r.get("ret_60d")),
            "sl_hit_60d":   boolval(r.get("sl_hit_60d")),
            "ret_90d":      num(r.get("ret_90d")),
            "sl_hit_90d":   boolval(r.get("sl_hit_90d")),
            "max_gain_90d": num(r.get("max_gain_90d")),
        })

    # Insert in batches
    inserted = 0
    for batch in chunks(records, BATCH_SIZE):
        sb.table("backtest_signals").insert(batch).execute()
        inserted += len(batch)
        print(f"  Inserted {inserted}/{len(records)}...")

    print(f"  Done. Total inserted: {inserted}")
    return inserted


# ─── STEP 2: stock_tiers ─────────────────────────────────────────────────────

def seed_stock_tiers():
    print("\n── STEP 2: stock_tiers ──────────────────────────────────────────")

    # DELETE existing
    print(f"  Deleting existing rows with pattern_slug = '{PATTERN_ID}'...")
    sb.table("stock_tiers").delete().eq("pattern_slug", PATTERN_ID).execute()
    print("  Deleted.")

    # Read CSV, filter hold_days == 90
    rows = []
    with open(TIERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("hold_days", "")).strip() == "90":
                rows.append(row)
    print(f"  Read {len(rows)} rows (hold_days=90) from CSV.")

    now_iso = datetime.now(timezone.utc).isoformat()

    records = []
    for r in rows:
        records.append({
            "symbol":       r["symbol"].strip(),
            "pattern_slug": PATTERN_ID,
            "tier":         int(r["tier"]) if r.get("tier") and r["tier"].strip() else None,
            "total_signals":int(float(r["signals"])) if r.get("signals") and r["signals"].strip() else None,
            "win_rate_30d": num(r.get("win_rate")),
            "avg_win":      num(r.get("avg_win")),
            "avg_loss":     num(r.get("avg_loss")),
            "rr":           num(r.get("rr")),
            "expectancy":   num(r.get("expectancy")),
            "updated_at":   now_iso,
        })

    # Insert in batches
    inserted = 0
    for batch in chunks(records, BATCH_SIZE):
        sb.table("stock_tiers").insert(batch).execute()
        inserted += len(batch)
        print(f"  Inserted {inserted}/{len(records)}...")

    print(f"  Done. Total inserted: {inserted}")
    return inserted


# ─── STEP 3: Verify ──────────────────────────────────────────────────────────

def verify():
    print("\n── STEP 3: Verify ───────────────────────────────────────────────")

    sig_count = (
        sb.table("backtest_signals")
        .select("id", count="exact")
        .eq("pattern_id", PATTERN_ID)
        .execute()
        .count
    )
    print(f"  backtest_signals (bull_flag): {sig_count} rows")

    tier_rows = (
        sb.table("stock_tiers")
        .select("tier")
        .eq("pattern_slug", PATTERN_ID)
        .execute()
        .data or []
    )
    tier_count = len(tier_rows)

    from collections import Counter
    dist = Counter(r["tier"] for r in tier_rows)
    print(f"  stock_tiers    (bull_flag): {tier_count} rows")
    print(f"  Tier distribution: { dict(sorted(dist.items())) }")


if __name__ == "__main__":
    seed_backtest_signals()
    seed_stock_tiers()
    verify()
