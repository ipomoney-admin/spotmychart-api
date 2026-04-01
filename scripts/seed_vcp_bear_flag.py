"""
seed_vcp_bear_flag.py

Seeds backtest_signals and stock_tiers for vcp and bear_flag patterns.
"""

import csv
import os
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BATCH_SIZE = 500

DIR = "/Users/sahib/spotmychart"

FILES = {
    "vcp": {
        "backtest": f"{DIR}/vcp_backtest_raw.csv",
        "tiers":    f"{DIR}/vcp_stock_tiers.csv",
    },
    "bear_flag": {
        "backtest": f"{DIR}/bear_flag_backtest_raw.csv",
        "tiers":    f"{DIR}/bear_flag_stock_tiers.csv",
    },
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def rr_val(v):
    """Return None if rr is null, empty, or 999."""
    f = num(v)
    if f is None or f == 999:
        return None
    return f


def boolval(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def intval(v):
    f = num(v)
    return int(f) if f is not None else None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def insert_batches(table, records):
    inserted = 0
    for batch in chunks(records, BATCH_SIZE):
        sb.table(table).insert(batch).execute()
        inserted += len(batch)
        print(f"    Inserted {inserted}/{len(records)}...")
    return inserted


# ─── VCP backtest_signals ─────────────────────────────────────────────────────

def seed_vcp_backtest():
    pattern_id = "vcp"
    print(f"\n── STEP 1: backtest_signals [{pattern_id}] ──────────────────────")

    print(f"  Deleting existing rows (pattern_id='{pattern_id}')...")
    sb.table("backtest_signals").delete().eq("pattern_id", pattern_id).execute()
    print("  Deleted.")

    rows = read_csv(FILES["vcp"]["backtest"])
    print(f"  Read {len(rows)} rows from CSV.")

    records = []
    for r in rows:
        records.append({
            "symbol":       r["symbol"].strip(),
            "name":         r.get("name", "").strip() or None,
            "sector":       r.get("sector", "").strip() or None,
            "signal_date":  r["signal_date"].strip(),
            "stage":        intval(r.get("stage")),
            "pattern_id":   pattern_id,
            "entry_price":  num(r.get("entry_price")),
            "sl_price":     num(r.get("sl_price")),
            "sl_pct":       num(r.get("sl_pct")),
            "resistance":   num(r.get("resistance")),
            "pct_from_res": num(r.get("pct_from_res")),
            "contractions": intval(r.get("contractions")),
            "final_depth":  num(r.get("final_depth")),
            "confidence":   intval(r.get("confidence")),
            # struct SL variant → primary ret columns
            "ret_15d":      num(r.get("ret_struct_15d")),
            "sl_hit_15d":   boolval(r.get("sl_hit_struct_15d")),
            "ret_30d":      num(r.get("ret_struct_30d")),
            "sl_hit_30d":   boolval(r.get("sl_hit_struct_30d")),
            "ret_60d":      num(r.get("ret_struct_60d")),
            "sl_hit_60d":   boolval(r.get("sl_hit_struct_60d")),
            "ret_90d":      num(r.get("ret_struct_90d")),
            "sl_hit_90d":   boolval(r.get("sl_hit_struct_90d")),
            "max_gain_90d": num(r.get("max_gain_90d")),
        })

    inserted = insert_batches("backtest_signals", records)
    print(f"  Done. Total inserted: {inserted}")


# ─── VCP stock_tiers ──────────────────────────────────────────────────────────

def seed_vcp_tiers():
    pattern_slug = "vcp"
    print(f"\n── STEP 2: stock_tiers [{pattern_slug}] ─────────────────────────")

    print(f"  Deleting existing rows (pattern_slug='{pattern_slug}')...")
    sb.table("stock_tiers").delete().eq("pattern_slug", pattern_slug).execute()
    print("  Deleted.")

    rows = read_csv(FILES["vcp"]["tiers"])
    print(f"  Read {len(rows)} rows from CSV.")

    now_iso = datetime.now(timezone.utc).isoformat()

    records = []
    for r in rows:
        records.append({
            "symbol":        r["symbol"].strip(),
            "pattern_slug":  pattern_slug,
            "tier":          intval(r.get("tier")),
            "total_signals": intval(r.get("signals")),
            "win_rate_30d":  num(r.get("win_rate")),
            "avg_win":       num(r.get("avg_win")),
            "avg_loss":      num(r.get("avg_loss")),
            "rr":            rr_val(r.get("rr")),
            "expectancy":    num(r.get("expectancy")),
            "updated_at":    now_iso,
        })

    inserted = insert_batches("stock_tiers", records)
    print(f"  Done. Total inserted: {inserted}")


# ─── Bear Flag backtest_signals ───────────────────────────────────────────────

def seed_bear_flag_backtest():
    pattern_id = "bear_flag"
    print(f"\n── STEP 3: backtest_signals [{pattern_id}] ─────────────────────")

    print(f"  Deleting existing rows (pattern_id='{pattern_id}')...")
    sb.table("backtest_signals").delete().eq("pattern_id", pattern_id).execute()
    print("  Deleted.")

    rows = read_csv(FILES["bear_flag"]["backtest"])
    print(f"  Read {len(rows)} rows from CSV.")

    records = []
    for r in rows:
        records.append({
            "symbol":       r["symbol"].strip(),
            "name":         r.get("name", "").strip() or None,
            "sector":       r.get("sector", "").strip() or None,
            "signal_date":  r["signal_date"].strip(),
            "stage":        intval(r.get("stage")),
            "pattern_id":   pattern_id,
            "entry_price":  num(r.get("entry_price")),
            "sl_price":     num(r.get("sl_price")),
            "sl_pct":       num(r.get("sl_pct")),
            "resistance":   num(r.get("resistance")),
            "pct_from_res": num(r.get("pct_from_res")),
            "confidence":   intval(r.get("confidence")),
            "pole_move":    num(r.get("pole_move_pct")),
            "flag_len":     intval(r.get("flag_len")),
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

    inserted = insert_batches("backtest_signals", records)
    print(f"  Done. Total inserted: {inserted}")


# ─── Bear Flag stock_tiers ────────────────────────────────────────────────────

def seed_bear_flag_tiers():
    pattern_slug = "bear_flag"
    print(f"\n── STEP 4: stock_tiers [{pattern_slug}] ──────────────────────────")

    print(f"  Deleting existing rows (pattern_slug='{pattern_slug}')...")
    sb.table("stock_tiers").delete().eq("pattern_slug", pattern_slug).execute()
    print("  Deleted.")

    rows = read_csv(FILES["bear_flag"]["tiers"])
    # filter hold_days == 90
    rows = [r for r in rows if str(r.get("hold_days", "")).strip() == "90"]
    print(f"  Read {len(rows)} rows (hold_days=90) from CSV.")

    now_iso = datetime.now(timezone.utc).isoformat()

    records = []
    for r in rows:
        records.append({
            "symbol":        r["symbol"].strip(),
            "pattern_slug":  pattern_slug,
            "tier":          intval(r.get("tier")),
            "total_signals": intval(r.get("signals")),
            "win_rate_30d":  num(r.get("win_rate")),
            "avg_win":       num(r.get("avg_win")),
            "avg_loss":      num(r.get("avg_loss")),
            "rr":            rr_val(r.get("rr")),
            "expectancy":    num(r.get("expectancy")),
            "updated_at":    now_iso,
        })

    inserted = insert_batches("stock_tiers", records)
    print(f"  Done. Total inserted: {inserted}")


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify():
    print("\n── STEP 5: Verify ───────────────────────────────────────────────")

    # backtest_signals counts per pattern_id
    patterns = ["vcp", "bear_flag", "bull_flag"]
    print("\n  backtest_signals row counts:")
    for pid in patterns:
        count = (
            sb.table("backtest_signals")
            .select("id", count="exact")
            .eq("pattern_id", pid)
            .execute()
            .count
        )
        print(f"    {pid:<15} {count:>6} rows")

    # stock_tiers tier distribution per pattern_slug
    print("\n  stock_tiers tier distribution:")
    for slug in patterns:
        rows = (
            sb.table("stock_tiers")
            .select("tier")
            .eq("pattern_slug", slug)
            .execute()
            .data or []
        )
        dist = dict(sorted(Counter(r["tier"] for r in rows).items()))
        print(f"    {slug:<15} total={len(rows):>4}  tiers={dist}")


if __name__ == "__main__":
    seed_vcp_backtest()
    seed_vcp_tiers()
    seed_bear_flag_backtest()
    seed_bear_flag_tiers()
    verify()
