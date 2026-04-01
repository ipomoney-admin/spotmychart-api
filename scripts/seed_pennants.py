"""
seed_pennants.py

Seeds backtest_signals and stock_tiers for bull_pennant and bear_pennant.
Hold periods: 15, 30, 60, 90d only (45d excluded from tiers).
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

BATCH_SIZE   = 500
DIR          = "/Users/sahib/spotmychart"
TIER_HOLD    = "90"   # one row per (symbol, pattern_slug) — use 90d period


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
    f = num(v)
    return None if f is None or f == 999 else f


def boolval(v):
    s = str(v).strip().lower() if v is not None else ""
    if s in ("true", "1", "yes"):  return True
    if s in ("false", "0", "no"):  return False
    return None


def intval(v):
    f = num(v)
    return int(f) if f is not None else None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def insert_batches(table, records, label):
    inserted = 0
    for batch in chunks(records, BATCH_SIZE):
        sb.table(table).insert(batch).execute()
        inserted += len(batch)
        print(f"    Inserted {inserted}/{len(records)}...")
    print(f"  Done. {label}: {inserted} rows")
    return inserted


# ─── backtest_signals ─────────────────────────────────────────────────────────

def seed_backtest(pattern_id, csv_path):
    print(f"\n── backtest_signals [{pattern_id}] ──────────────────────────────")
    print(f"  Deleting existing (pattern_id='{pattern_id}')...")
    sb.table("backtest_signals").delete().eq("pattern_id", pattern_id).execute()
    print("  Deleted.")

    rows = read_csv(csv_path)
    print(f"  Read {len(rows)} rows from {csv_path.split('/')[-1]}")

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

    return insert_batches("backtest_signals", records, pattern_id)


# ─── stock_tiers ──────────────────────────────────────────────────────────────

def seed_tiers(pattern_slug, csv_path):
    print(f"\n── stock_tiers [{pattern_slug}] ──────────────────────────────────")
    print(f"  Deleting existing (pattern_slug='{pattern_slug}')...")
    sb.table("stock_tiers").delete().eq("pattern_slug", pattern_slug).execute()
    print("  Deleted.")

    all_rows = read_csv(csv_path)
    rows = [r for r in all_rows if str(r.get("hold_days", "")).strip() == TIER_HOLD]
    print(f"  Read {len(all_rows)} total rows, keeping {len(rows)} (hold_days={TIER_HOLD})")

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

    return insert_batches("stock_tiers", records, pattern_slug)


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify():
    print("\n── Verify ───────────────────────────────────────────────────────")

    patterns = ["bull_pennant", "bear_pennant", "bull_flag", "bear_flag", "vcp"]
    print("\n  backtest_signals:")
    for pid in patterns:
        count = (
            sb.table("backtest_signals")
            .select("id", count="exact")
            .eq("pattern_id", pid)
            .execute()
            .count
        )
        print(f"    {pid:<15} {count:>6} rows")

    print("\n  stock_tiers tier distribution:")
    for slug in ["bull_pennant", "bear_pennant"]:
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
    seed_backtest("bull_pennant", f"{DIR}/bull_pennant_backtest_raw.csv")
    seed_tiers("bull_pennant",   f"{DIR}/bull_pennant_stock_tiers.csv")

    seed_backtest("bear_pennant", f"{DIR}/bear_pennant_backtest_raw.csv")
    seed_tiers("bear_pennant",   f"{DIR}/bear_pennant_stock_tiers.csv")

    verify()
