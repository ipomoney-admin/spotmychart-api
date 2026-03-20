"""
migrate_backtest.py

Migrates backtest_signals → smc_signals, smc_trades, smc_metrics.

Handles duplicate (symbol, pattern, date) combos in source data:
  - One smc_signal per unique (ticker, pattern_key, detection_date)
  - One smc_trade per source row, all linked to the deduplicated signal
"""

import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client, Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.tier_assigner import assign_tier

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BATCH_SIZE = 100

PATTERN_MAP = {
    "vcp": "vcp",
    "bullish_flag": "bull_flag",
    "bullish_pennant": "bull_pennant",
    "bullish_wedge_flag": "bull_flag",
    "bearish_flag": "bear_flag",
    "bearish_pennant": "bear_pennant",
    "bearish_wedge": "rising_wedge",
}

BEARISH_PATTERNS = {"bear_flag", "bear_pennant", "rising_wedge"}


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def fetch_all_backtest_signals():
    rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = (
            sb.table("backtest_signals")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def clean_partial_data():
    """Delete any previously-inserted partial data so re-runs are safe."""
    print("  Checking for existing data to clean up...")
    sig_count = sb.table("smc_signals").select("id", count="exact").execute().count or 0
    trd_count = sb.table("smc_trades").select("id", count="exact").execute().count or 0
    met_count = sb.table("smc_metrics").select("stock_ticker", count="exact").execute().count or 0

    if sig_count or trd_count or met_count:
        print(f"  Found {sig_count} signals, {trd_count} trades, {met_count} metrics — deleting...")
        # Delete in dependency order: trades → signals → metrics
        if trd_count:
            sb.table("smc_trades").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        if sig_count:
            sb.table("smc_signals").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        if met_count:
            sb.table("smc_metrics").delete().neq("stock_ticker", "").execute()
        print("  Cleanup done.")
    else:
        print("  Tables are empty — no cleanup needed.")


def main():
    print("=== migrate_backtest.py ===")

    # 0. Clean up any partial previous run
    print("\n[0/4] Cleanup...")
    clean_partial_data()

    # 1. Fetch source data
    print("\n[1/4] Fetching backtest_signals...")
    all_rows = fetch_all_backtest_signals()
    print(f"      Fetched {len(all_rows)} rows")

    # Filter to rows with a known pattern mapping
    valid_rows = [r for r in all_rows if r["pattern_id"] in PATTERN_MAP]
    skipped = len(all_rows) - len(valid_rows)
    if skipped:
        print(f"      Skipping {skipped} rows with unknown pattern_id")

    # Attach mapped pattern_key and derived fields to each valid row
    for r in valid_rows:
        pk = PATTERN_MAP[r["pattern_id"]]
        r["_pattern_key"] = pk
        # Use 30d return/sl for bearish patterns, 90d for bullish
        if pk in BEARISH_PATTERNS:
            r["_return_pct"] = r.get("ret_30d")
            r["_is_winner"] = not r.get("sl_hit_30d", True)
        else:
            r["_return_pct"] = r.get("ret_90d")
            r["_is_winner"] = not r.get("sl_hit_90d", True)

    # 2. Insert smc_signals (deduplicated by ticker+pattern+date)
    print(f"\n[2/4] Inserting unique signals into smc_signals...")

    # Build ordered list of unique signal keys (preserve first-seen order)
    seen_signal_keys = {}  # (ticker, pattern_key, date) → index into unique_signal_dicts
    unique_signal_dicts = []
    for r in valid_rows:
        key = (r["symbol"], r["_pattern_key"], r["signal_date"])
        if key not in seen_signal_keys:
            seen_signal_keys[key] = len(unique_signal_dicts)
            unique_signal_dicts.append(
                {
                    "stock_ticker": r["symbol"],
                    "pattern_key": r["_pattern_key"],
                    "detection_date": r["signal_date"],
                    "confidence_score": r.get("confidence"),
                    "state": "confirmed",
                }
            )

    print(f"      {len(valid_rows)} source rows → {len(unique_signal_dicts)} unique signals")

    inserted_signals = []
    for i, batch in enumerate(chunks(unique_signal_dicts, BATCH_SIZE)):
        resp = sb.table("smc_signals").insert(batch).execute()
        inserted_signals.extend(resp.data or [])
        processed = min((i + 1) * BATCH_SIZE, len(unique_signal_dicts))
        if processed % 100 == 0 or processed == len(unique_signal_dicts):
            print(f"      signals inserted: {processed}/{len(unique_signal_dicts)}")

    # Build lookup: (ticker, pattern_key, date) → signal_id
    signal_id_lookup = {
        (s["stock_ticker"], s["pattern_key"], s["detection_date"]): s["id"]
        for s in inserted_signals
    }

    # 3. Insert smc_trades (one per source row)
    print(f"\n[3/4] Inserting {len(valid_rows)} rows into smc_trades...")
    trade_records = []
    for r in valid_rows:
        sig_key = (r["symbol"], r["_pattern_key"], r["signal_date"])
        signal_id = signal_id_lookup.get(sig_key)
        if not signal_id:
            print(f"  [WARN] No signal_id for {sig_key} — skipping trade")
            continue
        status = "closed_win" if r["_is_winner"] else "closed_loss"
        trade_records.append(
            {
                "signal_id": signal_id,
                "stock_ticker": r["symbol"],
                "pattern_key": r["_pattern_key"],
                "entry_date": r["signal_date"],
                "return_pct": r["_return_pct"],
                "status": status,
                "close_reason": "time_expired",
            }
        )

    inserted_trades = []
    for i, batch in enumerate(chunks(trade_records, BATCH_SIZE)):
        resp = sb.table("smc_trades").insert(batch).execute()
        inserted_trades.extend(resp.data or [])
        processed = min((i + 1) * BATCH_SIZE, len(trade_records))
        if processed % 100 == 0 or processed == len(trade_records):
            print(f"      trades inserted: {processed}/{len(trade_records)}")

    # 4. Calculate and upsert smc_metrics
    print("\n[4/4] Calculating and upserting smc_metrics...")
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in valid_rows:
        ret = r["_return_pct"]
        if ret is not None:
            groups[(r["symbol"], r["_pattern_key"])].append(float(ret))

    metrics_records = []
    for (ticker, pattern_key), returns in groups.items():
        total = len(returns)
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        win_rate = (len(wins) / total * 100) if total else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else None
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        timeline = 30 if pattern_key in BEARISH_PATTERNS else 90
        tier = assign_tier(win_rate, total)

        metrics_records.append(
            {
                "stock_ticker": ticker,
                "pattern_key": pattern_key,
                "timeline_days": timeline,
                "win_rate": round(win_rate, 4),
                "avg_win_pct": round(avg_win, 4),
                "avg_loss_pct": round(avg_loss, 4),
                "risk_reward": round(risk_reward, 4) if risk_reward is not None else None,
                "expectancy": round(expectancy, 4),
                "max_gain_pct": round(max(returns), 4),
                "max_loss_pct": round(min(returns), 4),
                "occurrence_count": total,
                "tier": tier,
            }
        )

    upserted_metrics = []
    for i, batch in enumerate(chunks(metrics_records, BATCH_SIZE)):
        resp = sb.table("smc_metrics").insert(batch).execute()
        upserted_metrics.extend(resp.data or [])
        processed = min((i + 1) * BATCH_SIZE, len(metrics_records))
        if processed % 100 == 0 or processed == len(metrics_records):
            print(f"      metrics inserted: {processed}/{len(metrics_records)}")

    # Summary
    print("\n=== Migration Complete ===")
    print(f"  Source rows fetched  : {len(all_rows)}")
    print(f"  Rows skipped         : {skipped}")
    print(f"  Signals inserted     : {len(inserted_signals)}")
    print(f"  Trades inserted      : {len(inserted_trades)}")
    print(f"  Metrics inserted     : {len(upserted_metrics)}")


if __name__ == "__main__":
    main()
