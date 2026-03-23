import logging
from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from core.supabase_client import sb
from engine.metrics_calculator import calculate_metrics
from engine.tier_assigner import assign_tier

logger = logging.getLogger(__name__)

LOOKBACK_YEARS = 5


def get_lookback_start() -> date:
    today = date.today()
    return today - relativedelta(years=LOOKBACK_YEARS)


def reassign_all_tiers() -> dict:
    logger.info("=== Quarterly tier reassignment started ===")
    lookback_start = get_lookback_start()
    logger.info(f"Using data from {lookback_start} to today (5-year window)")

    # Fetch all unique stock+pattern combinations
    resp = sb.table("smc_trades").select("stock_ticker, pattern_key").execute()
    combos = list({(r["stock_ticker"], r["pattern_key"]) for r in resp.data
                   if r.get("stock_ticker") and r.get("pattern_key")})
    logger.info(f"Total stock+pattern combos: {len(combos)}")

    updated = 0
    errors  = 0

    for stock_ticker, pattern_key in combos:
        try:
            # Fetch only last 5 years of closed trades
            resp = (
                sb.table("smc_trades")
                .select("return_pct, status, entry_date, exit_date")
                .eq("stock_ticker", stock_ticker)
                .eq("pattern_key", pattern_key)
                .in_("status", ["closed_win", "closed_loss"])
                .gte("entry_date", lookback_start.isoformat())
                .execute()
            )

            trades  = resp.data
            metrics = calculate_metrics(trades)
            tier    = assign_tier(metrics["win_rate"], len(trades))

            # Calculate avg holding period
            holding_periods = []
            for t in trades:
                if t.get("entry_date") and t.get("exit_date"):
                    ed = date.fromisoformat(t["entry_date"])
                    xd = date.fromisoformat(t["exit_date"])
                    holding_periods.append((xd - ed).days)
            avg_hold = round(sum(holding_periods) / len(holding_periods), 1) if holding_periods else 0

            sb.table("smc_metrics").upsert({
                "stock_ticker":     stock_ticker,
                "pattern_key":      pattern_key,
                "win_rate":         metrics["win_rate"],
                "avg_win":          metrics["avg_win_pct"],
                "avg_loss":         metrics["avg_loss_pct"],
                "rr_ratio":         metrics["risk_reward"],
                "expectancy":       metrics["expectancy"],
                "max_gain":         metrics["max_gain_pct"],
                "max_loss":         metrics["max_loss_pct"],
                "occurrences":      len(trades),
                "tier":             tier,
                "avg_holding_days": avg_hold,
                "timeline_days":    90,
                "last_updated":     datetime.utcnow().isoformat(),
            }, on_conflict="stock_ticker,pattern_key").execute()

            updated += 1
        except Exception as e:
            logger.error(f"Error reassigning {stock_ticker}/{pattern_key}: {e}")
            errors += 1

    logger.info(f"=== Tier reassignment complete: {updated} updated, {errors} errors ===")
    return {"updated": updated, "errors": errors}
