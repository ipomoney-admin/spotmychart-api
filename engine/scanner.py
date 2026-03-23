import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

from core.nse_calendar import is_trading_day
from core.supabase_client import sb
from data.corporate_actions import check_corporate_action_proximity
from data.fetcher import fetch_ohlcv
from detection.patterns import (
    ascending_triangle,
    bear_flag,
    bear_pennant,
    bull_flag,
    bull_pennant,
    cup_handle,
    desc_triangle,
    double_bottom,
    double_top,
    falling_wedge,
    head_shoulders,
    inv_cup_handle,
    inv_head_shoulders,
    rising_wedge,
    rounding_bottom,
    rounding_top,
    sym_triangle_bear,
    sym_triangle_bull,
    triple_bottom,
    triple_top,
    vcp,
)
from detection.stage_analyzer import get_stage
from detection.zigzag_pivots import get_pivots
from engine.metrics_calculator import calculate_metrics
from engine.signal_locker import acquire_lock, is_locked, release_lock
from engine.tier_assigner import assign_tier

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Pattern registry — add new detectors here only                      #
# ------------------------------------------------------------------ #
DETECTORS: dict = {
    "vcp":                 vcp.detect,
    "bull_flag":           bull_flag.detect,
    "bull_pennant":        bull_pennant.detect,
    "ascending_triangle":  ascending_triangle.detect,
    "sym_triangle_bull":   sym_triangle_bull.detect,
    "cup_handle":          cup_handle.detect,
    "double_bottom":       double_bottom.detect,
    "triple_bottom":       triple_bottom.detect,
    "inv_head_shoulders":  inv_head_shoulders.detect,
    "rounding_bottom":     rounding_bottom.detect,
    "falling_wedge":       falling_wedge.detect,
    "bear_flag":           bear_flag.detect,
    "bear_pennant":        bear_pennant.detect,
    "desc_triangle":       desc_triangle.detect,
    "sym_triangle_bear":   sym_triangle_bear.detect,
    "inv_cup_handle":      inv_cup_handle.detect,
    "double_top":          double_top.detect,
    "triple_top":          triple_top.detect,
    "head_shoulders":      head_shoulders.detect,
    "rounding_top":        rounding_top.detect,
    "rising_wedge":        rising_wedge.detect,
}

# Fallback tickers if smc_metrics has no data yet
_FALLBACK_TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# In-memory scan state for get_scan_status()
_scan_state: dict = {
    "last_scan_at":    None,
    "tickers_scanned": 0,
    "signals_found":   0,
    "errors":          0,
}

OHLCV_START = date(2015, 1, 1)
# Lock duration: 20 trading days (~1 calendar month)
LOCK_DAYS = 30
# Max hold days before auto-close
PRIMARY_HOLD_DAYS = 20


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _fetch_active_tickers() -> list[str]:
    try:
        resp = sb.table("smc_metrics").select("ticker").execute()
        tickers = list({row["ticker"] for row in resp.data if row.get("ticker")})
    except Exception as e:
        logger.warning(f"Could not fetch tickers from smc_metrics: {e}. Using fallback list.")
        tickers = []

    merged = list({*tickers, *_FALLBACK_TICKERS})
    return merged


def _insert_signal(ticker: str, pattern_result: dict, stage: int, today: date) -> Optional[str]:
    signal_id = str(uuid.uuid4())
    try:
        sb.table("smc_signals").insert({
            "id":               signal_id,
            "ticker":           ticker,
            "pattern_key":      pattern_result["pattern_key"],
            "state":            pattern_result["state"],
            "confidence_score": pattern_result["confidence_score"],
            "resistance":       pattern_result["resistance"],
            "sl_price":         pattern_result["sl_price"],
            "contraction_count": pattern_result.get("contraction_count"),
            "final_depth_pct":  pattern_result.get("final_depth_pct"),
            "stage":            stage,
            "signal_date":      today.isoformat(),
        }).execute()
        return signal_id
    except Exception as e:
        logger.error(f"[{ticker}] Failed to insert signal: {e}")
        return None


def _update_trades(ticker: str, current_close: float, today: date) -> list[str]:
    """
    Close active trades for ticker where SL is hit or hold period expired.
    Returns list of pattern_keys whose trades were closed (for metrics refresh).
    """
    affected_patterns: list[str] = []
    try:
        resp = (
            sb.table("smc_trades")
            .select("*")
            .eq("ticker", ticker)
            .eq("status", "active")
            .execute()
        )
    except Exception as e:
        logger.error(f"[{ticker}] Failed to fetch active trades: {e}")
        return affected_patterns

    for trade in resp.data:
        try:
            sl_price          = float(trade["sl_price"])
            entry_date        = date.fromisoformat(trade["entry_date"])
            pattern_key       = trade["pattern_key"]
            primary_hold_days = trade.get("primary_hold_days", PRIMARY_HOLD_DAYS)
            days_held         = (today - entry_date).days

            sl_hit      = current_close <= sl_price
            time_expired = days_held >= primary_hold_days

            if not (sl_hit or time_expired):
                continue

            entry_price  = float(trade["entry_price"])
            return_pct   = (current_close - entry_price) / entry_price * 100
            close_status = "closed_win" if return_pct >= 0 else "closed_loss"
            close_reason = "sl_hit" if sl_hit else "time_expiry"

            sb.table("smc_trades").update({
                "status":       close_status,
                "close_price":  current_close,
                "close_date":   today.isoformat(),
                "return_pct":   round(return_pct, 2),
                "close_reason": close_reason,
            }).eq("id", trade["id"]).execute()

            release_lock(ticker)
            affected_patterns.append(pattern_key)
            logger.info(
                f"[{ticker}] Trade closed — reason={close_reason} "
                f"return={return_pct:.2f}% status={close_status}"
            )
            _recalculate_metrics(ticker, pattern_key)
        except Exception as e:
            logger.error(f"[{ticker}] Failed to close trade {trade.get('id')}: {e}")

    return affected_patterns


def _recalculate_metrics(ticker: str, pattern_key: str) -> None:
    try:
        resp = (
            sb.table("smc_trades")
            .select("return_pct, status")
            .eq("ticker", ticker)
            .eq("pattern_key", pattern_key)
            .in_("status", ["closed_win", "closed_loss"])
            .execute()
        )
        trades = resp.data
        metrics = calculate_metrics(trades)
        tier    = assign_tier(metrics["win_rate"], len(trades))

        sb.table("smc_metrics").upsert({
            "ticker":           ticker,
            "pattern_key":      pattern_key,
            "win_rate":         metrics["win_rate"],
            "avg_win_pct":      metrics["avg_win_pct"],
            "avg_loss_pct":     metrics["avg_loss_pct"],
            "risk_reward":      metrics["risk_reward"],
            "expectancy":       metrics["expectancy"],
            "max_gain_pct":     metrics["max_gain_pct"],
            "max_loss_pct":     metrics["max_loss_pct"],
            "occurrence_count": len(trades),
            "tier":             tier,
            "updated_at":       datetime.utcnow().isoformat(),
        }, on_conflict="ticker,pattern_key").execute()

        logger.info(f"[{ticker}] Metrics updated — pattern={pattern_key} tier={tier} win_rate={metrics['win_rate']}%")
    except Exception as e:
        logger.error(f"[{ticker}] Failed to recalculate metrics for {pattern_key}: {e}")


# ------------------------------------------------------------------ #
# Main pipeline                                                       #
# ------------------------------------------------------------------ #

def run_daily_scan() -> None:
    # ── Step 1: trading day guard ────────────────────────────────── #
    today = date.today()
    if not is_trading_day(today):
        logger.info(f"Scan skipped — {today} is not an NSE trading day.")
        return

    logger.info(f"=== SpotMyChart daily scan started: {today} ===")

    tickers_scanned = 0
    signals_found   = 0
    errors          = 0

    # ── Step 2: fetch tickers ────────────────────────────────────── #
    tickers = _fetch_active_tickers()
    logger.info(f"Scanning {len(tickers)} tickers.")

    for ticker in tickers:
        try:
            # ── Step 3: fetch OHLCV ──────────────────────────────── #
            df = fetch_ohlcv(ticker, start_date=OHLCV_START, end_date=today)
            if df.empty:
                logger.warning(f"[{ticker}] Empty OHLCV — skipping.")
                errors += 1
                continue

            current_close = float(df["close"].iloc[-1])

            # ── Step 10: close stale/stopped trades (before emitting new signals) ──
            # _recalculate_metrics is called inside _update_trades for each closed trade
            _update_trades(ticker, current_close, today)

            # ── Step 4: stage analysis ───────────────────────────── #
            stage = get_stage(df)

            # ── Step 5: pivots ───────────────────────────────────── #
            pivots = get_pivots(df)

            # ── Step 6: run all detectors ────────────────────────── #
            for pattern_key, detector_fn in DETECTORS.items():
                try:
                    result = detector_fn(df, stage, pivots)
                    if result is None:
                        continue

                    # ── Step 7: check lock ───────────────────────── #
                    if is_locked(ticker):
                        logger.debug(f"[{ticker}] Locked — skipping {pattern_key} signal.")
                        continue

                    # Corporate action proximity check
                    if check_corporate_action_proximity(ticker, today):
                        logger.info(f"[{ticker}] Corporate action proximity — signal quarantined.")
                        continue

                    # Only emit on confirmed or forming signals
                    # Forming signals are recorded but don't acquire a lock
                    signal_id = _insert_signal(ticker, result, stage, today)
                    if signal_id is None:
                        continue

                    signals_found += 1
                    logger.info(
                        f"[{ticker}] Signal emitted — pattern={pattern_key} "
                        f"state={result['state']} confidence={result['confidence_score']}"
                    )

                    # ── Step 8 / 9: lock on confirmed signals only ── #
                    if result["state"] == "confirmed":
                        expires_at = today + timedelta(days=LOCK_DAYS)
                        acquired = acquire_lock(ticker, pattern_key, signal_id, expires_at)
                        if not acquired:
                            logger.warning(f"[{ticker}] Lock acquisition failed for {signal_id}.")

                except Exception as e:
                    logger.error(f"[{ticker}] Detector '{pattern_key}' error: {e}")
                    errors += 1

            tickers_scanned += 1

        except Exception as e:
            logger.error(f"[{ticker}] Unhandled error during scan: {e}")
            errors += 1

    # ── Step 12: scan summary ────────────────────────────────────── #
    _scan_state["last_scan_at"]    = datetime.utcnow().isoformat()
    _scan_state["tickers_scanned"] = tickers_scanned
    _scan_state["signals_found"]   = signals_found
    _scan_state["errors"]          = errors

    logger.info(
        f"=== Scan complete: tickers={tickers_scanned} signals={signals_found} errors={errors} ==="
    )


# ------------------------------------------------------------------ #
# Status                                                              #
# ------------------------------------------------------------------ #

def get_scan_status() -> dict:
    return {**_scan_state}
