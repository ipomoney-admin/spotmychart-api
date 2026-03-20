import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from core.supabase_client import sb
from engine.scanner import get_scan_status, run_daily_scan

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------ #
# GET /home                                                           #
# ------------------------------------------------------------------ #

@router.get("/home")
def get_home():
    try:
        top_patterns = (
            sb.table("smc_metrics")
            .select("*, smc_patterns(*)")
            .eq("tier", 1)
            .eq("timeline_days", 90)
            .order("win_rate", desc=True)
            .limit(4)
            .execute()
        ).data

        latest_signals = (
            sb.table("smc_signals")
            .select("pattern_key, stock_ticker, detection_date, confidence_score")
            .eq("state", "confirmed")
            .order("detection_date", desc=True)
            .limit(10)
            .execute()
        ).data

        hall_of_fame = (
            sb.table("smc_hall_of_fame")
            .select("*")
            .order("return_pct", desc=True)
            .limit(8)
            .execute()
        ).data

        return {
            "top_patterns":    top_patterns,
            "latest_signals":  latest_signals,
            "hall_of_fame":    hall_of_fame,
        }

    except Exception as e:
        logger.error(f"GET /home error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch home data.")


# ------------------------------------------------------------------ #
# GET /stock/{ticker}                                                 #
# ------------------------------------------------------------------ #

@router.get("/stock/{ticker}")
def get_stock(ticker: str):
    ticker = ticker.upper().strip()
    try:
        top_patterns = (
            sb.table("smc_metrics")
            .select("*")
            .eq("stock_ticker", ticker)
            .eq("tier", 1)
            .eq("timeline_days", 90)
            .order("win_rate", desc=True)
            .limit(3)
            .execute()
        ).data

        recent_trades = (
            sb.table("smc_trades")
            .select("*")
            .eq("stock_ticker", ticker)
            .order("entry_date", desc=True)
            .limit(10)
            .execute()
        ).data

        lock_resp = (
            sb.table("smc_active_locks")
            .select("*")
            .eq("stock_ticker", ticker)
            .limit(1)
            .execute()
        ).data
        active_lock = lock_resp[0] if lock_resp else None

        all_patterns = (
            sb.table("smc_metrics")
            .select("*")
            .eq("stock_ticker", ticker)
            .eq("timeline_days", 90)
            .order("win_rate", desc=True)
            .execute()
        ).data

        return {
            "ticker":        ticker,
            "top_patterns":  top_patterns,
            "recent_trades": recent_trades,
            "active_lock":   active_lock,
            "all_patterns":  all_patterns,
        }

    except Exception as e:
        logger.error(f"GET /stock/{ticker} error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for {ticker}.")


# ------------------------------------------------------------------ #
# GET /pattern/{pattern_key}                                          #
# ------------------------------------------------------------------ #

@router.get("/pattern/{pattern_key}")
def get_pattern(pattern_key: str):
    pattern_key = pattern_key.lower().strip()
    try:
        pattern_info_resp = (
            sb.table("smc_patterns")
            .select("*")
            .eq("pattern_key", pattern_key)
            .limit(1)
            .execute()
        ).data
        if not pattern_info_resp:
            raise HTTPException(status_code=404, detail=f"Pattern '{pattern_key}' not found.")
        pattern_info = pattern_info_resp[0]

        top_stocks = (
            sb.table("smc_metrics")
            .select("*")
            .eq("pattern_key", pattern_key)
            .eq("tier", 1)
            .eq("timeline_days", 90)
            .order("win_rate", desc=True)
            .limit(20)
            .execute()
        ).data

        recent_trades = (
            sb.table("smc_trades")
            .select("*")
            .eq("pattern_key", pattern_key)
            .order("entry_date", desc=True)
            .limit(10)
            .execute()
        ).data

        return {
            "pattern_info":  pattern_info,
            "top_stocks":    top_stocks,
            "recent_trades": recent_trades,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GET /pattern/{pattern_key} error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data for pattern '{pattern_key}'.")


# ------------------------------------------------------------------ #
# GET /watchlist/{user_id}                                            #
# ------------------------------------------------------------------ #

@router.get("/watchlist/{user_id}")
def get_watchlist(user_id: str):
    try:
        watchlist = (
            sb.table("smc_watchlist")
            .select("*, smc_active_locks(pattern_key, signal_id, expires_at)")
            .eq("user_id", user_id)
            .execute()
        ).data

        return {"user_id": user_id, "watchlist": watchlist}

    except Exception as e:
        logger.error(f"GET /watchlist/{user_id} error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch watchlist.")


# ------------------------------------------------------------------ #
# POST /watchlist                                                     #
# ------------------------------------------------------------------ #

class WatchlistAddRequest(BaseModel):
    user_id: str
    stock_ticker: str


@router.post("/watchlist", status_code=201)
def add_to_watchlist(body: WatchlistAddRequest):
    ticker = body.stock_ticker.upper().strip()
    try:
        sb.table("smc_watchlist").upsert(
            {"user_id": body.user_id, "stock_ticker": ticker},
            on_conflict="user_id,stock_ticker",
        ).execute()
        return {"user_id": body.user_id, "ticker": ticker, "status": "added"}

    except Exception as e:
        logger.error(f"POST /watchlist error ({body.user_id}, {ticker}): {e}")
        raise HTTPException(status_code=500, detail="Failed to add to watchlist.")


# ------------------------------------------------------------------ #
# DELETE /watchlist/{user_id}/{ticker}                                #
# ------------------------------------------------------------------ #

@router.delete("/watchlist/{user_id}/{ticker}")
def remove_from_watchlist(user_id: str, ticker: str):
    ticker = ticker.upper().strip()
    try:
        sb.table("smc_watchlist").delete().eq("user_id", user_id).eq("stock_ticker", ticker).execute()
        return {"user_id": user_id, "ticker": ticker, "status": "removed"}

    except Exception as e:
        logger.error(f"DELETE /watchlist/{user_id}/{ticker} error: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove from watchlist.")


# ------------------------------------------------------------------ #
# GET /scan/status                                                    #
# ------------------------------------------------------------------ #

@router.get("/scan/status")
def scan_status():
    return get_scan_status()


# ------------------------------------------------------------------ #
# POST /scan/run                                                      #
# ------------------------------------------------------------------ #

@router.post("/scan/run")
def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_daily_scan)
    return {"status": "scan started"}
