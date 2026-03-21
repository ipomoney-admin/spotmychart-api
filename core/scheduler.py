import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

scheduler = BackgroundScheduler(timezone=IST)

# In-memory scan status — updated by run_scan_job()
_scan_status: dict = {
    "last_run":      None,   # ISO datetime string (UTC)
    "status":        "idle", # "idle" | "running"
    "stocks_scanned": 0,
}


def get_status() -> dict:
    return {**_scan_status}


def run_scan_job() -> None:
    """Scheduler-facing wrapper: updates status around the real scan."""
    from engine.scanner import get_scan_status, run_daily_scan  # late import avoids circular

    _scan_status["status"] = "running"
    logger.info("Scheduler: starting daily scan job.")
    try:
        run_daily_scan()
        state = get_scan_status()
        _scan_status["stocks_scanned"] = state.get("tickers_scanned", 0)
    except Exception as e:
        logger.error(f"Scheduler: scan job raised an unhandled exception: {e}")
    finally:
        _scan_status["last_run"] = datetime.utcnow().isoformat()
        _scan_status["status"]   = "idle"
        logger.info("Scheduler: scan job finished.")
