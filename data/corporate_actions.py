import logging
from datetime import date

from core.nse_calendar import get_previous_trading_day, is_trading_day
from core.supabase_client import sb

logger = logging.getLogger(__name__)

_PROXIMITY_DAYS = 5
_TABLE = "smc_corporate_actions"


def _trading_day_window(signal_date: date, n: int) -> tuple[date, date]:
    """Return the date range covering n trading days before and after signal_date."""
    # Walk back n trading days
    start = signal_date
    steps = 0
    while steps < n:
        start = get_previous_trading_day(start)
        steps += 1

    # Walk forward n trading days
    end = signal_date
    steps = 0
    from datetime import timedelta
    candidate = signal_date + timedelta(days=1)
    while steps < n:
        if is_trading_day(candidate):
            end = candidate
            steps += 1
        candidate += timedelta(days=1)

    return start, end


def check_corporate_action_proximity(ticker: str, signal_date: date) -> bool:
    """
    Returns True if any corporate action for `ticker` falls within
    _PROXIMITY_DAYS trading days of `signal_date`.
    Signals that return True should be flagged/quarantined.
    """
    try:
        window_start, window_end = _trading_day_window(signal_date, _PROXIMITY_DAYS)

        response = (
            sb.table(_TABLE)
            .select("id")
            .eq("ticker", ticker.upper().strip())
            .gte("action_date", window_start.isoformat())
            .lte("action_date", window_end.isoformat())
            .limit(1)
            .execute()
        )

        return len(response.data) > 0

    except Exception as e:
        logger.error(f"corporate_action_proximity check failed for {ticker} on {signal_date}: {e}")
        # Fail safe: flag the signal when we can't confirm
        return True
