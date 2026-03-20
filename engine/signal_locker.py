import json
import logging
from datetime import date

from core.redis_client import r
from core.supabase_client import sb

logger = logging.getLogger(__name__)

_TABLE = "smc_active_locks"


def _redis_key(ticker: str) -> str:
    return f"smc:lock:{ticker.upper().strip()}"


def _ttl_seconds(expires_at: date) -> int:
    today = date.today()
    delta = (expires_at - today).days
    return max(delta * 86400, 1)


# ------------------------------------------------------------------ #
# is_locked                                                           #
# ------------------------------------------------------------------ #

def is_locked(ticker: str) -> bool:
    ticker = ticker.upper().strip()

    # Primary: Redis
    if r is not None:
        try:
            return r.exists(_redis_key(ticker)) == 1
        except Exception as e:
            logger.warning(f"Redis is_locked failed for {ticker}, falling back to Supabase: {e}")

    # Fallback: Supabase
    try:
        today_iso = date.today().isoformat()
        resp = (
            sb.table(_TABLE)
            .select("id")
            .eq("ticker", ticker)
            .gte("expires_at", today_iso)
            .limit(1)
            .execute()
        )
        return len(resp.data) > 0
    except Exception as e:
        logger.error(f"Supabase is_locked fallback failed for {ticker}: {e}")
        return False


# ------------------------------------------------------------------ #
# acquire_lock                                                        #
# ------------------------------------------------------------------ #

def acquire_lock(ticker: str, pattern_key: str, signal_id: str, expires_at: date) -> bool:
    ticker = ticker.upper().strip()

    if is_locked(ticker):
        return False

    payload = json.dumps({
        "pattern_key": pattern_key,
        "signal_id":   signal_id,
        "expires_at":  expires_at.isoformat(),
    })

    # Write to Redis
    if r is not None:
        try:
            r.set(_redis_key(ticker), payload, ex=_ttl_seconds(expires_at))
        except Exception as e:
            logger.warning(f"Redis acquire_lock failed for {ticker}: {e}")

    # Write to Supabase (source of truth)
    try:
        sb.table(_TABLE).upsert({
            "ticker":      ticker,
            "pattern_key": pattern_key,
            "signal_id":   signal_id,
            "expires_at":  expires_at.isoformat(),
        }, on_conflict="ticker").execute()
    except Exception as e:
        logger.error(f"Supabase acquire_lock failed for {ticker}: {e}")
        # Roll back Redis entry to avoid ghost lock
        if r is not None:
            try:
                r.delete(_redis_key(ticker))
            except Exception:
                pass
        return False

    return True


# ------------------------------------------------------------------ #
# release_lock                                                        #
# ------------------------------------------------------------------ #

def release_lock(ticker: str) -> bool:
    ticker = ticker.upper().strip()
    success = True

    if r is not None:
        try:
            r.delete(_redis_key(ticker))
        except Exception as e:
            logger.warning(f"Redis release_lock failed for {ticker}: {e}")
            success = False

    try:
        sb.table(_TABLE).delete().eq("ticker", ticker).execute()
    except Exception as e:
        logger.error(f"Supabase release_lock failed for {ticker}: {e}")
        success = False

    return success
