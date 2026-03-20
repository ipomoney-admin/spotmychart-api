import logging
import time
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_EMPTY_DF = pd.DataFrame(columns=_REQUIRED_COLUMNS)


def _nse_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if not ticker.endswith(".NS"):
        ticker += ".NS"
    return ticker


def fetch_ohlcv(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    symbol = _nse_ticker(ticker)
    last_error = None

    for attempt in range(3):
        try:
            raw = yf.download(
                symbol,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=True,   # adjusts OHLCV for splits/dividends
                progress=False,
                threads=False,
            )

            if raw.empty:
                logger.warning(f"No data returned for {symbol} ({start_date} to {end_date})")
                return _EMPTY_DF

            # yfinance returns a MultiIndex when auto_adjust=True on some versions
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.rename(columns=str.lower)
            raw.index.name = "date"
            raw = raw.reset_index()
            raw["date"] = pd.to_datetime(raw["date"]).dt.date

            return raw[_REQUIRED_COLUMNS]

        except Exception as e:
            last_error = e
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(f"Attempt {attempt + 1}/3 failed for {symbol}: {e}. Retrying in {wait}s.")
            time.sleep(wait)

    logger.error(f"All retries exhausted for {symbol}: {last_error}")
    return _EMPTY_DF
