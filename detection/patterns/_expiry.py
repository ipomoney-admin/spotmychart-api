"""
Signal expiry helpers shared across all pattern detectors.

bars_since_cross   — how long ago price first crossed the breakout threshold
consecutive_forming_bars — how many consecutive bars price has been in the
                           0-5% forming zone (used to expire stale forming signals)
"""

import pandas as pd


def bars_since_cross(
    closes: pd.Series,
    last: int,
    level: float,
    direction: str = "bull",
    max_lookback: int = 180,
) -> int:
    """
    Scan forward from (last - max_lookback) to find the FIRST bar where price
    crossed the confirmed threshold.  Returns the number of bars that have
    elapsed since that first crossing.  Returns 0 if no crossing found within
    the lookback window (treat as just-crossed so callers apply no expiry).

    direction="bull": confirmed when close >= level * 1.005
    direction="bear": confirmed when close <= level * 0.995
    """
    threshold = level * 1.005 if direction == "bull" else level * 0.995
    start = max(0, last - max_lookback)

    for i in range(start, last + 1):
        price = float(closes.iloc[i])
        if direction == "bull" and price >= threshold:
            return last - i
        if direction == "bear" and price <= threshold:
            return last - i

    return 0  # current bar just crossed — no expiry


def consecutive_forming_bars(
    closes: pd.Series,
    last: int,
    level: float,
    direction: str = "bull",
) -> int:
    """
    Count consecutive bars ending at `last` where price sits in the
    0-5% forming zone below (bull) or above (bear) the pattern level.
    Breaks on the first bar outside the zone when scanning backward.

    direction="bull": forming when 0 <= (level - close) / level * 100 <= 5
    direction="bear": forming when 0 <= (close - level) / level * 100 <= 5
    """
    count = 0
    for i in range(last, -1, -1):
        price = float(closes.iloc[i])
        if direction == "bull":
            pct = (level - price) / level * 100
        else:
            pct = (price - level) / level * 100

        if 0.0 <= pct <= 5.0:
            count += 1
        else:
            break

    return count
