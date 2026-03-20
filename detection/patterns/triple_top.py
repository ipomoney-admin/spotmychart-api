from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import pandas as pd


PRIMARY_HOLD_DAYS = 15


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (3, 4):
        return None

    if len(df) < 150:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    lookback_start = last - 149  # 150 bars inclusive

    # ------------------------------------------------------------------ #
    # Collect peaks within lookback — use last 3                         #
    # ------------------------------------------------------------------ #
    peaks = [
        p for p in pivots.get("peaks", [])
        if p["index"] >= lookback_start
    ]

    if len(peaks) < 3:
        return None

    p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]

    # Each peak >= 8 bars apart
    if (p2["index"] - p1["index"]) < 8 or (p3["index"] - p2["index"]) < 8:
        return None

    # ------------------------------------------------------------------ #
    # Prior uptrend: close 150 bars ago < lowest peak * 0.90             #
    # ------------------------------------------------------------------ #
    highest_price = max(p1["price"], p2["price"], p3["price"])
    lowest_peak   = min(p1["price"], p2["price"], p3["price"])
    prior_close   = float(closes.iloc[lookback_start])

    if prior_close >= lowest_peak * 0.90:
        return None

    # ------------------------------------------------------------------ #
    # Peak equality: all 3 within 3% of each other                       #
    # ------------------------------------------------------------------ #
    prices         = [p1["price"], p2["price"], p3["price"]]
    peak_range_pct = (max(prices) - min(prices)) / min(prices) * 100

    if peak_range_pct > 3.0:
        return None

    # ------------------------------------------------------------------ #
    # Volume: 3-bar window centred on each peak                          #
    # ------------------------------------------------------------------ #
    def _peak_vol(idx: int) -> float:
        s = max(0, idx - 1)
        e = min(last, idx + 1)
        return float(volumes.iloc[s: e + 1].mean())

    v1 = _peak_vol(p1["index"])
    v2 = _peak_vol(p2["index"])
    v3 = _peak_vol(p3["index"])

    volume_declining = v1 >= v2 >= v3

    # ------------------------------------------------------------------ #
    # Neckline: lowest close across full range of all 3 peaks            #
    # ------------------------------------------------------------------ #
    neckline_slice = closes.iloc[p1["index"]: p3["index"] + 1]
    neckline       = float(neckline_slice.min())

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    support       = round(neckline, 2)
    current_close = float(closes.iloc[last])
    ma20_vol      = float(volumes.iloc[max(0, last - 19): last + 1].mean())
    today_vol     = float(volumes.iloc[last])

    above_pct = (current_close - support) / support * 100

    if current_close <= support * 0.995 and today_vol >= ma20_vol * 1.5:
        state = "confirmed"
    elif 0.0 <= above_pct <= 5.0:
        state = "forming"
    else:
        return None

    # ------------------------------------------------------------------ #
    # Expiry checks                                                        #
    # ------------------------------------------------------------------ #
    if state == "confirmed":
        since = bars_since_cross(closes, last, support, direction="bear")
        if since > PRIMARY_HOLD_DAYS:
            return None
    elif state == "forming":
        if consecutive_forming_bars(closes, last, support, direction="bear") > 30:
            return None

    # ------------------------------------------------------------------ #
    # Confidence score                                                     #
    # ------------------------------------------------------------------ #
    score = 55

    if peak_range_pct <= 1.5:
        score += 15
    elif peak_range_pct <= 3.0:
        score += 8

    score += 20 if volume_declining else 5

    if stage in (3, 4):
        score += 10

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    sl_price = round(highest_price * 1.02, 2)

    return {
        "pattern_key":      "triple_top",
        "state":            state,
        "confidence_score": score,
        "support":          support,
        "sl_price":         sl_price,
        "peak_range_pct":   round(peak_range_pct, 2),
        "volume_declining": volume_declining,
        "stage":            stage,
    }
