from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import pandas as pd


PRIMARY_HOLD_DAYS = 30


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (3, 4):
        return None

    if len(df) < 120:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    lookback_start = last - 119  # 120 bars inclusive

    # ------------------------------------------------------------------ #
    # Find 2 peaks within lookback — use last 2                          #
    # ------------------------------------------------------------------ #
    peaks = [
        p for p in pivots.get("peaks", [])
        if p["index"] >= lookback_start
    ]

    if len(peaks) < 2:
        return None

    p1, p2 = peaks[-2], peaks[-1]

    # Must be >= 10 bars apart
    if p2["index"] - p1["index"] < 10:
        return None

    # ------------------------------------------------------------------ #
    # Prior uptrend: close 120 bars ago < first peak * 0.90              #
    # ------------------------------------------------------------------ #
    prior_close = float(closes.iloc[lookback_start])
    if prior_close >= p1["price"] * 0.90:
        return None

    # ------------------------------------------------------------------ #
    # Peak equality: within 3%                                            #
    # ------------------------------------------------------------------ #
    peak_diff_pct = abs(p2["price"] - p1["price"]) / p1["price"] * 100
    if peak_diff_pct > 3.0:
        return None

    # ------------------------------------------------------------------ #
    # Neckline: lowest close between the two peaks                        #
    # ------------------------------------------------------------------ #
    between_slice = closes.iloc[p1["index"]: p2["index"] + 1]
    neckline      = float(between_slice.min())

    # ------------------------------------------------------------------ #
    # Second peak volume: 3-bar window centred on each peak              #
    # ------------------------------------------------------------------ #
    def _peak_vol(idx: int) -> float:
        s = max(0, idx - 1)
        e = min(last, idx + 1)
        return float(volumes.iloc[s: e + 1].mean())

    p1_vol = _peak_vol(p1["index"])
    p2_vol = _peak_vol(p2["index"])

    second_vol_ratio = (p2_vol / p1_vol) if p1_vol > 0 else 1.0

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

    if peak_diff_pct <= 1.5:
        score += 15
    elif peak_diff_pct <= 3.0:
        score += 8

    if second_vol_ratio < 0.70:
        score += 15
    elif second_vol_ratio <= 0.90:
        score += 8

    if stage in (3, 4):
        score += 10

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    # SL = higher of two peaks * 1.02
    higher_peak = max(p1["price"], p2["price"])
    sl_price = round(higher_peak * 1.02, 2)

    return {
        "pattern_key":      "double_top",
        "state":            state,
        "confidence_score": score,
        "support":          support,
        "sl_price":         sl_price,
        "peak_diff_pct":    round(peak_diff_pct, 2),
        "second_vol_ratio": round(second_vol_ratio, 2),
        "stage":            stage,
    }
