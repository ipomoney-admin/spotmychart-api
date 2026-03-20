from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import numpy as np
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

    peaks   = pivots.get("peaks",   [])
    troughs = pivots.get("troughs", [])

    if len(peaks) < 2:
        return None

    # ------------------------------------------------------------------ #
    # Identify inverted cup: left lip, arc high (peak between lips),      #
    # right lip. Try consecutive trough pairs as lips.                    #
    # ------------------------------------------------------------------ #
    cup = None

    # Use troughs as lips — inverted cup drops from left lip trough,
    # arcs up through a peak, then returns to the right lip trough.
    if len(troughs) < 2:
        return None

    for i in range(len(troughs) - 1):
        left_lip  = troughs[i]
        right_lip = troughs[i + 1]

        cup_len = right_lip["index"] - left_lip["index"]
        if not (35 <= cup_len <= 120):
            continue

        # Lip match: right lip within 5% of left lip
        lip_diff_pct = abs(right_lip["price"] - left_lip["price"]) / left_lip["price"] * 100
        if lip_diff_pct > 5.0:
            continue

        # Arc high: highest peak between the two lips
        arc_peaks = [p for p in peaks if left_lip["index"] < p["index"] < right_lip["index"]]
        if not arc_peaks:
            continue
        arc_high = max(arc_peaks, key=lambda p: p["price"])

        # Cup depth: 12-40% from lips to arc high
        avg_lip_price = (left_lip["price"] + right_lip["price"]) / 2
        cup_depth_pct = (arc_high["price"] - avg_lip_price) / avg_lip_price * 100
        if not (12.0 <= cup_depth_pct <= 40.0):
            continue

        # Cup shape: fit parabola — must be concave down (coefficient < 0)
        cup_slice = closes.iloc[left_lip["index"]: right_lip["index"] + 1].values
        x = np.arange(len(cup_slice), dtype=float)
        coeffs = np.polyfit(x, cup_slice, 2)

        if coeffs[0] >= 0:          # must be concave down
            continue

        fitted    = np.polyval(coeffs, x)
        residuals = np.abs(cup_slice - fitted)
        noise_pct = float(residuals.mean()) / float(cup_slice.mean()) * 100
        if noise_pct >= 12.0:
            continue

        cup = {
            "left_lip":     left_lip,
            "right_lip":    right_lip,
            "arc_high":     arc_high,
            "cup_depth_pct": round(cup_depth_pct, 2),
            "cup_len":      cup_len,
            "lip_diff_pct": round(lip_diff_pct, 2),
            "cup_vol_avg":  float(volumes.iloc[left_lip["index"]: right_lip["index"] + 1].mean()),
        }
        # Keep iterating — last valid cup is the most recent

    if cup is None:
        return None

    # ------------------------------------------------------------------ #
    # Handle: bars from right lip to current                              #
    # ------------------------------------------------------------------ #
    handle_start = cup["right_lip"]["index"] + 1
    handle_len   = last - handle_start + 1

    if not (5 <= handle_len <= 20):
        return None

    handle_closes  = closes.iloc[handle_start: last + 1].tolist()
    handle_volumes = volumes.iloc[handle_start: last + 1]

    # Handle: upward drift (positive slope — weak bounce before breakdown)
    x = np.arange(len(handle_closes), dtype=float)
    slope = float(np.polyfit(x, handle_closes, 1)[0])
    if slope <= 0:
        return None

    # Handle volume dry-up
    handle_vol_avg   = float(handle_volumes.mean())
    handle_vol_ratio = handle_vol_avg / cup["cup_vol_avg"] if cup["cup_vol_avg"] > 0 else 1.0
    vol_dry          = handle_vol_ratio < 0.65

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    support       = round(cup["right_lip"]["price"], 2)
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
    score = 50

    score += 15 if vol_dry else 5

    if stage in (3, 4):
        score += 10

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    sl_price = round(cup["arc_high"]["price"] * 1.02, 2)

    return {
        "pattern_key":      "inv_cup_handle",
        "state":            state,
        "confidence_score": score,
        "support":          support,
        "sl_price":         sl_price,
        "cup_depth_pct":    cup["cup_depth_pct"],
        "cup_length_bars":  cup["cup_len"],
        "stage":            stage,
    }
