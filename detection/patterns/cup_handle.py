from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import numpy as np
import pandas as pd


PRIMARY_HOLD_DAYS = 90


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (1, 2):
        return None

    if len(df) < 300:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    peaks   = pivots.get("peaks",   [])
    troughs = pivots.get("troughs", [])

    if len(peaks) < 2 or len(troughs) < 1:
        return None

    # ------------------------------------------------------------------ #
    # Identify cup: left lip = an earlier peak, base = trough between     #
    # two peaks, right lip = later peak. Try pairs of peaks.              #
    # ------------------------------------------------------------------ #
    cup = None

    for i in range(len(peaks) - 1):
        left_peak  = peaks[i]
        right_peak = peaks[i + 1]

        cup_len = right_peak["index"] - left_peak["index"]
        if not (35 <= cup_len <= 260):
            continue

        # Lip match: right lip within 5% of left lip
        lip_diff_pct = abs(right_peak["price"] - left_peak["price"]) / left_peak["price"] * 100
        if lip_diff_pct > 5.0:
            continue

        # Cup base: lowest trough between the two peaks
        cup_troughs = [
            t for t in troughs
            if left_peak["index"] < t["index"] < right_peak["index"]
        ]
        if not cup_troughs:
            continue
        base = min(cup_troughs, key=lambda t: t["price"])

        # Cup depth: 12-33%
        cup_depth_pct = (left_peak["price"] - base["price"]) / left_peak["price"] * 100
        if not (12.0 <= cup_depth_pct <= 33.0):
            continue

        # Prior rise >= 30% before cup start (check 200 bars before left lip)
        prior_start = max(0, left_peak["index"] - 300)
        prior_end   = max(0, left_peak["index"] - 200)
        if prior_end <= prior_start:
            continue
        prior_close = float(closes.iloc[prior_start])
        if prior_close <= 0:
            continue
        prior_rise = (left_peak["price"] - prior_close) / prior_close * 100
        if prior_rise < 30.0:
            continue

        # Cup shape: fit parabola to closes between the two lips
        cup_slice = closes.iloc[left_peak["index"]: right_peak["index"] + 1].values
        x = np.arange(len(cup_slice), dtype=float)
        coeffs = np.polyfit(x, cup_slice, 2)

        if coeffs[0] <= 0:          # must be concave up
            continue

        fitted     = np.polyval(coeffs, x)
        residuals  = np.abs(cup_slice - fitted)
        noise_pct  = float(residuals.mean()) / float(cup_slice.mean()) * 100
        if noise_pct >= 12.0:
            continue

        cup = {
            "left_peak":     left_peak,
            "right_peak":    right_peak,
            "base":          base,
            "cup_depth_pct": round(cup_depth_pct, 2),
            "cup_len":       cup_len,
            "lip_diff_pct":  round(lip_diff_pct, 2),
            "noise_pct":     round(noise_pct, 2),
            "cup_vol_avg":   float(volumes.iloc[left_peak["index"]: right_peak["index"] + 1].mean()),
        }
        # Keep the most recent valid cup
    # End loop — `cup` holds the last (most recent) qualifying formation

    if cup is None:
        return None

    # ------------------------------------------------------------------ #
    # Handle: bars from right lip to current                              #
    # ------------------------------------------------------------------ #
    handle_start = cup["right_peak"]["index"] + 1
    handle_len   = last - handle_start + 1

    if not (5 <= handle_len <= 20):
        return None

    handle_closes  = closes.iloc[handle_start: last + 1]
    handle_volumes = volumes.iloc[handle_start: last + 1]

    handle_low     = float(handle_closes.min())
    right_lip      = cup["right_peak"]["price"]
    cup_midpoint   = cup["base"]["price"] + (right_lip - cup["base"]["price"]) / 2

    # Handle must form in upper half of cup (above midpoint)
    if handle_low < cup_midpoint:
        return None

    # Handle depth < 15%
    handle_depth_pct = (right_lip - handle_low) / right_lip * 100
    if handle_depth_pct >= 15.0:
        return None

    # Handle volume dry-up
    handle_vol_avg = float(handle_volumes.mean())
    handle_vol_ratio = handle_vol_avg / cup["cup_vol_avg"] if cup["cup_vol_avg"] > 0 else 1.0
    if handle_vol_ratio >= 0.65:
        return None

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    resistance    = round(right_lip, 2)
    current_close = float(closes.iloc[last])
    ma20_vol      = float(volumes.iloc[max(0, last - 19): last + 1].mean())
    today_vol     = float(volumes.iloc[last])

    below_pct = (resistance - current_close) / resistance * 100

    if current_close >= resistance * 1.005 and today_vol >= ma20_vol * 1.5:
        state = "confirmed"
    elif 0.0 <= below_pct <= 5.0:
        state = "forming"
    else:
        return None

    # ------------------------------------------------------------------ #
    # Expiry checks                                                        #
    # ------------------------------------------------------------------ #
    if state == "confirmed":
        since = bars_since_cross(closes, last, resistance, direction="bull")
        if since > PRIMARY_HOLD_DAYS:
            return None
    elif state == "forming":
        if consecutive_forming_bars(closes, last, resistance, direction="bull") > 30:
            return None

    # ------------------------------------------------------------------ #
    # Confidence score                                                     #
    # ------------------------------------------------------------------ #
    score = 50

    if 15.0 <= cup["cup_depth_pct"] <= 30.0:
        score += 10
    elif 12.0 <= cup["cup_depth_pct"] <= 33.0:
        score += 5

    if cup["lip_diff_pct"] <= 2.5:
        score += 10
    elif cup["lip_diff_pct"] <= 5.0:
        score += 5

    if handle_vol_ratio < 0.50:
        score += 15
    elif handle_vol_ratio <= 0.65:
        score += 5

    if stage in (1, 2):
        score += 10

    if cup["noise_pct"] < 6.0:
        score += 5

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    sl_price = round(handle_low * 0.98, 2)

    return {
        "pattern_key":      "cup_handle",
        "state":            state,
        "confidence_score": score,
        "resistance":       resistance,
        "sl_price":         sl_price,
        "cup_depth_pct":    cup["cup_depth_pct"],
        "cup_length_bars":  cup["cup_len"],
        "handle_depth_pct": round(handle_depth_pct, 2),
        "stage":            stage,
    }
