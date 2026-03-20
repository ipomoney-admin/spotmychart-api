from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import numpy as np
import pandas as pd


PRIMARY_HOLD_DAYS = 15


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (3, 4):
        return None

    if len(df) < 100:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    # ------------------------------------------------------------------ #
    # Try windows from longest to shortest (prefer 200, min 100)         #
    # ------------------------------------------------------------------ #
    best = None

    for window in range(200, 99, -1):
        start_idx = last - window + 1
        if start_idx < 0:
            continue

        window_closes = closes.iloc[start_idx: last + 1].values
        x = np.arange(len(window_closes), dtype=float)

        coeffs = np.polyfit(x, window_closes, 2)
        if coeffs[0] >= 0:          # must be concave down (inverted U)
            continue

        fitted    = np.polyval(coeffs, x)
        residuals = np.abs(window_closes - fitted)
        noise_pct = float(residuals.mean()) / float(window_closes.mean()) * 100

        if noise_pct >= 8.0:
            continue

        best = {
            "start_idx":  start_idx,
            "window":     window,
            "closes":     window_closes,
            "noise_pct":  round(noise_pct, 2),
            "arc_high":   float(window_closes.max()),
        }
        break   # longest qualifying window wins

    if best is None:
        return None

    # ------------------------------------------------------------------ #
    # Support = left rim (close at start of window)                      #
    # ------------------------------------------------------------------ #
    support       = round(float(best["closes"][0]), 2)
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

    if best["noise_pct"] < 3.0:
        score += 15
    elif best["noise_pct"] <= 6.0:
        score += 8
    else:
        score += 3

    if stage in (3, 4):
        score += 10

    if best["window"] >= 150:
        score += 10
    else:
        score += 5

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    sl_price = round(best["arc_high"] * 1.02, 2)

    return {
        "pattern_key":      "rounding_top",
        "state":            state,
        "confidence_score": score,
        "support":          support,
        "sl_price":         sl_price,
        "noise_pct":        best["noise_pct"],
        "length_bars":      best["window"],
        "stage":            stage,
    }
