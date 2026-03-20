from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import numpy as np
import pandas as pd


def _slope(indices: list, prices: list) -> float:
    if len(indices) < 2:
        return 0.0
    x = np.array(indices, dtype=float)
    y = np.array(prices, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _project_trendline(indices: list, prices: list, target_idx: int) -> float:
    if len(indices) < 2:
        return prices[-1]
    x = np.array(indices, dtype=float)
    y = np.array(prices, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    return float(np.polyval(coeffs, target_idx))


PRIMARY_HOLD_DAYS = 30


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (3, 4):
        return None

    if len(df) < 55:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    peaks   = pivots.get("peaks",   [])
    troughs = pivots.get("troughs", [])

    if len(peaks) < 2 or len(troughs) < 2:
        return None

    # ------------------------------------------------------------------ #
    # Anchor pivots: up to last 3 of each                                 #
    # ------------------------------------------------------------------ #
    anchor_peaks   = peaks[-3:]   if len(peaks)   >= 3 else peaks[-2:]
    anchor_troughs = troughs[-3:] if len(troughs) >= 3 else troughs[-2:]

    triangle_start_idx = min(anchor_peaks[0]["index"], anchor_troughs[0]["index"])
    triangle_len       = last - triangle_start_idx + 1

    if not (15 <= triangle_len <= 60):
        return None

    # ------------------------------------------------------------------ #
    # Trendline slopes: upper falling, lower rising — converging          #
    # ------------------------------------------------------------------ #
    peak_indices   = [p["index"] for p in anchor_peaks]
    peak_prices    = [p["price"] for p in anchor_peaks]
    trough_indices = [t["index"] for t in anchor_troughs]
    trough_prices  = [t["price"] for t in anchor_troughs]

    upper_slope = _slope(peak_indices,   peak_prices)
    lower_slope = _slope(trough_indices, trough_prices)

    if upper_slope >= 0 or lower_slope <= 0:
        return None

    # ------------------------------------------------------------------ #
    # Volume dry-up during triangle                                        #
    # ------------------------------------------------------------------ #
    vol_40_avg       = float(volumes.iloc[max(0, triangle_start_idx - 40): triangle_start_idx].mean())
    triangle_vol_avg = float(volumes.iloc[triangle_start_idx: last + 1].mean())

    if vol_40_avg == 0:
        return None

    vol_ratio = triangle_vol_avg / vol_40_avg
    if vol_ratio >= 0.80:
        return None

    # ------------------------------------------------------------------ #
    # Support = lower trendline projected to current bar                  #
    # ------------------------------------------------------------------ #
    support       = round(_project_trendline(trough_indices, trough_prices, last), 2)
    current_close = float(closes.iloc[last])
    ma20_vol      = float(volumes.iloc[max(0, last - 19): last + 1].mean())
    today_vol     = float(volumes.iloc[last])

    if support <= 0:
        return None

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

    if vol_ratio < 0.80:
        score += 15

    if stage == 4:
        score += 10
    elif stage == 3:
        score += 5

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    # SL = most recent peak * 1.02
    sl_price = round(anchor_peaks[-1]["price"] * 1.02, 2)

    return {
        "pattern_key":      "sym_triangle_bear",
        "state":            state,
        "confidence_score": score,
        "support":          support,
        "sl_price":         sl_price,
        "upper_slope":      round(upper_slope, 4),
        "lower_slope":      round(lower_slope, 4),
        "stage":            stage,
    }
