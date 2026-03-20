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


PRIMARY_HOLD_DAYS = 90


def detect(
    df: pd.DataFrame,
    stage: int,
    pivots: dict,
    segment: str = "large",
) -> Optional[dict]:
    if stage not in (1, 2):
        return None

    if len(df) < 55:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    peaks   = pivots.get("peaks", [])
    troughs = pivots.get("troughs", [])

    if len(peaks) < 2 or len(troughs) < 2:
        return None

    # ------------------------------------------------------------------ #
    # Identify triangle window: use last 2-3 peaks to anchor the pattern  #
    # ------------------------------------------------------------------ #
    # Take up to the last 3 peaks to define the flat top
    anchor_peaks = peaks[-3:] if len(peaks) >= 3 else peaks[-2:]

    triangle_start_idx = anchor_peaks[0]["index"]
    triangle_len       = last - triangle_start_idx + 1

    if not (15 <= triangle_len <= 60):
        return None

    # ------------------------------------------------------------------ #
    # Flat top: slope of anchor peaks must be near-zero                   #
    # < 0.2% of avg peak price per bar                                    #
    # ------------------------------------------------------------------ #
    peak_indices = [p["index"] for p in anchor_peaks]
    peak_prices  = [p["price"] for p in anchor_peaks]
    avg_peak_price = sum(peak_prices) / len(peak_prices)

    flat_top_slope = _slope(peak_indices, peak_prices)
    slope_pct_per_bar = abs(flat_top_slope) / avg_peak_price * 100

    if slope_pct_per_bar >= 0.2:
        return None

    # ------------------------------------------------------------------ #
    # Rising lows: troughs after triangle start, each higher than last    #
    # ------------------------------------------------------------------ #
    triangle_troughs = [t for t in troughs if t["index"] >= triangle_start_idx]

    if len(triangle_troughs) < 2:
        return None

    # Check each trough is higher than the previous
    for i in range(1, len(triangle_troughs)):
        if triangle_troughs[i]["price"] <= triangle_troughs[i - 1]["price"]:
            return None

    trough_indices = [t["index"] for t in triangle_troughs]
    trough_prices  = [t["price"] for t in triangle_troughs]
    rising_low_slope = _slope(trough_indices, trough_prices)

    if rising_low_slope <= 0:
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
    # Resistance = average of recent anchor peaks                         #
    # ------------------------------------------------------------------ #
    resistance    = round(avg_peak_price, 2)
    current_close = float(closes.iloc[last])

    breakout_multiplier = 1.8 if segment == "small" else 1.5
    ma20_vol  = float(volumes.iloc[max(0, last - 19): last + 1].mean())
    today_vol = float(volumes.iloc[last])

    below_pct = (resistance - current_close) / resistance * 100

    if current_close >= resistance * 1.005 and today_vol >= ma20_vol * breakout_multiplier:
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
    score = 55

    if len(anchor_peaks) >= 3 and len(triangle_troughs) >= 3:
        score += 5

    if vol_ratio < 0.60:
        score += 10
    elif vol_ratio <= 0.80:
        score += 5

    if stage in (1, 2):
        score += 10

    if slope_pct_per_bar < 0.1:
        score += 5

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    # ------------------------------------------------------------------ #
    # Stop loss: lowest trough in triangle * 0.98                         #
    # ------------------------------------------------------------------ #
    lowest_trough = min(t["price"] for t in triangle_troughs)
    sl_price = round(lowest_trough * 0.98, 2)

    return {
        "pattern_key":      "ascending_triangle",
        "state":            state,
        "confidence_score": score,
        "resistance":       resistance,
        "sl_price":         sl_price,
        "flat_top_slope":   round(flat_top_slope, 4),
        "rising_low_slope": round(rising_low_slope, 4),
        "stage":            stage,
    }
