from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import numpy as np
import pandas as pd


def _linear_slope(values: list) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


PRIMARY_HOLD_DAYS = 30


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage != 4:
        return None

    if len(df) < 40:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    # ------------------------------------------------------------------ #
    # Find pole: best qualifying drop ending within last 5-35 bars        #
    # ------------------------------------------------------------------ #
    pole = None

    for pole_end_offset in range(5, 36):
        pole_end_idx = last - pole_end_offset
        if pole_end_idx < 20:
            break

        pole_end_price = closes.iloc[pole_end_idx]

        for pole_len in range(5, 21):
            pole_start_idx = pole_end_idx - pole_len
            if pole_start_idx < 20:
                break

            pole_start_price = closes.iloc[pole_start_idx]
            if pole_start_price <= 0:
                continue

            drop_pct = (pole_start_price - pole_end_price) / pole_start_price * 100
            if drop_pct < 15.0:
                continue

            # Pole low should be near its end (not mid-pole)
            pole_range = closes.iloc[pole_start_idx: pole_end_idx + 1]
            if closes.iloc[pole_end_idx] > pole_range.min() * 1.02:
                continue

            prior_vol_avg = volumes.iloc[pole_start_idx - 20: pole_start_idx].mean()
            pole_vol_avg  = volumes.iloc[pole_start_idx: pole_end_idx + 1].mean()
            if prior_vol_avg == 0 or pole_vol_avg < prior_vol_avg * 1.5:
                continue

            pole = {
                "start_idx":   pole_start_idx,
                "end_idx":     pole_end_idx,
                "start_price": float(pole_start_price),   # pole high
                "end_price":   float(pole_end_price),     # pole low
                "drop_pct":    round(drop_pct, 2),
                "vol_avg":     float(pole_vol_avg),
            }
            break

        if pole:
            break

    if pole is None:
        return None

    # ------------------------------------------------------------------ #
    # Flag: bars from pole_end+1 to current                              #
    # ------------------------------------------------------------------ #
    flag_start = pole["end_idx"] + 1
    flag_end   = last
    flag_len   = flag_end - flag_start + 1

    if not (3 <= flag_len <= 15):
        return None

    flag_closes  = closes.iloc[flag_start: flag_end + 1].tolist()
    flag_volumes = volumes.iloc[flag_start: flag_end + 1]

    pole_drop = pole["start_price"] - pole["end_price"]

    # Retrace: how far price has bounced from pole low
    flag_high      = max(flag_closes)
    retrace_amount = flag_high - pole["end_price"]
    retrace_pct    = retrace_amount / pole_drop * 100

    if retrace_pct > 50.0:
        return None

    # Flag volume dry-up
    flag_vol_avg = float(flag_volumes.mean())
    vol_ratio    = flag_vol_avg / pole["vol_avg"] if pole["vol_avg"] > 0 else 1.0
    if vol_ratio >= 0.75:
        return None

    # Flag slope must be positive (upward drift / counter-trend bounce)
    slope = _linear_slope(flag_closes)
    if slope <= 0:
        return None

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    support       = pole["end_price"]
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

    if pole["drop_pct"] >= 25.0:
        score += 15
    else:
        score += 8

    if vol_ratio < 0.50:
        score += 15
    elif vol_ratio <= 0.75:
        score += 5

    if stage == 4:
        score += 10

    if retrace_pct <= 35.0:
        score += 10
    elif retrace_pct <= 50.0:
        score += 5

    if state == "confirmed":
        score += 5

    score = min(score, 100)

    # SL = pole high * 1.02 (stop above the pole top)
    sl_price = round(pole["start_price"] * 1.02, 2)

    return {
        "pattern_key":      "bear_flag",
        "state":            state,
        "confidence_score": score,
        "support":          round(support, 2),
        "sl_price":         sl_price,
        "pole_drop_pct":    pole["drop_pct"],
        "flag_retrace_pct": round(retrace_pct, 2),
        "stage":            stage,
    }
