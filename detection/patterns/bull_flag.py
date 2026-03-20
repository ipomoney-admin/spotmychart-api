from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import numpy as np
import pandas as pd


def _linear_slope(values: list) -> float:
    """Return slope of OLS fit over the given sequence."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


PRIMARY_HOLD_DAYS = 90


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage != 2:
        return None

    if len(df) < 40:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    # ------------------------------------------------------------------ #
    # Find pole: best qualifying rally ending within last 5-35 bars       #
    # ------------------------------------------------------------------ #
    pole = None

    for pole_end_offset in range(5, 36):
        pole_end_idx = last - pole_end_offset
        if pole_end_idx < 20:
            break

        pole_end_price = closes.iloc[pole_end_idx]

        # Search for pole start: 5-20 bars before pole_end
        for pole_len in range(5, 21):
            pole_start_idx = pole_end_idx - pole_len
            if pole_start_idx < 20:
                break

            pole_start_price = closes.iloc[pole_start_idx]
            if pole_start_price <= 0:
                continue

            gain_pct = (pole_end_price - pole_start_price) / pole_start_price * 100
            if gain_pct < 15.0:
                continue

            # Pole must be the high point in the range (no lower low partway)
            pole_range = closes.iloc[pole_start_idx: pole_end_idx + 1]
            if closes.iloc[pole_end_idx] < pole_range.max() * 0.98:
                continue

            # Volume: pole avg vs prior 20-bar avg
            prior_vol_avg = volumes.iloc[pole_start_idx - 20: pole_start_idx].mean()
            pole_vol_avg  = volumes.iloc[pole_start_idx: pole_end_idx + 1].mean()
            if prior_vol_avg == 0 or pole_vol_avg < prior_vol_avg * 1.5:
                continue

            pole = {
                "start_idx":   pole_start_idx,
                "end_idx":     pole_end_idx,
                "start_price": float(pole_start_price),
                "end_price":   float(pole_end_price),
                "gain_pct":    round(gain_pct, 2),
                "vol_avg":     float(pole_vol_avg),
            }
            break  # take first valid pole_len for this pole_end_offset

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

    if not (5 <= flag_len <= 20):
        return None

    flag_closes  = closes.iloc[flag_start: flag_end + 1].tolist()
    flag_volumes = volumes.iloc[flag_start: flag_end + 1]

    pole_height = pole["end_price"] - pole["start_price"]

    # Retrace: how far price has pulled back from pole top
    flag_low       = min(flag_closes)
    retrace_amount = pole["end_price"] - flag_low
    retrace_pct    = retrace_amount / pole_height * 100

    if retrace_pct > 50.0:
        return None

    # Flag volume dry-up
    flag_vol_avg = float(flag_volumes.mean())
    vol_ratio    = flag_vol_avg / pole["vol_avg"] if pole["vol_avg"] > 0 else 1.0
    if vol_ratio >= 0.75:
        return None

    # Flag slope must be negative (downward drift)
    slope = _linear_slope(flag_closes)
    if slope >= 0:
        return None

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    resistance    = pole["end_price"]
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
    # Confidence score                                                    #
    # ------------------------------------------------------------------ #
    score = 50

    if pole["gain_pct"] >= 25.0:
        score += 15
    else:
        score += 8

    if vol_ratio < 0.50:
        score += 15
    elif vol_ratio <= 0.75:
        score += 5

    if stage == 2:
        score += 10

    if retrace_pct <= 35.0:
        score += 10
    elif retrace_pct <= 50.0:
        score += 5

    if state == "confirmed":
        score += 5

    score = min(score, 100)

    # ------------------------------------------------------------------ #
    # Stop loss: pole low * 0.98                                         #
    # ------------------------------------------------------------------ #
    sl_price = round(pole["start_price"] * 0.98, 2)

    return {
        "pattern_key":      "bull_flag",
        "state":            state,
        "confidence_score": score,
        "resistance":       round(resistance, 2),
        "sl_price":         sl_price,
        "pole_gain_pct":    pole["gain_pct"],
        "flag_retrace_pct": round(retrace_pct, 2),
        "stage":            stage,
    }
