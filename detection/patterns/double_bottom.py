from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import pandas as pd


PRIMARY_HOLD_DAYS = 90


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (0, 1, 2):
        return None

    if len(df) < 120:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    lookback_start = last - 119  # 120 bars inclusive

    # ------------------------------------------------------------------ #
    # Find exactly 2 troughs within lookback                              #
    # ------------------------------------------------------------------ #
    troughs = [
        t for t in pivots.get("troughs", [])
        if t["index"] >= lookback_start
    ]

    if len(troughs) < 2:
        return None

    # Use the last 2 troughs chronologically
    t1, t2 = troughs[-2], troughs[-1]

    # Must be >= 10 bars apart
    if t2["index"] - t1["index"] < 10:
        return None

    # ------------------------------------------------------------------ #
    # Prior downtrend: close 120 bars ago must be > first trough * 1.10  #
    # ------------------------------------------------------------------ #
    prior_close = float(closes.iloc[lookback_start])
    if prior_close <= t1["price"] * 1.10:
        return None

    # ------------------------------------------------------------------ #
    # Trough equality: within 3%                                          #
    # ------------------------------------------------------------------ #
    trough_diff_pct = abs(t2["price"] - t1["price"]) / t1["price"] * 100
    if trough_diff_pct > 3.0:
        return None

    # ------------------------------------------------------------------ #
    # Neckline: highest close between the two troughs                     #
    # ------------------------------------------------------------------ #
    between_slice = closes.iloc[t1["index"]: t2["index"] + 1]
    neckline      = float(between_slice.max())

    # ------------------------------------------------------------------ #
    # Second-bottom volume: compare avg volume around each trough         #
    # Use a 3-bar window centred on each trough index                     #
    # ------------------------------------------------------------------ #
    def _trough_vol(idx: int) -> float:
        s = max(0, idx - 1)
        e = min(last, idx + 1)
        return float(volumes.iloc[s: e + 1].mean())

    t1_vol = _trough_vol(t1["index"])
    t2_vol = _trough_vol(t2["index"])

    second_vol_ratio = (t2_vol / t1_vol) if t1_vol > 0 else 1.0

    # Second bottom must show lower or equal volume (not required to fail,
    # but impacts confidence scoring)

    # ------------------------------------------------------------------ #
    # Rally volume from second bottom: average volume from t2 to current  #
    # should be higher than during the second trough formation            #
    # (soft check — does not gate detection)                              #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    resistance    = round(neckline, 2)
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
    score = 55

    if trough_diff_pct <= 1.5:
        score += 15
    elif trough_diff_pct <= 3.0:
        score += 8

    if second_vol_ratio < 0.70:
        score += 15
    elif second_vol_ratio <= 0.90:
        score += 8

    if stage in (0, 1, 2):
        score += 10

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    # ------------------------------------------------------------------ #
    # Stop loss: lower of two troughs * 0.98                              #
    # ------------------------------------------------------------------ #
    lower_trough = min(t1["price"], t2["price"])
    sl_price = round(lower_trough * 0.98, 2)

    return {
        "pattern_key":      "double_bottom",
        "state":            state,
        "confidence_score": score,
        "resistance":       resistance,
        "sl_price":         sl_price,
        "trough_diff_pct":  round(trough_diff_pct, 2),
        "second_vol_ratio": round(second_vol_ratio, 2),
        "stage":            stage,
    }
