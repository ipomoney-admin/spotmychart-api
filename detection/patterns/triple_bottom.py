from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import pandas as pd


PRIMARY_HOLD_DAYS = 90


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (1, 2):
        return None

    if len(df) < 150:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    lookback_start = last - 149  # 150 bars inclusive

    # ------------------------------------------------------------------ #
    # Collect troughs within lookback, need exactly 3                     #
    # ------------------------------------------------------------------ #
    troughs = [
        t for t in pivots.get("troughs", [])
        if t["index"] >= lookback_start
    ]

    if len(troughs) < 3:
        return None

    # Use the last 3 troughs
    t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]

    # Each trough >= 8 bars apart from each other
    if (t2["index"] - t1["index"]) < 8 or (t3["index"] - t2["index"]) < 8:
        return None

    # ------------------------------------------------------------------ #
    # Prior downtrend: close 150 bars ago > lowest trough * 1.10         #
    # ------------------------------------------------------------------ #
    lowest_price = min(t1["price"], t2["price"], t3["price"])
    prior_close  = float(closes.iloc[lookback_start])

    if prior_close <= lowest_price * 1.10:
        return None

    # ------------------------------------------------------------------ #
    # Trough equality: all 3 within 3% of each other                     #
    # Range = (max - min) / min * 100                                     #
    # ------------------------------------------------------------------ #
    prices         = [t1["price"], t2["price"], t3["price"]]
    trough_range_pct = (max(prices) - min(prices)) / min(prices) * 100

    if trough_range_pct > 3.0:
        return None

    # ------------------------------------------------------------------ #
    # Volume: 3-bar window centred on each trough                        #
    # ------------------------------------------------------------------ #
    def _trough_vol(idx: int) -> float:
        s = max(0, idx - 1)
        e = min(last, idx + 1)
        return float(volumes.iloc[s: e + 1].mean())

    v1 = _trough_vol(t1["index"])
    v2 = _trough_vol(t2["index"])
    v3 = _trough_vol(t3["index"])

    volume_declining = v1 >= v2 >= v3

    # ------------------------------------------------------------------ #
    # Neckline: highest close across the full range of all 3 troughs     #
    # ------------------------------------------------------------------ #
    neckline_slice = closes.iloc[t1["index"]: t3["index"] + 1]
    neckline       = float(neckline_slice.max())

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

    if trough_range_pct <= 1.5:
        score += 15
    elif trough_range_pct <= 3.0:
        score += 8

    score += 20 if volume_declining else 5

    if stage in (1, 2):
        score += 10

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    sl_price = round(lowest_price * 0.98, 2)

    return {
        "pattern_key":      "triple_bottom",
        "state":            state,
        "confidence_score": score,
        "resistance":       resistance,
        "sl_price":         sl_price,
        "trough_range_pct": round(trough_range_pct, 2),
        "volume_declining": volume_declining,
        "stage":            stage,
    }
