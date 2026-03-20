from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import pandas as pd


PRIMARY_HOLD_DAYS = 90


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (0, 1, 2):
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
    ls, hd, rs = troughs[-3], troughs[-2], troughs[-1]  # left shoulder, head, right shoulder

    # Each trough >= 8 bars apart
    if (hd["index"] - ls["index"]) < 8 or (rs["index"] - hd["index"]) < 8:
        return None

    # ------------------------------------------------------------------ #
    # Head must be the LOWEST trough                                      #
    # ------------------------------------------------------------------ #
    if not (hd["price"] < ls["price"] and hd["price"] < rs["price"]):
        return None

    # ------------------------------------------------------------------ #
    # Shoulder equality: left and right shoulders within 8%               #
    # ------------------------------------------------------------------ #
    shoulder_diff_pct = abs(rs["price"] - ls["price"]) / ls["price"] * 100
    if shoulder_diff_pct > 8.0:
        return None

    # ------------------------------------------------------------------ #
    # Neckline: avg of peak between LS-HD and peak between HD-RS          #
    # ------------------------------------------------------------------ #
    ls_to_hd = closes.iloc[ls["index"]: hd["index"] + 1]
    hd_to_rs = closes.iloc[hd["index"]: rs["index"] + 1]

    left_neckline_pt  = float(ls_to_hd.max())
    right_neckline_pt = float(hd_to_rs.max())
    neckline          = (left_neckline_pt + right_neckline_pt) / 2

    # ------------------------------------------------------------------ #
    # Volume: 3-bar window centred on each trough                        #
    # ------------------------------------------------------------------ #
    def _trough_vol(idx: int) -> float:
        s = max(0, idx - 1)
        e = min(last, idx + 1)
        return float(volumes.iloc[s: e + 1].mean())

    v_ls = _trough_vol(ls["index"])
    v_hd = _trough_vol(hd["index"])
    v_rs = _trough_vol(rs["index"])

    # Correct pattern: head volume >= right shoulder volume (selling climax at head)
    volume_correct = v_hd >= v_rs

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

    score += 15 if volume_correct else 5

    if shoulder_diff_pct <= 4.0:
        score += 10
    elif shoulder_diff_pct <= 8.0:
        score += 5

    if stage in (0, 1, 2):
        score += 10

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    # SL = right shoulder low * 0.98
    sl_price = round(rs["price"] * 0.98, 2)

    return {
        "pattern_key":       "inv_head_shoulders",
        "state":             state,
        "confidence_score":  score,
        "resistance":        resistance,
        "sl_price":          sl_price,
        "shoulder_diff_pct": round(shoulder_diff_pct, 2),
        "head_price":        round(hd["price"], 2),
        "stage":             stage,
    }
