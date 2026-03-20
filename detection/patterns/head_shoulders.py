from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import pandas as pd


PRIMARY_HOLD_DAYS = 15


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage not in (3, 4):
        return None

    if len(df) < 150:
        return None

    closes  = df["close"].reset_index(drop=True)
    volumes = df["volume"].reset_index(drop=True)
    last    = len(df) - 1

    lookback_start = last - 149  # 150 bars inclusive

    # ------------------------------------------------------------------ #
    # Collect peaks within lookback — use last 3                         #
    # ------------------------------------------------------------------ #
    peaks = [
        p for p in pivots.get("peaks", [])
        if p["index"] >= lookback_start
    ]

    if len(peaks) < 3:
        return None

    ls, hd, rs = peaks[-3], peaks[-2], peaks[-1]  # left shoulder, head, right shoulder

    # Each peak >= 8 bars apart
    if (hd["index"] - ls["index"]) < 8 or (rs["index"] - hd["index"]) < 8:
        return None

    # ------------------------------------------------------------------ #
    # Head must be the HIGHEST peak                                       #
    # ------------------------------------------------------------------ #
    if not (hd["price"] > ls["price"] and hd["price"] > rs["price"]):
        return None

    # ------------------------------------------------------------------ #
    # Shoulder equality: left and right within 8%                         #
    # ------------------------------------------------------------------ #
    shoulder_diff_pct = abs(rs["price"] - ls["price"]) / ls["price"] * 100
    if shoulder_diff_pct > 8.0:
        return None

    rs_lower_than_ls = rs["price"] < ls["price"]

    # ------------------------------------------------------------------ #
    # Neckline: avg of trough between LS-HD and trough between HD-RS     #
    # ------------------------------------------------------------------ #
    ls_to_hd = closes.iloc[ls["index"]: hd["index"] + 1]
    hd_to_rs = closes.iloc[hd["index"]: rs["index"] + 1]

    left_neckline_pt  = float(ls_to_hd.min())
    right_neckline_pt = float(hd_to_rs.min())
    neckline          = (left_neckline_pt + right_neckline_pt) / 2

    # ------------------------------------------------------------------ #
    # Volume: 3-bar window centred on each peak                          #
    # ------------------------------------------------------------------ #
    def _peak_vol(idx: int) -> float:
        s = max(0, idx - 1)
        e = min(last, idx + 1)
        return float(volumes.iloc[s: e + 1].mean())

    v_ls = _peak_vol(ls["index"])
    v_hd = _peak_vol(hd["index"])
    v_rs = _peak_vol(rs["index"])

    volume_declining = v_ls >= v_hd >= v_rs

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    support       = round(neckline, 2)
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
    score = 55

    score += 15 if volume_declining else 5

    if shoulder_diff_pct <= 4.0:
        score += 10
    elif shoulder_diff_pct <= 8.0:
        score += 5

    score += 10 if rs_lower_than_ls else 5

    if stage in (3, 4):
        score += 10

    if state == "confirmed":
        score += 10

    score = min(score, 100)

    sl_price = round(rs["price"] * 1.02, 2)

    return {
        "pattern_key":       "head_shoulders",
        "state":             state,
        "confidence_score":  score,
        "support":           support,
        "sl_price":          sl_price,
        "shoulder_diff_pct": round(shoulder_diff_pct, 2),
        "head_price":        round(hd["price"], 2),
        "rs_lower_than_ls":  rs_lower_than_ls,
        "stage":             stage,
    }
