from typing import Optional

from detection.patterns._expiry import bars_since_cross, consecutive_forming_bars

import pandas as pd


PRIMARY_HOLD_DAYS = 90


def detect(df: pd.DataFrame, stage: int, pivots: dict) -> Optional[dict]:
    if stage != 2:
        return None

    peaks   = pivots.get("peaks", [])
    troughs = pivots.get("troughs", [])

    if len(peaks) < 3 or len(troughs) < 3:
        return None

    # ------------------------------------------------------------------ #
    # Build contraction pairs: match each trough to the peak that         #
    # immediately precedes it in time.                                     #
    # ------------------------------------------------------------------ #
    pairs = []
    for trough in troughs:
        preceding = [p for p in peaks if p["index"] < trough["index"]]
        if not preceding:
            continue
        peak = max(preceding, key=lambda p: p["index"])
        depth_pct = (peak["price"] - trough["price"]) / peak["price"] * 100
        pairs.append({"peak": peak, "trough": trough, "depth_pct": depth_pct})

    # Sort chronologically by trough index
    pairs.sort(key=lambda x: x["trough"]["index"])

    if len(pairs) < 3:
        return None

    # ------------------------------------------------------------------ #
    # Validate contraction sequence                                        #
    # ------------------------------------------------------------------ #
    first_depth = pairs[0]["depth_pct"]
    final_depth = pairs[-1]["depth_pct"]

    # First contraction: 10-25%
    if not (10.0 <= first_depth <= 25.0):
        return None

    # Final contraction: 3-10%
    if not (3.0 <= final_depth <= 10.0):
        return None

    # Final must be <= 20% of first
    if final_depth > first_depth * 0.20:
        return None

    # Each contraction must be strictly smaller than the previous
    for i in range(1, len(pairs)):
        if pairs[i]["depth_pct"] >= pairs[i - 1]["depth_pct"]:
            return None

    # ------------------------------------------------------------------ #
    # No lower lows: final trough <= 8% below first trough               #
    # ------------------------------------------------------------------ #
    first_trough_price = pairs[0]["trough"]["price"]
    final_trough_price = pairs[-1]["trough"]["price"]
    if final_trough_price < first_trough_price * 0.92:
        return None

    # ------------------------------------------------------------------ #
    # Volume contraction across pairs                                      #
    # ------------------------------------------------------------------ #
    def _avg_volume(start_idx: int, end_idx: int) -> float:
        slice_ = df.iloc[start_idx : end_idx + 1]["volume"]
        return float(slice_.mean()) if len(slice_) > 0 else 0.0

    avg_vols = [
        _avg_volume(p["peak"]["index"], p["trough"]["index"])
        for p in pairs
    ]

    # Volume must decrease across contractions
    for i in range(1, len(avg_vols)):
        if avg_vols[i] >= avg_vols[i - 1]:
            return None

    first_vol = avg_vols[0]
    final_vol = avg_vols[-1]

    if first_vol == 0:
        return None

    # Final contraction volume <= 50% of first
    vol_ratio = final_vol / first_vol
    if vol_ratio > 0.50:
        return None

    # ------------------------------------------------------------------ #
    # 52-week high proximity                                               #
    # ------------------------------------------------------------------ #
    lookback_252 = df.iloc[max(0, len(df) - 252):]
    high_52w = float(lookback_252["high"].max())
    current_close = float(df["close"].iloc[-1])

    if high_52w == 0:
        return None

    proximity_pct = current_close / high_52w * 100
    if not (75.0 <= proximity_pct <= 95.0):
        return None

    # ------------------------------------------------------------------ #
    # Resistance and state                                                 #
    # ------------------------------------------------------------------ #
    resistance = float(pairs[-1]["peak"]["price"])

    # State: "forming" or "confirmed"
    ma20_vol = float(df["volume"].iloc[-20:].mean()) if len(df) >= 20 else float(df["volume"].mean())
    today_vol = float(df["volume"].iloc[-1])

    below_resistance_pct = (resistance - current_close) / resistance * 100

    if current_close >= resistance * 1.005 and today_vol >= ma20_vol * 1.5:
        state = "confirmed"
    elif 0.0 <= below_resistance_pct <= 5.0:
        state = "forming"
    else:
        return None

    # ------------------------------------------------------------------ #
    # Expiry checks                                                        #
    # ------------------------------------------------------------------ #
    if state == "confirmed":
        since = bars_since_cross(df["close"].reset_index(drop=True), len(df) - 1, resistance, direction="bull")
        if since > PRIMARY_HOLD_DAYS:
            return None
    elif state == "forming":
        if consecutive_forming_bars(df["close"].reset_index(drop=True), len(df) - 1, resistance, direction="bull") > 30:
            return None

    # ------------------------------------------------------------------ #
    # Confidence score                                                     #
    # ------------------------------------------------------------------ #
    score = 50

    # Final depth
    if 3.0 <= final_depth <= 5.0:
        score += 20
    elif 5.0 < final_depth <= 8.0:
        score += 15
    elif 8.0 < final_depth <= 12.0:
        score += 10
    elif 12.0 < final_depth <= 20.0:
        score += 5

    # Contraction count
    if len(pairs) >= 4:
        score += 10
    else:
        score += 5

    # Volume ratio (final / first)
    if vol_ratio < 0.40:
        score += 15
    elif vol_ratio <= 0.60:
        score += 10
    elif vol_ratio <= 0.80:
        score += 5

    # 52-week high proximity
    if proximity_pct >= 95.0:
        score += 10
    elif proximity_pct >= 85.0:
        score += 7
    elif proximity_pct >= 75.0:
        score += 3

    # Perfect Stage 2 stack
    if stage == 2:
        score += 5

    # Volume confirmed breakout
    if state == "confirmed":
        score += 10

    score = min(score, 100)

    # ------------------------------------------------------------------ #
    # Stop loss                                                            #
    # ------------------------------------------------------------------ #
    sl_price = round(final_trough_price * 0.98, 2)

    return {
        "pattern_key":       "vcp",
        "state":             state,
        "confidence_score":  score,
        "resistance":        round(resistance, 2),
        "sl_price":          sl_price,
        "contraction_count": len(pairs),
        "final_depth_pct":   round(final_depth, 2),
        "stage":             stage,
    }
