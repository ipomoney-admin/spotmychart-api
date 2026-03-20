import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def get_pivots(df: pd.DataFrame) -> dict:
    """
    Identify zigzag swing highs (peaks) and swing lows (troughs).

    Noise threshold  = ATR(14) * 0.75
    Min bar gap      = 5 bars between consecutive pivots of the same type

    Returns:
        {
            "peaks":   [{"index": int, "date": date, "price": float}, ...],
            "troughs": [{"index": int, "date": date, "price": float}, ...],
        }
    """
    MIN_BARS = 5

    df = df.reset_index(drop=True)
    atr_series = _atr(df)

    peaks:   list[dict] = []
    troughs: list[dict] = []

    # State machine
    # direction: +1 = currently trending up (looking for peak)
    #            -1 = currently trending down (looking for trough)
    direction   = 0
    extreme_idx = 0
    extreme_val = df["close"].iloc[0]

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        noise = (atr_series.iloc[i] * 0.75) if not pd.isna(atr_series.iloc[i]) else 0.0

        if direction == 0:
            # Bootstrap direction from first meaningful move
            if price > extreme_val + noise:
                direction   = 1
                extreme_idx = i
                extreme_val = price
            elif price < extreme_val - noise:
                direction   = -1
                extreme_idx = i
                extreme_val = price
            continue

        if direction == 1:
            if price > extreme_val:
                extreme_idx = i
                extreme_val = price
            elif price < extreme_val - noise and i - extreme_idx >= MIN_BARS:
                # Confirmed peak
                peaks.append({
                    "index": extreme_idx,
                    "date":  df["date"].iloc[extreme_idx],
                    "price": float(extreme_val),
                })
                direction   = -1
                extreme_idx = i
                extreme_val = price

        else:  # direction == -1
            if price < extreme_val:
                extreme_idx = i
                extreme_val = price
            elif price > extreme_val + noise and i - extreme_idx >= MIN_BARS:
                # Confirmed trough
                troughs.append({
                    "index": extreme_idx,
                    "date":  df["date"].iloc[extreme_idx],
                    "price": float(extreme_val),
                })
                direction   = 1
                extreme_idx = i
                extreme_val = price

    return {"peaks": peaks, "troughs": troughs}
