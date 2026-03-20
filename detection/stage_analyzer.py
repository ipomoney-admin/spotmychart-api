import pandas as pd


def get_stage(df: pd.DataFrame) -> int:
    """
    Determine Weinstein stage (1-4) for the most recent bar.
    Returns 0 if there is insufficient data.

    Stage 2: Advancing — close > MA50 > MA150 > MA200, MA200 rising
    Stage 4: Declining — close < MA50 < MA150 < MA200, MA200 falling
    Stage 3: Topping — close > MA200 but full Stage 2 stack broken
    Stage 1: Basing  — close < MA200 but MA200 flattening or turning up
    """
    if len(df) < 200 + 21:
        return 0

    close = df["close"].reset_index(drop=True)

    ma50  = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    last = len(close) - 1

    c      = close.iloc[last]
    m50    = ma50.iloc[last]
    m150   = ma150.iloc[last]
    m200   = ma200.iloc[last]
    m200_21 = ma200.iloc[last - 21]

    if any(pd.isna(v) for v in [c, m50, m150, m200, m200_21]):
        return 0

    ma200_rising   = m200 > m200_21
    ma200_flat_up  = m200 >= m200_21  # flattening or turning up

    # Stage 2: full bullish stack + MA200 rising
    if c > m50 > m150 > m200 and m150 > m200 and ma200_rising:
        return 2

    # Stage 4: full bearish stack + MA200 declining
    if c < m50 < m150 < m200 and not ma200_rising:
        return 4

    # Stage 3: still above MA200 but stack broken
    if c > m200:
        return 3

    # Stage 1: below MA200 but MA200 flattening or turning up
    if c < m200 and ma200_flat_up:
        return 1

    return 4  # below MA200 and MA200 still declining
