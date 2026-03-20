from datetime import date

from data.fetcher import fetch_ohlcv
from detection.stage_analyzer import get_stage
from detection.zigzag_pivots import get_pivots
from detection.patterns import vcp, bull_flag, bull_pennant

TICKERS = [
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "TITAN", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "SBILIFE",
    "NESTLEIND", "HINDUNILVR", "DABUR", "MARICO", "COLPAL",
]
START = date(2015, 1, 1)
END   = date.today()

DETECTORS = {
    "vcp":          vcp.detect,
    "bull_flag":    bull_flag.detect,
    "bull_pennant": bull_pennant.detect,
}

STAGE_LABELS = {
    0: "Insufficient data",
    1: "Stage 1 — Basing",
    2: "Stage 2 — Advancing",
    3: "Stage 3 — Topping",
    4: "Stage 4 — Declining",
}

print(f"\n{'='*60}")
print(f"  SpotMyChart — Stage 2 scan ({len(TICKERS)} tickers)")
print(f"{'='*60}")

stage2_results = []  # [(ticker, stage, {pattern_key: result})]

for ticker in TICKERS:
    df = fetch_ohlcv(ticker, START, END)
    if df.empty:
        print(f"  {ticker:<14}  FETCH FAILED")
        continue

    stage = get_stage(df)
    if stage != 2:
        print(f"  {ticker:<14}  Stage {stage}")
        continue

    pivots   = get_pivots(df)
    detected = {k: fn(df, stage, pivots) for k, fn in DETECTORS.items()}
    hits     = [k for k, v in detected.items() if v is not None]

    summary = f"patterns={hits}" if hits else "no patterns"
    print(f"  {ticker:<14}  Stage 2 ***  {summary}")
    stage2_results.append((ticker, detected))

print(f"\n{'='*60}")
print(f"  Stage 2 tickers: {len(stage2_results)}")
print(f"{'='*60}")

for ticker, detected in stage2_results:
    print(f"\n  {ticker}")
    print(f"  {'-'*40}")
    any_hit = False
    for pattern_key, result in detected.items():
        if result is None:
            print(f"  {pattern_key:<14}: None")
        else:
            any_hit = True
            print(f"  {pattern_key:<14}: DETECTED")
            for k, v in result.items():
                print(f"    {k:<20}: {v}")

print(f"\n{'='*60}\n")
