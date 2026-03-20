from datetime import date

from data.fetcher import fetch_ohlcv
from detection.stage_analyzer import get_stage
from detection.zigzag_pivots import get_pivots
from detection.patterns import (
    ascending_triangle, bear_flag, bear_pennant, bull_flag, bull_pennant,
    cup_handle, desc_triangle, double_bottom, double_top, falling_wedge,
    head_shoulders, inv_cup_handle, inv_head_shoulders, rising_wedge,
    rounding_bottom, rounding_top, sym_triangle_bear, sym_triangle_bull,
    triple_bottom, triple_top, vcp,
)

TICKERS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "TITAN", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "SBILIFE",
    "NESTLEIND", "HINDUNILVR", "DABUR", "MARICO", "COLPAL",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "ONGC",
    "NTPC", "POWERGRID", "TATAPOWER", "ADANIPORTS", "ADANIENT",
    "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR",
    "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT",
    "INDUSINDBK", "FEDERALBNK", "BANDHANBNK", "IDFCFIRSTB", "RBLBANK",
    "IRFC", "RVNL", "NBCC", "INOXWIND", "SUZLON",
]

DETECTORS = {
    "vcp":                vcp.detect,
    "bull_flag":          bull_flag.detect,
    "bull_pennant":       bull_pennant.detect,
    "ascending_triangle": ascending_triangle.detect,
    "sym_triangle_bull":  sym_triangle_bull.detect,
    "cup_handle":         cup_handle.detect,
    "double_bottom":      double_bottom.detect,
    "triple_bottom":      triple_bottom.detect,
    "inv_head_shoulders": inv_head_shoulders.detect,
    "rounding_bottom":    rounding_bottom.detect,
    "falling_wedge":      falling_wedge.detect,
    "bear_flag":          bear_flag.detect,
    "bear_pennant":       bear_pennant.detect,
    "desc_triangle":      desc_triangle.detect,
    "sym_triangle_bear":  sym_triangle_bear.detect,
    "inv_cup_handle":     inv_cup_handle.detect,
    "double_top":         double_top.detect,
    "triple_top":         triple_top.detect,
    "head_shoulders":     head_shoulders.detect,
    "rounding_top":       rounding_top.detect,
    "rising_wedge":       rising_wedge.detect,
}

START = date(2015, 1, 1)
END   = date.today()

print(f"\n{'='*70}")
print(f"  SpotMyChart — full scan  |  {len(TICKERS)} tickers  |  {len(DETECTORS)} patterns")
print(f"{'='*70}\n")

hits = []   # [(confidence_score, ticker, pattern_key, state, result)]
errors = 0

for ticker in TICKERS:
    df = fetch_ohlcv(ticker, START, END)
    if df.empty:
        print(f"  [{ticker}] FETCH FAILED")
        errors += 1
        continue

    stage  = get_stage(df)
    pivots = get_pivots(df)

    for pattern_key, fn in DETECTORS.items():
        try:
            result = fn(df, stage, pivots)
            if result is not None:
                hits.append((
                    result["confidence_score"],
                    ticker,
                    pattern_key,
                    result.get("state", "—"),
                    result,
                ))
        except Exception as e:
            print(f"  [{ticker}] {pattern_key} error: {e}")
            errors += 1

hits.sort(key=lambda x: x[0], reverse=True)

print(f"  {'TICKER':<14} {'PATTERN':<22} {'STATE':<12} {'SCORE':>5}")
print(f"  {'-'*14} {'-'*22} {'-'*12} {'-'*5}")

for score, ticker, pattern_key, state, result in hits:
    print(f"  {ticker:<14} {pattern_key:<22} {state:<12} {score:>5}")

print(f"\n{'='*70}")
print(f"  Signals found : {len(hits)}")
print(f"  Errors        : {errors}")
print(f"  Tickers scanned: {len(TICKERS)}")
print(f"{'='*70}\n")
