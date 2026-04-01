import sys
sys.path.insert(0, "/Users/sahib/spotmychart-api")
from dotenv import load_dotenv
load_dotenv("/Users/sahib/spotmychart-api/.env")

import pandas as pd
from datetime import date, datetime, timedelta
from collections import defaultdict

from core.supabase_client import sb
from data.fetcher import fetch_ohlcv
from detection.stage_analyzer import get_stage
from detection.zigzag_pivots import get_pivots
from detection.patterns.vcp import detect

START_DATE = date(2015, 1, 1)
END_DATE   = date.today()
OUT_FILE   = "/Users/sahib/spotmychart-vcp_final.txt"


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_ema20(close_series):
    """Returns list of EMA20 values (same length as close_series)."""
    closes = list(close_series)
    n = len(closes)
    ema = [0.0] * n
    k = 2 / (20 + 1)
    # seed with SMA of first 20
    if n < 20:
        for i in range(n):
            ema[i] = closes[i]
        return ema
    ema[19] = sum(closes[:20]) / 20
    for i in range(20, n):
        ema[i] = closes[i] * k + ema[i - 1] * (1 - k)
    # fill the warm-up period with a simple forward-fill from index 19
    for i in range(19):
        ema[i] = ema[19]
    return ema


def calc_supertrend(df, period=14, multiplier=2):
    """Returns (st_values, st_direction) lists.  direction 1=bullish, -1=bearish."""
    high  = list(df["high"])
    low   = list(df["low"])
    close = list(df["close"])
    n     = len(df)
    atr   = [0.0] * n
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )
        if i < period:
            atr[i] = tr
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period

    st  = [0.0] * n
    dr  = [1]   * n
    for i in range(period, n):
        hl2   = (high[i] + low[i]) / 2
        upper = hl2 + multiplier * atr[i]
        lower = hl2 - multiplier * atr[i]
        if dr[i - 1] == 1:
            curr = max(lower, st[i - 1])
            if close[i] < curr:
                st[i] = upper
                dr[i] = -1
            else:
                st[i] = curr
                dr[i] = 1
        else:
            curr = min(upper, st[i - 1])
            if close[i] > curr:
                st[i] = lower
                dr[i] = 1
            else:
                st[i] = curr
                dr[i] = -1
    return st, dr


# ── Supabase helpers ──────────────────────────────────────────────────────────

def fetch_all_tickers():
    tickers = set()
    offset  = 0
    while True:
        resp = sb.table("smc_metrics").select("stock_ticker").range(offset, offset + 999).execute()
        if not resp.data:
            break
        for r in resp.data:
            if r.get("stock_ticker"):
                tickers.add(r["stock_ticker"])
        offset += 1000
        if len(resp.data) < 1000:
            break
    return sorted(tickers)


# ── Tier from confidence score ────────────────────────────────────────────────

def score_to_tier(score):
    if score >= 85:
        return 1
    elif score >= 75:
        return 2
    elif score >= 65:
        return 3
    elif score >= 55:
        return 4
    else:
        return 5


# ── Peak concurrent trades ────────────────────────────────────────────────────

def peak_concurrent(signals):
    events = []
    for s in signals:
        try:
            entry = datetime.strptime(str(s["date"]), "%Y-%m-%d")
            exit_ = entry + timedelta(days=s["days"])
            events.append((entry,  1))
            events.append((exit_, -1))
        except Exception:
            continue
    events.sort(key=lambda x: x[0])
    peak = cur = 0
    for _, v in events:
        cur += v
        peak = max(peak, cur)
    return peak


# ── Core backtest loop for one stock ─────────────────────────────────────────

def backtest_stock(stock, df_full, st_vals, st_dirs, ema20_vals):
    signals      = []
    active_trade = None

    for i in range(200, len(df_full)):
        current_date = df_full["date"].iloc[i]
        close        = float(df_full["close"].iloc[i])
        st_val       = st_vals[i]
        ema20_val    = ema20_vals[i]

        # ── Manage open trade ─────────────────────────────────────────────
        if active_trade is not None:
            sl = active_trade["sl"]
            be = active_trade["breakeven_triggered"]

            # Breakeven: if profit >= initial risk → move SL to entry
            if not be:
                initial_risk = active_trade["entry_price"] - active_trade["initial_sl"]
                if initial_risk > 0 and close >= active_trade["entry_price"] + initial_risk:
                    active_trade["sl"] = active_trade["entry_price"]
                    active_trade["breakeven_triggered"] = True
                    sl = active_trade["sl"]

            # After breakeven: trail SL with supertrend
            if active_trade["breakeven_triggered"]:
                if st_dirs[i] == 1 and st_val > 0:
                    active_trade["sl"] = max(active_trade["sl"], st_val)
                    sl = active_trade["sl"]

            # SL conditions: hard 8%, EMA20 cross below, supertrend flip
            sl_8pct      = active_trade["entry_price"] * 0.92
            hard_sl      = min(sl, sl_8pct)

            ema_exit     = close < ema20_val and ema20_val > 0
            st_exit      = st_dirs[i] == -1
            price_sl_hit = close <= hard_sl

            exit_hit = price_sl_hit or ema_exit or st_exit

            if exit_hit:
                days_held = (current_date - active_trade["entry_date"]).days
                if price_sl_hit:
                    reason = "sl_hit"
                elif ema_exit:
                    reason = "ema_exit"
                else:
                    reason = "st_exit"
                ret = round((close - active_trade["entry_price"]) / active_trade["entry_price"] * 100, 2)
                signals.append({
                    "stock":  stock,
                    "date":   active_trade["entry_date"],
                    "entry":  active_trade["entry_price"],
                    "exit":   round(close, 2),
                    "ret":    ret,
                    "win":    ret >= 0,
                    "days":   days_held,
                    "reason": reason,
                    "tier":   active_trade["tier"],
                })
                active_trade = None
            continue

        # ── Look for new entry ────────────────────────────────────────────
        # Only enter when supertrend is bullish
        if st_dirs[i] != 1:
            continue

        df_slice = df_full.iloc[: i + 1].reset_index(drop=True)
        try:
            stage  = get_stage(df_slice)
            pivots = get_pivots(df_slice)
            result = detect(df_slice, stage, pivots)
        except Exception:
            continue

        if result is None or result.get("state") != "confirmed":
            continue

        tier       = score_to_tier(result["confidence_score"])
        entry_sl   = result["sl_price"]
        hard_sl    = close * 0.92
        initial_sl = min(entry_sl, hard_sl)

        active_trade = {
            "entry_date":          current_date,
            "entry_price":         close,
            "sl":                  initial_sl,
            "initial_sl":          initial_sl,
            "breakeven_triggered": False,
            "tier":                tier,
        }

    # Close any open trade at end of data
    if active_trade is not None:
        last_i    = len(df_full) - 1
        last_date = df_full["date"].iloc[last_i]
        last_close = float(df_full["close"].iloc[last_i])
        days_held  = (last_date - active_trade["entry_date"]).days
        ret        = round((last_close - active_trade["entry_price"]) / active_trade["entry_price"] * 100, 2)
        signals.append({
            "stock":  stock,
            "date":   active_trade["entry_date"],
            "entry":  active_trade["entry_price"],
            "exit":   round(last_close, 2),
            "ret":    ret,
            "win":    ret >= 0,
            "days":   days_held,
            "reason": "open",
            "tier":   active_trade["tier"],
        })

    return signals


# ── Summary helpers ───────────────────────────────────────────────────────────

def summarize(signals):
    if not signals:
        return
    print(f"\n{'='*70}")
    print(f"VCP FINAL — {START_DATE} to {END_DATE} — ALL TIERS")
    print(f"{'='*70}")
    print(f"Total signals: {len(signals)}")

    for tier in [1, 2, 3, 4, 5]:
        ts = [s for s in signals if s["tier"] == tier]
        if not ts:
            continue
        tw  = [s for s in ts if s["win"]]
        tl  = [s for s in ts if not s["win"]]
        wr  = round(len(tw) / len(ts) * 100, 1)
        aw  = round(sum(s["ret"] for s in tw) / len(tw), 1) if tw else 0
        al  = round(sum(s["ret"] for s in tl) / len(tl), 1) if tl else 0
        rr  = round(abs(aw / al), 2) if al else 0
        exp = round((len(tw) / len(ts)) * aw + (len(tl) / len(ts)) * al, 1)
        tyr = round(len(ts) / ((END_DATE - START_DATE).days / 365), 1)
        pk  = peak_concurrent(ts)
        stocks_count = len(set(s["stock"] for s in ts))
        print(
            f"\n  T{tier} | {len(ts)} signals | {stocks_count} stocks | "
            f"WR:{wr}% | AvgW:+{aw}% | AvgL:{al}% | RR:{rr} | "
            f"Exp:{exp:+.1f}% | T/Yr:{tyr} | Peak:{pk}"
        )


def save_results(signals):
    wins   = [s for s in signals if s["win"]]
    losses = [s for s in signals if not s["win"]]
    wr  = round(len(wins) / len(signals) * 100, 1) if signals else 0
    rr  = 0
    exp = 0
    if wins and losses:
        aw  = sum(s["ret"] for s in wins)  / len(wins)
        al  = sum(s["ret"] for s in losses) / len(losses)
        rr  = round(abs(aw / al), 2) if al else 0
        exp = round((len(wins) / len(signals)) * aw + (len(losses) / len(signals)) * al, 1)

    with open(OUT_FILE, "w") as f:
        f.write(f"VCP FINAL — {len(set(s['stock'] for s in signals))} stocks — {START_DATE} to {END_DATE}\n")
        f.write(f"Total: {len(signals)} | WR: {wr}% | R:R: {rr} | Exp: {exp:+.1f}%\n\n")
        f.write(f"{'STOCK':<12} {'DATE':<12} {'ENTRY':>8} {'EXIT':>8} {'RET%':>7}  {'W':>1}  {'DAYS':>5} {'REASON':<10} {'TIER'}\n")
        f.write("-" * 75 + "\n")
        for s in sorted(signals, key=lambda x: str(x["date"])):
            w = "W" if s["win"] else "L"
            f.write(
                f"{s['stock']:<12} {str(s['date']):<12} {s['entry']:>8.2f} {s['exit']:>8.2f} "
                f"{s['ret']:>+7.1f}%  {w:>1}  {s['days']:>5} {s['reason']:<10} T{s['tier']}\n"
            )
    print(f"\nSaved to: {OUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

print("Fetching tickers from Supabase...")
tickers = fetch_all_tickers()
print(f"Total tickers: {len(tickers)}")

all_signals = []
total = len(tickers)

for idx, stock in enumerate(tickers):
    if (idx + 1) % 10 == 0:
        print(f"  [{idx+1}/{total}] {stock} — signals so far: {len(all_signals)}")

    df_full = fetch_ohlcv(stock, start_date=START_DATE, end_date=END_DATE)
    if df_full is None or df_full.empty or len(df_full) < 250:
        continue

    try:
        st_vals, st_dirs = calc_supertrend(df_full)
        ema20_vals       = calc_ema20(df_full["close"])
    except Exception as e:
        print(f"  Indicator error for {stock}: {e}")
        continue

    try:
        sigs = backtest_stock(stock, df_full, st_vals, st_dirs, ema20_vals)
        all_signals.extend(sigs)
    except Exception as e:
        print(f"  Backtest error for {stock}: {e}")
        continue

print(f"\nDone. Total signals: {len(all_signals)}")

save_results(all_signals)
summarize(all_signals)
