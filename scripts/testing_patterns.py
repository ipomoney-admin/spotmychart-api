import sys
sys.path.insert(0, '/Users/sahib/spotmychart-api')
from dotenv import load_dotenv
load_dotenv('/Users/sahib/spotmychart-api/.env')

import pandas as pd
from datetime import date, timedelta
from collections import defaultdict
from data.fetcher import fetch_ohlcv
from detection.stage_analyzer import get_stage
from detection.zigzag_pivots import get_pivots

STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "POWERGRID", "NTPC", "TECHM", "BAJAJFINSV",
    "ONGC", "JSWSTEEL", "TATAMOTORS", "ADANIENT", "COALINDIA",
    "TATASTEEL", "GRASIM", "DRREDDY", "DIVISLAB", "CIPLA",
    "APOLLOHOSP", "BPCL", "HEROMOTOCO", "EICHERMOT", "TATACONSUM",
    "BRITANNIA", "HINDALCO", "INDUSINDBK", "UPL", "SBILIFE",
    "HDFCLIFE", "BAJAJ-AUTO", "ADANIPORTS", "M&M", "LTIM"
]

SETUPS = {
    "A": dict(vol_mult=1.2, final_lo=2, final_hi=12, last3_ratio=1.00),
    "B": dict(vol_mult=1.5, final_lo=2, final_hi=8,  last3_ratio=0.85),
    "C": dict(vol_mult=1.8, final_lo=2, final_hi=6,  last3_ratio=0.75),
}

MAX_LOSS_CAP = 0.08  # 8%

def calc_supertrend(df, period=14, multiplier=2):
    high  = df['high'].reset_index(drop=True)
    low   = df['low'].reset_index(drop=True)
    close = df['close'].reset_index(drop=True)
    n     = len(df)
    atr   = [0.0] * n
    for i in range(1, n):
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        atr[i] = tr if i < period else (atr[i-1]*(period-1)+tr)/period
    st = [0.0]*n
    dr = [1]*n
    for i in range(period, n):
        hl2   = (high[i]+low[i])/2
        upper = hl2 + multiplier*atr[i]
        lower = hl2 - multiplier*atr[i]
        if dr[i-1] == 1:
            curr = max(lower, st[i-1])
            if close[i] < curr:
                st[i]=upper; dr[i]=-1
            else:
                st[i]=curr;  dr[i]=1
        else:
            curr = min(upper, st[i-1])
            if close[i] > curr:
                st[i]=lower; dr[i]=1
            else:
                st[i]=curr;  dr[i]=-1
    return st, dr

def calc_ema20(df):
    close = df['close'].reset_index(drop=True)
    ema   = [0.0] * len(close)
    k     = 2 / (20+1)
    ema[0] = float(close[0])
    for i in range(1, len(close)):
        ema[i] = float(close[i])*k + ema[i-1]*(1-k)
    return ema

def detect_vcp(df, stage, pivots, cfg):
    if stage not in (1, 2):
        return None
    peaks   = pivots.get('peaks', [])
    troughs = pivots.get('troughs', [])
    cutoff  = len(df) - 252
    peaks_r   = [p for p in peaks   if p['index'] >= cutoff]
    troughs_r = [p for p in troughs if p['index'] >= cutoff]
    pairs = []
    for trough in troughs_r:
        preceding = [p for p in peaks_r if p['index'] < trough['index']]
        if not preceding:
            continue
        peak  = max(preceding, key=lambda p: p['index'])
        depth = (peak['price']-trough['price'])/peak['price']*100
        pairs.append({'peak': peak, 'trough': trough, 'depth': depth})
    pairs.sort(key=lambda x: x['trough']['index'])
    if len(pairs) < 3:
        return None
    first_depth = pairs[0]['depth']
    final_depth = pairs[-1]['depth']
    if not (5.0 <= first_depth <= 35.0):
        return None
    if not (cfg['final_lo'] <= final_depth <= cfg['final_hi']):
        return None
    first3_avg = sum(p['depth'] for p in pairs[:3])/3
    last3_avg  = sum(p['depth'] for p in pairs[-3:])/3
    if last3_avg >= first3_avg * cfg['last3_ratio']:
        return None
    if pairs[-1]['trough']['price'] < pairs[0]['trough']['price']*0.90:
        return None
    high_52w = float(df.iloc[max(0,len(df)-252):]['high'].max())
    close    = float(df['close'].iloc[-1])
    if high_52w == 0:
        return None
    prox = close/high_52w*100
    if not (60.0 <= prox <= 100.0):
        return None
    resistance = float(pairs[-1]['peak']['price'])
    ma20_vol   = float(df['volume'].iloc[-20:].mean())
    today_vol  = float(df['volume'].iloc[-1])
    below_pct  = (resistance-close)/resistance*100
    if close >= resistance*1.005 and today_vol >= ma20_vol*cfg['vol_mult']:
        state = 'confirmed'
    elif 0.0 <= below_pct <= 7.0:
        state = 'forming'
    else:
        return None
    if state != 'confirmed':
        return None
    return {
        'resistance': resistance,
        'sl_price':   float(pairs[-1]['trough']['price'])*0.98,
        'pairs':      len(pairs),
        'f3avg':      round(first3_avg,1),
        'l3avg':      round(last3_avg,1),
        'final_d':    round(final_depth,1),
        'prox':       round(prox,1),
        'stage':      stage,
    }

def run_setup(setup_name):
    cfg     = SETUPS[setup_name]
    signals = []

    for idx, stock in enumerate(STOCKS):
        if (idx+1) % 10 == 0 or idx == 0: print(f"  [{setup_name}] {idx+1}/{len(STOCKS)} — {stock}")
        df_full = fetch_ohlcv(stock, start_date=date(2015,1,1), end_date=date.today())
        if df_full is None or df_full.empty:
            continue

        st_vals, st_dir = calc_supertrend(df_full)
        ema20_vals      = calc_ema20(df_full)
        active_trade    = None

        for i in range(200, len(df_full)):
            current_date = df_full['date'].iloc[i]
            close        = float(df_full['close'].iloc[i])
            st_val       = st_vals[i]
            ema20_val    = ema20_vals[i]

            if active_trade:
                ep        = active_trade['entry_price']
                init_sl   = active_trade['init_sl']
                current_sl= active_trade['current_sl']
                risk      = ep - init_sl  # initial risk in Rs
                be_triggered = active_trade['be_triggered']

                # Check breakeven trigger
                if not be_triggered and close >= ep + risk:
                    active_trade['current_sl'] = ep  # SL to cost
                    active_trade['be_triggered'] = True
                    current_sl = ep

                # After breakeven — trail with supertrend
                if be_triggered:
                    new_sl = st_val  # trail with supertrend
                    if new_sl > current_sl:
                        active_trade['current_sl'] = new_sl
                        current_sl = new_sl

                days_held = (current_date - active_trade['entry_date']).days

                # Exit conditions — whichever first
                cap_sl    = ep * (1 - MAX_LOSS_CAP)
                exit_sl   = max(current_sl, cap_sl) if not be_triggered else current_sl
                st_exit   = close <= st_val
                ema_exit  = close <= ema20_val
                cap_exit  = close <= cap_sl

                should_exit = st_exit or ema_exit or cap_exit

                if should_exit:
                    ret    = round((close-ep)/ep*100, 2)
                    reason = 'st_exit' if st_exit else ('ema_exit' if ema_exit else 'cap_exit')
                    signals.append({
                        'stock':       stock,
                        'date':        active_trade['entry_date'],
                        'exit_date':   current_date,
                        'year':        active_trade['entry_date'].year,
                        'entry':       ep,
                        'exit':        round(close,2),
                        'ret':         ret,
                        'win':         ret >= 0,
                        'reason':      reason,
                        'days':        days_held,
                        'be_hit':      active_trade['be_triggered'],
                        **active_trade['meta'],
                    })
                    active_trade = None
                continue

            # New signal — no supertrend entry condition
            df_slice = df_full[df_full['date'] <= current_date].reset_index(drop=True)
            try:
                stage  = get_stage(df_slice)
                pivots = get_pivots(df_slice)
                result = detect_vcp(df_slice, stage, pivots, cfg)
            except:
                continue

            if result is None:
                continue

            active_trade = {
                'entry_date':   current_date,
                'entry_price':  close,
                'init_sl':      result['sl_price'],
                'current_sl':   result['sl_price'],
                'be_triggered': False,
                'meta': {
                    'pairs':   result['pairs'],
                    'f3avg':   result['f3avg'],
                    'l3avg':   result['l3avg'],
                    'final_d': result['final_d'],
                    'prox':    result['prox'],
                    'stage':   result['stage'],
                }
            }

    return signals

def print_results(setup_name, signals):
    cfg = SETUPS[setup_name]
    print(f"\n{'='*70}")
    print(f"VCP SETUP {setup_name} | vol={cfg['vol_mult']}x | final={cfg['final_lo']}-{cfg['final_hi']}% | SL=8%cap+ST+EMA20 | Trail=BE+ST")
    print(f"{'='*70}")

    if not signals:
        print("No signals found.")
        return

    wins   = [s for s in signals if s['win']]
    losses = [s for s in signals if not s['win']]
    wr     = round(len(wins)/len(signals)*100,1)
    avgw   = round(sum(s['ret'] for s in wins)/len(wins),1) if wins else 0
    avgl   = round(sum(s['ret'] for s in losses)/len(losses),1) if losses else 0
    rr     = round(abs(avgw/avgl),2) if avgl else 0
    exp    = round((len(wins)/len(signals))*avgw+(len(losses)/len(signals))*avgl,1)
    avgh   = round(sum(s['days'] for s in signals)/len(signals),1)
    tpy    = round(len(signals)/11,1)
    be_pct = round(sum(1 for s in signals if s['be_hit'])/len(signals)*100,1)

    print(f"Total     : {len(signals)} | Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate  : {wr}%")
    print(f"Avg Win   : +{avgw}% | Avg Loss: {avgl}%")
    print(f"R:R       : {rr} | Expectancy: {exp:+.1f}%")
    print(f"Avg Hold  : {avgh} days")
    print(f"T/Year    : {tpy}")
    print(f"BE Hit    : {be_pct}% of trades reached breakeven")

    # Per stock summary
    by_stock = defaultdict(list)
    for s in signals:
        by_stock[s['stock']].append(s)
    print(f"\n{'STOCK':<14} {'SIG':>4} {'WR%':>6} {'AVG':>8} {'EXP':>7}")
    print('-'*38)
    for stk in STOCKS:
        ss = by_stock.get(stk,[])
        if not ss: continue
        sw   = sum(1 for s in ss if s['win'])
        swr  = round(sw/len(ss)*100)
        savg = round(sum(s['ret'] for s in ss)/len(ss),1)
        sww  = [s for s in ss if s['win']]
        swl  = [s for s in ss if not s['win']]
        sexp = round((sw/len(ss))*(sum(s['ret'] for s in sww)/len(sww) if sww else 0)+
                     ((len(ss)-sw)/len(ss))*(sum(s['ret'] for s in swl)/len(swl) if swl else 0),1)
        print(f"{stk:<14} {len(ss):>4} {swr:>5}% {savg:>+8.1f}% {sexp:>+7.1f}%")

    # Year breakdown
    by_year = defaultdict(list)
    for s in signals:
        by_year[s['year']].append(s)
    print(f"\n{'YEAR':<6} {'SIG':>5} {'WR%':>6} {'AVG':>8} {'EXP':>7} {'HOLD':>6} {'BE%':>5}")
    print('-'*45)
    for yr in sorted(by_year.keys()):
        ys   = by_year[yr]
        yw   = [s for s in ys if s['win']]
        yl   = [s for s in ys if not s['win']]
        ywr  = round(len(yw)/len(ys)*100)
        yavg = round(sum(s['ret'] for s in ys)/len(ys),1)
        yww  = sum(s['ret'] for s in yw)
        ywl  = sum(s['ret'] for s in yl)
        yexp = round((len(yw)/len(ys))*(yww/len(yw) if yw else 0)+
                     (len(yl)/len(ys))*(ywl/len(yl) if yl else 0),1)
        yavh = round(sum(s['days'] for s in ys)/len(ys),1)
        ybe  = round(sum(1 for s in ys if s['be_hit'])/len(ys)*100)
        print(f"{yr:<6} {len(ys):>5} {ywr:>5}% {yavg:>+8.1f}% {yexp:>+7.1f}% {yavh:>6.1f} {ybe:>4}%")

    # Exit reason breakdown
    reasons = defaultdict(int)
    for s in signals:
        reasons[s['reason']] += 1
    print(f"\nExit reasons:")
    for r,c in sorted(reasons.items(), key=lambda x:-x[1]):
        print(f"  {r}: {c} ({round(c/len(signals)*100,1)}%)")

print("Testing VCP — Setups A,B,C — Nifty 50 — New Rules\n")
for setup in ["A","B","C"]:
    print(f"\nRunning Setup {setup}...")
    signals = run_setup(setup)
    print_results(setup, signals)
print("\nDone.")