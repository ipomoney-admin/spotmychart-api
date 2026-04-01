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
from core.supabase_client import sb

HOLD = 90

SETUPS = {
    "A": dict(vol_mult=1.2, final_lo=2, final_hi=12, last3_ratio=1.00),
    "B": dict(vol_mult=1.5, final_lo=2, final_hi=8,  last3_ratio=0.85),
    "C": dict(vol_mult=1.8, final_lo=2, final_hi=6,  last3_ratio=0.75),
    "D": dict(vol_mult=2.0, final_lo=2, final_hi=5,  last3_ratio=0.65),
}

def fetch_all_tickers():
    all_tickers = set()
    offset = 0
    while True:
        resp = sb.table("smc_metrics").select("stock_ticker").range(offset, offset+999).execute()
        if not resp.data:
            break
        for r in resp.data:
            if r.get("stock_ticker"):
                all_tickers.add(r["stock_ticker"])
        offset += 1000
        if len(resp.data) < 1000:
            break
    tickers = sorted(list(all_tickers))
    print(f"Total tickers: {len(tickers)}")
    return tickers

def calc_supertrend(df, period=14, multiplier=2):
    high  = df['high'].reset_index(drop=True)
    low   = df['low'].reset_index(drop=True)
    close = df['close'].reset_index(drop=True)
    n     = len(df)
    atr   = [0.0] * n
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i]  - close[i-1])
        )
        atr[i] = tr if i < period else (atr[i-1] * (period-1) + tr) / period
    st = [0.0] * n
    dr = [1]   * n
    for i in range(period, n):
        hl2   = (high[i] + low[i]) / 2
        upper = hl2 + multiplier * atr[i]
        lower = hl2 - multiplier * atr[i]
        if dr[i-1] == 1:
            curr = max(lower, st[i-1])
            if close[i] < curr:
                st[i] = upper; dr[i] = -1
            else:
                st[i] = curr;  dr[i] = 1
        else:
            curr = min(upper, st[i-1])
            if close[i] > curr:
                st[i] = lower; dr[i] = 1
            else:
                st[i] = curr;  dr[i] = -1
    return st, dr

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
        depth = (peak['price'] - trough['price']) / peak['price'] * 100
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
    first3_avg = sum(p['depth'] for p in pairs[:3]) / 3
    last3_avg  = sum(p['depth'] for p in pairs[-3:]) / 3
    if last3_avg >= first3_avg * cfg['last3_ratio']:
        return None
    if pairs[-1]['trough']['price'] < pairs[0]['trough']['price'] * 0.90:
        return None
    high_52w = float(df.iloc[max(0, len(df)-252):]['high'].max())
    close    = float(df['close'].iloc[-1])
    if high_52w == 0:
        return None
    prox = close / high_52w * 100
    if not (60.0 <= prox <= 100.0):
        return None
    resistance = float(pairs[-1]['peak']['price'])
    ma20_vol   = float(df['volume'].iloc[-20:].mean())
    today_vol  = float(df['volume'].iloc[-1])
    below_pct  = (resistance - close) / resistance * 100
    if close >= resistance * 1.005 and today_vol >= ma20_vol * cfg['vol_mult']:
        state = 'confirmed'
    elif 0.0 <= below_pct <= 7.0:
        state = 'forming'
    else:
        return None
    if state != 'confirmed':
        return None
    return {
        'resistance': resistance,
        'sl_price':   float(pairs[-1]['trough']['price']) * 0.98,
        'pairs':      len(pairs),
        'f3avg':      round(first3_avg, 1),
        'l3avg':      round(last3_avg, 1),
        'final_d':    round(final_depth, 1),
        'prox':       round(prox, 1),
        'stage':      stage,
    }

def run_setup(setup_name, tickers):
    cfg     = SETUPS[setup_name]
    signals = []
    total   = len(tickers)

    for idx, stock in enumerate(tickers):
        if (idx+1) % 10 == 0:
            print(f"  [{setup_name}] {idx+1}/{total} — {stock}")
        df_full = fetch_ohlcv(stock, start_date=date(2015,1,1), end_date=date.today())
        if df_full is None or df_full.empty:
            continue
        try:
            st_vals, st_dir = calc_supertrend(df_full)
        except:
            continue
        active_trade = None
        for i in range(200, len(df_full)):
            current_date = df_full['date'].iloc[i]
            close        = float(df_full['close'].iloc[i])
            if active_trade:
                days_held = (current_date - active_trade['entry_date']).days
                sl_hit    = close <= active_trade['sl_price']
                time_up   = days_held >= HOLD
                if sl_hit or time_up:
                    ret = round((close - active_trade['entry_price']) / active_trade['entry_price'] * 100, 2)
                    signals.append({
                        'stock':  stock,
                        'date':   active_trade['entry_date'],
                        'year':   active_trade['entry_date'].year,
                        'entry':  active_trade['entry_price'],
                        'exit':   round(close, 2),
                        'ret':    ret,
                        'win':    ret >= 0,
                        'reason': 'sl_hit' if sl_hit else 'time_up',
                        'days':   days_held,
                        **active_trade['meta'],
                    })
                    active_trade = None
                continue
            if st_dir[i] != 1:
                continue
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
                'entry_date':  current_date,
                'entry_price': close,
                'sl_price':    result['sl_price'],
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

def print_summary(setup_name, signals):
    if not signals:
        print(f"\nSetup {setup_name}: No signals")
        return
    wins   = [s for s in signals if s['win']]
    losses = [s for s in signals if not s['win']]
    wr     = round(len(wins)/len(signals)*100, 1)
    avgw   = round(sum(s['ret'] for s in wins)/len(wins), 1) if wins else 0
    avgl   = round(sum(s['ret'] for s in losses)/len(losses), 1) if losses else 0
    rr     = round(abs(avgw/avgl), 2) if avgl != 0 else 0
    exp    = round((len(wins)/len(signals))*avgw + (len(losses)/len(signals))*avgl, 1)
    maxg   = round(max(s['ret'] for s in signals), 1)
    maxl   = round(min(s['ret'] for s in signals), 1)

    print(f"\n{'='*60}")
    print(f"VCP SETUP {setup_name} — {len(signals)} signals — {len(set(s['stock'] for s in signals))} stocks")
    print(f"{'='*60}")
    print(f"Win Rate   : {wr}%")
    print(f"Avg Win    : +{avgw}% | Avg Loss: {avgl}%")
    print(f"Max Gain   : +{maxg}% | Max Loss: {maxl}%")
    print(f"R:R        : {rr} | Expectancy: {exp:+.1f}%")

    by_year = defaultdict(list)
    for s in signals:
        by_year[s['year']].append(s)
    print(f"\n{'YEAR':<6} {'SIG':>5} {'WR%':>6} {'AVG':>8} {'EXP':>8}")
    print('-'*36)
    for yr in sorted(by_year.keys()):
        ys   = by_year[yr]
        yw   = sum(1 for s in ys if s['win'])
        ywr  = round(yw/len(ys)*100)
        yavg = round(sum(s['ret'] for s in ys)/len(ys), 1)
        yal  = [s for s in ys if not s['win']]
        yaw  = [s for s in ys if s['win']]
        yexp = round((yw/len(ys))*(sum(s['ret'] for s in yaw)/len(yaw) if yaw else 0) +
                     ((len(ys)-yw)/len(ys))*(sum(s['ret'] for s in yal)/len(yal) if yal else 0), 1)
        print(f"{yr:<6} {len(ys):>5} {ywr:>5}% {yavg:>+8.1f}% {yexp:>+8.1f}%")

    # Save to file
    out_file = f"/Users/sahib/spotmychart-api/scripts/results_vcp_setup_{setup_name}.txt"
    with open(out_file, 'w') as f:
        f.write(f"VCP SETUP {setup_name} — 2126 stocks — 2015-2026\n")
        f.write(f"Total: {len(signals)} | WR: {wr}% | R:R: {rr} | Exp: {exp:+.1f}%\n\n")
        f.write(f"{'STOCK':<12} {'DATE':<12} {'ENTRY':>8} {'EXIT':>8} {'RET%':>7} {'W':>2} {'DAYS':>5} {'REASON':<8}\n")
        f.write('-'*65 + '\n')
        for s in sorted(signals, key=lambda x: x['date']):
            w = 'W' if s['win'] else 'L'
            f.write(f"{s['stock']:<12} {str(s['date']):<12} {s['entry']:>8.2f} {s['exit']:>8.2f} {s['ret']:>+7.1f}% {w:>2} {s['days']:>5} {s['reason']:<8}\n")
    print(f"\nSaved to: {out_file}")

# ── MAIN ─────────────────────────────────────────────────────────────
print("Fetching tickers from Supabase...")
tickers = fetch_all_tickers()

print(f"\nRunning VCP — All 4 Setups — {len(tickers)} stocks — 2015-2026")
print("Estimated time: 3-5 hours\n")

for setup in ["A", "B", "C", "D"]:
    print(f"\n{'='*60}")
    print(f"Starting Setup {setup}...")
    signals = run_setup(setup, tickers)
    print_summary(setup, signals)

print("\n\nAll done!")