"""Rolling backtest wrapper for the competition baseline (StockTransformer).

Reuses model/60_158+39/best_model.pth + scaler.pkl + config.json.

For each eval date T:
  1. Slice raw stock data up to T
  2. Run engineer_features_158plus39 once per stock (cached)
  3. Build [1, N, 60, F] sequence tensor for the last 60 days ending at T
  4. Forward StockTransformer → score per stock → Top-5
  5. Compute realized open_{T+5}/open_{T+1}-1

Compare against LGB-only rolling backtest (same eval window).
"""
from __future__ import annotations
import argparse, json, os, sys, multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code" / "src"))

from model import StockTransformer  # type: ignore
from utils import engineer_features_158plus39  # type: ignore

FEATURE_COLS_158_39 = [
    'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
    'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
    'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
    'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
    'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
    'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
    'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
    'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
    'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
    'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
    'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
    'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
    'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
    'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
]


def realized_return(stock_df, T, tickers):
    future = stock_df[stock_df['日期'] > T].sort_values(['股票代码', '日期'])
    out = {}
    for tic in tickers:
        g = future[future['股票代码'] == tic].head(5)
        if len(g) < 5:
            out[tic] = np.nan
        else:
            p1 = float(g.iloc[0]['开盘']); p5 = float(g.iloc[-1]['开盘'])
            out[tic] = (p5 - p1) / (p1 + 1e-12)
    return pd.Series(out, name='ret')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='./data/train.csv')
    ap.add_argument('--model_dir', default='./model/60_158+39')
    ap.add_argument('--eval_start', default='2025-11-03')
    ap.add_argument('--eval_end', default='2026-03-06')
    ap.add_argument('--out', default='test/rolling_backtest_baseline.csv')
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, 'config.json')) as f:
        cfg = json.load(f)

    raw = pd.read_csv(args.data_path, dtype={'股票代码': str})
    raw['股票代码'] = raw['股票代码'].astype(str).str.zfill(6)
    raw['日期'] = pd.to_datetime(raw['日期'])

    stock_ids = sorted(raw['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    print(f'[baseline-rb] stocks={len(stock_ids)}  rows={len(raw)}')

    # Feature engineering once per stock (on full history), cached
    print('[baseline-rb] feature engineering (one-shot per stock)')
    groups = [g for _, g in raw.groupby('股票代码', sort=False)]
    n_proc = min(6, mp.cpu_count())
    with mp.Pool(processes=n_proc) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_158plus39, groups),
                                    total=len(groups), desc='feat'))
    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])
    features = FEATURE_COLS_158_39
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(os.path.join(args.model_dir, 'scaler.pkl'))
    processed[features] = scaler.transform(processed[features])

    all_dates = sorted(pd.to_datetime(raw['日期'].unique()))
    eval_start = pd.Timestamp(args.eval_start); eval_end = pd.Timestamp(args.eval_end)
    eval_dates = [d for d in all_dates if eval_start <= d <= eval_end]
    # Need 5 future days for return
    max_idx = len(all_dates) - 6
    max_T = all_dates[max_idx]
    eval_dates = [d for d in eval_dates if d <= max_T]
    print(f'[baseline-rb] {len(eval_dates)} eval dates: {eval_dates[0].date()}..{eval_dates[-1].date()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(input_dim=len(features), config=cfg, num_stocks=len(stock_ids)).to(device)
    sd = torch.load(os.path.join(args.model_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(sd)
    model.eval()

    seq_len = cfg['sequence_length']
    # Group processed by stock for fast slicing
    proc_by_stock = {sid: g.sort_values('日期').reset_index(drop=True)
                     for sid, g in processed.groupby('股票代码', sort=False)}

    rows = []
    for T in tqdm(eval_dates, desc='rolling'):
        seqs = []; seq_ids = []
        for sid in stock_ids:
            g = proc_by_stock.get(sid)
            if g is None:
                continue
            g_hist = g[g['日期'] <= T].tail(seq_len)
            if len(g_hist) == seq_len:
                seqs.append(g_hist[features].values.astype(np.float32))
                seq_ids.append(sid)
        if len(seqs) < 5:
            continue
        x = torch.from_numpy(np.asarray(seqs)).unsqueeze(0).to(device)  # [1,N,60,F]
        with torch.no_grad():
            scores = model(x).squeeze(0).detach().cpu().numpy()
        order = np.argsort(scores)[::-1]
        top5 = [seq_ids[i] for i in order[:5]]
        ret = realized_return(raw, T, top5)
        rows.append({
            'date': T.date(),
            'top5': ','.join(top5),
            'mean_ret': float(ret.mean()),
            'n_valid': int(ret.notna().sum()),
        })
        # Periodic flush
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Summary
    r = df['mean_ret'].dropna()
    print('=' * 60)
    print(f'  Baseline StockTransformer — {len(r)} dates')
    print(f'  Mean 5d return    : {r.mean()*100:+.4f}%')
    print(f'  Median            : {r.median()*100:+.4f}%')
    print(f'  Std               : {r.std()*100:.4f}%')
    print(f'  Win rate (>0)     : {(r>0).mean()*100:.2f}%')
    from scipy import stats
    t, p = stats.ttest_1samp(r, 0.0)
    print(f'  t vs 0            : t={t:+.3f} p={p:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
