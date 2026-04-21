"""Valuation / Market / Beta / log-market-cap features.

These feature groups are used in addition to Alpha158/360:
- valuation: PE/PB/PS/PCF raw + 20/60d z-score + industry rank (16 columns)
- market  : CSI300-derived 63-dim market context (for MASTER gating)
- beta60  : 60d rolling OLS slope of stock return on index return
- log_mktcap : 20d mean of log(amount) as a market-cap proxy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------- Valuation ----------------

_VAL_RAW = ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
_VAL_PREFIX = {'peTTM': 'pe', 'pbMRQ': 'pb', 'psTTM': 'ps', 'pcfNcfTTM': 'pcf'}


def engineer_valuation(df: pd.DataFrame, industry_map: pd.DataFrame) -> pd.DataFrame:
    """Valuation factors + z-scores + industry cross-sectional rank.

    Parameters
    ----------
    df : DataFrame
        Must have 股票代码, 日期, peTTM, pbMRQ, psTTM, pcfNcfTTM.
    industry_map : DataFrame
        Columns: stock_id, sw_industry.

    Returns
    -------
    DataFrame with columns: instrument, datetime, <16 feature cols>.
    """
    ind_map = dict(zip(industry_map['stock_id'].astype(str).str.zfill(6),
                       industry_map['sw_industry'].fillna('UNKNOWN')))

    frames = []
    for code, g in tqdm(df.groupby('股票代码'), desc="valuation z-score"):
        g = g.sort_values('日期').reset_index(drop=True).copy()
        row = pd.DataFrame({
            'instrument': str(code).zfill(6),
            'datetime': pd.to_datetime(g['日期']),
        })
        for raw in _VAL_RAW:
            pref = _VAL_PREFIX[raw]
            x = pd.to_numeric(g[raw], errors='coerce')
            row[f'{pref}_raw'] = x.values
            m20 = x.rolling(20, min_periods=5).mean()
            s20 = x.rolling(20, min_periods=5).std().replace(0, np.nan)
            m60 = x.rolling(60, min_periods=15).mean()
            s60 = x.rolling(60, min_periods=15).std().replace(0, np.nan)
            row[f'{pref}_z20'] = ((x - m20) / s20).values
            row[f'{pref}_z60'] = ((x - m60) / s60).values
        frames.append(row)

    out = pd.concat(frames, ignore_index=True)
    out['sw_industry'] = out['instrument'].map(ind_map).fillna('UNKNOWN')

    # Industry cross-sectional pct rank per date
    for raw in _VAL_RAW:
        pref = _VAL_PREFIX[raw]
        out[f'{pref}_ind_rank'] = (
            out.groupby(['datetime', 'sw_industry'])[f'{pref}_raw']
               .rank(pct=True, method='average')
        )

    out = out.drop(columns=['sw_industry'])
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


VALUATION_COLUMNS = [
    f'{p}_{s}'
    for p in ['pe', 'pb', 'ps', 'pcf']
    for s in ['raw', 'z20', 'z60', 'ind_rank']
]


# ---------------- Market (CSI300 context, 63-dim) ----------------

def engineer_market(csi300: pd.DataFrame, lookback: int = 8) -> pd.DataFrame:
    """Generate 63-dim market context per trading date from CSI300 index.

    Components:
      - past `lookback` days of 7 series (open/close, high/close, low/close,
        volume, amount, pctChg, 振幅)                               -> 56
      - rolling mean/std over past 20 / 60 days on pctChg           -> 4
      - momentum 5 / 20 / 60 days on close                           -> 3
    Total = 63.
    """
    df = csi300.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg', '振幅']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    series = pd.DataFrame(index=df.index)
    series['s0'] = df['open'] / (df['close'] + 1e-12)
    series['s1'] = df['high'] / (df['close'] + 1e-12)
    series['s2'] = df['low'] / (df['close'] + 1e-12)
    # volume/amount normalize by their own rolling mean to stabilize scale
    series['s3'] = df['volume'] / (df['volume'].rolling(20, min_periods=1).mean() + 1e-12)
    series['s4'] = df['amount'] / (df['amount'].rolling(20, min_periods=1).mean() + 1e-12)
    series['s5'] = df['pctChg'] / 100.0
    series['s6'] = df['振幅'] / 100.0

    mat = series.values.astype(np.float32)  # (T, 7)
    T = len(df)
    F = 7 * lookback + 4 + 3  # 63
    out = np.zeros((T, F), dtype=np.float32)

    for t in range(T):
        if t < lookback - 1:
            continue
        win = mat[t - lookback + 1: t + 1]  # (lookback, 7)
        out[t, :7 * lookback] = win.reshape(-1)
    # rolling stats on pctChg
    pct = series['s5']
    out[:, 56] = pct.rolling(20, min_periods=5).mean().fillna(0).values
    out[:, 57] = pct.rolling(20, min_periods=5).std().fillna(0).values
    out[:, 58] = pct.rolling(60, min_periods=15).mean().fillna(0).values
    out[:, 59] = pct.rolling(60, min_periods=15).std().fillna(0).values
    # momentum
    close = df['close']
    out[:, 60] = (close / close.shift(5) - 1.0).fillna(0).values
    out[:, 61] = (close / close.shift(20) - 1.0).fillna(0).values
    out[:, 62] = (close / close.shift(60) - 1.0).fillna(0).values

    cols = [f'market_{i}' for i in range(F)]
    out_df = pd.DataFrame(out, columns=cols)
    out_df.insert(0, 'datetime', df['date'].values)
    out_df = out_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out_df


MARKET_COLUMNS = [f'market_{i}' for i in range(63)]


# ---------------- Beta-60d ----------------

def engineer_beta(stock_df: pd.DataFrame, csi300: pd.DataFrame,
                  window: int = 60) -> pd.DataFrame:
    """Rolling 60-day OLS slope of stock return on CSI300 return."""
    idx = csi300.copy()
    idx['datetime'] = pd.to_datetime(idx['date'])
    idx['idx_ret'] = pd.to_numeric(idx['pctChg'], errors='coerce') / 100.0
    idx = idx[['datetime', 'idx_ret']].sort_values('datetime')

    frames = []
    for code, g in tqdm(stock_df.groupby('股票代码'), desc="beta60"):
        g = g.sort_values('日期').reset_index(drop=True).copy()
        g['datetime'] = pd.to_datetime(g['日期'])
        g['stk_ret'] = pd.to_numeric(g['涨跌幅'], errors='coerce') / 100.0
        m = g[['datetime', 'stk_ret']].merge(idx, on='datetime', how='left')
        x = m['idx_ret']
        y = m['stk_ret']
        # rolling covariance and variance
        cov = x.rolling(window, min_periods=window // 2).cov(y)
        var = x.rolling(window, min_periods=window // 2).var()
        beta = (cov / var.replace(0, np.nan)).fillna(0.0)
        frames.append(pd.DataFrame({
            'instrument': str(code).zfill(6),
            'datetime': m['datetime'].values,
            'beta60': beta.values.astype(np.float32),
        }))

    out = pd.concat(frames, ignore_index=True)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


# ---------------- log market-cap proxy ----------------

def engineer_log_mktcap(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """log(amount) rolling mean as market-cap proxy (20-day smoothing).

    Since shares-outstanding is not directly provided, we use amount
    (traded value) as a liquidity-weighted proxy for size.
    """
    frames = []
    for code, g in tqdm(df.groupby('股票代码'), desc="log_mktcap"):
        g = g.sort_values('日期').reset_index(drop=True)
        amount = pd.to_numeric(g['成交额'], errors='coerce')
        log_amt = np.log(amount.replace(0, np.nan) + 1.0)
        smooth = log_amt.rolling(window, min_periods=5).mean().fillna(0.0)
        frames.append(pd.DataFrame({
            'instrument': str(code).zfill(6),
            'datetime': pd.to_datetime(g['日期']).values,
            'log_mktcap': smooth.values.astype(np.float32),
        }))
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], 0.0).fillna(0.0)
