"""Unified feature-building pipeline.

Orchestrates:
  1. Load CSVs from data/ directory (stock_data, industry_map, stock_basic,
     hs300_history, csi300_index, trade_calendar).
  2. Sample-pool filter: HS300 constituent ∩ listed ≥250d ∩ not long-suspended.
  3. Compute Alpha158 / Alpha360 / valuation / market / beta / log_mktcap.
  4. Neutralize + rank-gauss.
  5. Build label panel.
  6. Cache to `temp/features/*.parquet`.

`build_feature_sets()` returns a dict keyed by model:
    {
        'lgb':    DataFrame[instrument, datetime, <alpha158 + valuation + beta>] + label
        'master': dict{features: long-form panel of alpha158,
                       market: per-date 63-dim market context,
                       label_raw_df, label_df},
        'mixer':  dict{features: long-form panel of alpha360, market, label},
    }

Caching keys = hash(start_date, end_date, feature_version).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from .alpha158 import engineer_alpha158, ALPHA158_COLUMNS
from .alpha360 import engineer_alpha360, ALPHA360_COLUMNS
from .valuation import (
    engineer_valuation, engineer_market, engineer_beta, engineer_log_mktcap,
    VALUATION_COLUMNS, MARKET_COLUMNS,
)
from .neutralize import neutralize_cross_section, build_label, rank_gauss_label


FEATURE_VERSION = 'v1.0'


# ---------------- Data loading ----------------

def load_raw_data(data_path: str) -> dict:
    """Read all CSVs in data/ into DataFrames."""
    paths = {
        'stock': os.path.join(data_path, 'stock_data.csv'),
        'industry': os.path.join(data_path, 'industry_map.csv'),
        'basic': os.path.join(data_path, 'stock_basic.csv'),
        'hs300_hist': os.path.join(data_path, 'hs300_history.csv'),
        'csi300': os.path.join(data_path, 'csi300_index.csv'),
        'calendar': os.path.join(data_path, 'trade_calendar.csv'),
    }
    out = {}
    for name, p in paths.items():
        if os.path.exists(p):
            out[name] = pd.read_csv(p)
        else:
            out[name] = None
    # Normalize stock_id formatting
    if out['stock'] is not None:
        out['stock']['股票代码'] = out['stock']['股票代码'].astype(str).str.zfill(6)
    if out['industry'] is not None:
        out['industry']['stock_id'] = out['industry']['stock_id'].astype(str).str.zfill(6)
    return out


# ---------------- Sample-pool filter ----------------

def _listing_mask(stock_df: pd.DataFrame, basic_df: Optional[pd.DataFrame],
                  min_list_days: int = 250) -> pd.Series:
    """Keep only rows where the stock has been listed for >= min_list_days."""
    if basic_df is None or 'ipoDate' not in basic_df.columns:
        return pd.Series(True, index=stock_df.index)
    ipo = dict(zip(basic_df['stock_id'].astype(str).str.zfill(6),
                   pd.to_datetime(basic_df['ipoDate'], errors='coerce')))
    codes = stock_df['股票代码'].astype(str).str.zfill(6)
    ipo_series = codes.map(ipo)
    dt = pd.to_datetime(stock_df['日期'])
    diff = (dt - ipo_series).dt.days
    return diff >= min_list_days


def _suspend_mask(stock_df: pd.DataFrame, max_suspend: int = 5) -> pd.Series:
    """Drop rows where the stock was suspended (volume==0) for >max_suspend
    consecutive trading days ending at t-1."""
    mask = pd.Series(True, index=stock_df.index)
    for code, g in stock_df.groupby('股票代码'):
        g = g.sort_values('日期')
        suspended = (pd.to_numeric(g['成交量'], errors='coerce').fillna(0) == 0)
        rolling = suspended.rolling(max_suspend, min_periods=1).sum()
        mask.loc[g.index] = rolling < max_suspend
    return mask


def _hs300_membership_mask(stock_df: pd.DataFrame,
                           hs300_hist: Optional[pd.DataFrame]) -> pd.Series:
    """Keep rows whose stock was an HS300 constituent as of the most recent
    snapshot on or before the row's date."""
    if hs300_hist is None or hs300_hist.empty:
        return pd.Series(True, index=stock_df.index)
    hs300_hist = hs300_hist.copy()
    hs300_hist['date'] = pd.to_datetime(hs300_hist['date'])
    hs300_hist['stock_id'] = hs300_hist['stock_id'].astype(str).str.zfill(6)

    codes = stock_df['股票代码'].astype(str).str.zfill(6).values
    dates = pd.to_datetime(stock_df['日期']).values

    snap_dates = np.array(sorted(hs300_hist['date'].unique()))
    constituents = {d: set(hs300_hist.loc[hs300_hist['date'] == d, 'stock_id'])
                    for d in snap_dates}

    out = np.ones(len(stock_df), dtype=bool)
    for i, (c, d) in enumerate(zip(codes, dates)):
        # find most recent snapshot <= d
        idx = np.searchsorted(snap_dates, d, side='right') - 1
        if idx < 0:
            continue  # before earliest snapshot, allow
        if c not in constituents[snap_dates[idx]]:
            out[i] = False
    return pd.Series(out, index=stock_df.index)


def apply_sample_pool(stock_df: pd.DataFrame, raw: dict,
                      min_list_days: int = 250,
                      max_suspend: int = 5,
                      use_hs300_membership: bool = True) -> pd.DataFrame:
    m1 = _listing_mask(stock_df, raw.get('basic'), min_list_days)
    m2 = _suspend_mask(stock_df, max_suspend)
    m3 = (_hs300_membership_mask(stock_df, raw.get('hs300_hist'))
          if use_hs300_membership
          else pd.Series(True, index=stock_df.index))
    mask = m1 & m2 & m3
    return stock_df.loc[mask].reset_index(drop=True)


# ---------------- Cache ----------------

def _cache_key(start: str, end: str, version: str = FEATURE_VERSION) -> str:
    raw = json.dumps({'start': start, 'end': end, 'v': version}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(temp_dir: str, key: str, tag: str) -> str:
    os.makedirs(os.path.join(temp_dir, 'features'), exist_ok=True)
    return os.path.join(temp_dir, 'features', f'{tag}_{key}.parquet')


# ---------------- Main builder ----------------

def build_feature_sets(
    data_path: str,
    temp_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> dict:
    """Build all feature panels. Heavy but idempotent with cache."""
    raw = load_raw_data(data_path)
    stock = raw['stock']
    if stock is None:
        raise FileNotFoundError(f'stock_data.csv not found in {data_path}')

    stock['日期'] = pd.to_datetime(stock['日期']).dt.strftime('%Y-%m-%d')
    if start_date:
        stock = stock[stock['日期'] >= start_date]
    if end_date:
        stock = stock[stock['日期'] <= end_date]

    actual_start = stock['日期'].min()
    actual_end = stock['日期'].max()
    key = _cache_key(actual_start, actual_end)

    # 1. Sample pool
    stock = apply_sample_pool(stock, raw)

    # 2. Auxiliary panels ----------------
    mktcap_df = engineer_log_mktcap(stock)
    beta_df = (engineer_beta(stock, raw['csi300'])
               if raw['csi300'] is not None
               else pd.DataFrame(columns=['instrument', 'datetime', 'beta60']))

    # Industry broadcast
    ind_map = raw.get('industry')
    if ind_map is None:
        ind_map = pd.DataFrame({'stock_id': stock['股票代码'].unique(),
                                'sw_industry': 'UNKNOWN'})

    # 3. Alpha158 ----------------
    a158_path = _cache_path(temp_dir, key, 'alpha158')
    if use_cache and os.path.exists(a158_path):
        a158 = pd.read_parquet(a158_path)
    else:
        a158 = engineer_alpha158(stock)[
            ['instrument', 'datetime'] + ALPHA158_COLUMNS
        ]
        a158['instrument'] = a158['instrument'].astype(str).str.zfill(6)
        a158.to_parquet(a158_path)

    # 4. Alpha360 ----------------
    a360_path = _cache_path(temp_dir, key, 'alpha360')
    if use_cache and os.path.exists(a360_path):
        a360 = pd.read_parquet(a360_path)
    else:
        a360 = engineer_alpha360(stock)
        a360['instrument'] = a360['instrument'].astype(str).str.zfill(6)
        a360.to_parquet(a360_path)

    # 5. Valuation ----------------
    val_path = _cache_path(temp_dir, key, 'valuation')
    if use_cache and os.path.exists(val_path):
        val = pd.read_parquet(val_path)
    else:
        val = engineer_valuation(stock, ind_map)
        val.to_parquet(val_path)

    # 6. Market (CSI300, 63-dim) ----------------
    if raw['csi300'] is not None:
        market = engineer_market(raw['csi300'])
    else:
        market = pd.DataFrame(columns=['datetime'] + MARKET_COLUMNS)

    # 7. Label ----------------
    label_raw = build_label(stock)
    label = rank_gauss_label(label_raw).drop(columns=['label_raw'], errors='ignore')

    # 8. Merge auxiliary (industry, log_mktcap, beta) onto instrument/datetime
    ind_small = ind_map[['stock_id', 'sw_industry']].rename(columns={'stock_id': 'instrument'})

    def _attach(panel: pd.DataFrame) -> pd.DataFrame:
        p = panel.copy()
        p['instrument'] = p['instrument'].astype(str).str.zfill(6)
        p = p.merge(ind_small, on='instrument', how='left')
        p = p.merge(mktcap_df, on=['instrument', 'datetime'], how='left')
        p = p.merge(beta_df, on=['instrument', 'datetime'], how='left')
        p['log_mktcap'] = p['log_mktcap'].fillna(0.0)
        p['beta60'] = p['beta60'].fillna(0.0)
        p['sw_industry'] = p['sw_industry'].fillna('UNKNOWN')
        return p

    a158_full = _attach(a158)
    a360_full = _attach(a360)
    val_full = _attach(val)

    # 9. Neutralize
    a158_neu_path = _cache_path(temp_dir, key, 'alpha158_neu')
    if use_cache and os.path.exists(a158_neu_path):
        a158_neu = pd.read_parquet(a158_neu_path)
    else:
        a158_neu = neutralize_cross_section(a158_full, ALPHA158_COLUMNS)
        a158_neu.to_parquet(a158_neu_path)

    val_neu_path = _cache_path(temp_dir, key, 'valuation_neu')
    if use_cache and os.path.exists(val_neu_path):
        val_neu = pd.read_parquet(val_neu_path)
    else:
        val_neu = neutralize_cross_section(val_full, VALUATION_COLUMNS)
        val_neu.to_parquet(val_neu_path)

    # Alpha360 used by StockMixer: keep raw (not neutralized) but attach label

    # 10. LGB feature set = a158_neu + val_neu (dropping duplicate aux cols)
    lgb_feats = a158_neu.merge(
        val_neu[['instrument', 'datetime'] + VALUATION_COLUMNS],
        on=['instrument', 'datetime'], how='inner'
    )
    lgb_feats = lgb_feats.merge(label, on=['instrument', 'datetime'], how='inner')

    # 11. MASTER feature set = a158_neu + market (broadcast by date) + label
    master_feats = a158_neu.merge(label, on=['instrument', 'datetime'], how='inner')

    # 12. Mixer feature set = a360 (raw) + label
    mixer_feats = a360_full.merge(label, on=['instrument', 'datetime'], how='inner')

    return {
        'lgb': {
            'panel': lgb_feats,
            'feature_cols': ALPHA158_COLUMNS + VALUATION_COLUMNS,
        },
        'master': {
            'panel': master_feats,
            'feature_cols': ALPHA158_COLUMNS,
            'market': market,
        },
        'mixer': {
            'panel': mixer_feats,
            'feature_cols': ALPHA360_COLUMNS,
            'market': market,
        },
    }
