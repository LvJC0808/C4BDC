"""Alpha158 feature engineering.

Reuses the `engineer_features` implementation from the baseline (`code/src/utils.py`)
but wraps it as a clean module for multi-stock DataFrames.

Input DataFrame must have columns:
    股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 涨跌幅 (at minimum)

Output: DataFrame with 158 alpha columns + original columns, indexed by (date, stock_id).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import _baseline_utils as bu


def engineer_alpha158(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Alpha158 factors per stock and concatenate.

    Parameters
    ----------
    df : DataFrame
        Must contain 股票代码, 日期, 开盘/收盘/最高/最低/成交量/成交额.

    Returns
    -------
    DataFrame with 158 alpha columns + raw columns. datetime column named 'datetime',
    stock code column named 'instrument'.
    """
    out_frames = []
    MIN_ROWS = 60  # largest alpha158 window
    for code, g in tqdm(df.groupby('股票代码'), desc="alpha158"):
        g = g.sort_values('日期').reset_index(drop=True)
        if len(g) < MIN_ROWS:
            continue
        try:
            feat = bu.engineer_features(g)
        except Exception as e:
            print(f"  [alpha158] skip {code}: {e}")
            continue
        feat['instrument'] = code
        out_frames.append(feat)
    result = pd.concat(out_frames, ignore_index=True)
    result = result.rename(columns={'日期': 'datetime'})
    result['datetime'] = pd.to_datetime(result['datetime'])
    return result


ALPHA158_COLUMNS = [
    # 1. K-line
    'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
    # 2. Price ratio
    'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
] + [f'{p}{w}' for p in ['ROC', 'MA', 'STD'] for w in [5, 10, 20, 30, 60]] + [
    f'{p}{w}' for w in [5, 10, 20, 30, 60] for p in ['BETA', 'RSQR', 'RESI']
] + [f'MAX{w}' for w in [5, 10, 20, 30, 60]] + [f'MIN{w}' for w in [5, 10, 20, 30, 60]] + [
    f'QTLU{w}' for w in [5, 10, 20, 30, 60]
] + [f'QTLD{w}' for w in [5, 10, 20, 30, 60]] + [f'RANK{w}' for w in [5, 10, 20, 30, 60]] + [
    f'RSV{w}' for w in [5, 10, 20, 30, 60]
] + [f'IMAX{w}' for w in [5, 10, 20, 30, 60]] + [f'IMIN{w}' for w in [5, 10, 20, 30, 60]] + [
    f'IMXD{w}' for w in [5, 10, 20, 30, 60]
] + [f'CORR{w}' for w in [5, 10, 20, 30, 60]] + [f'CORD{w}' for w in [5, 10, 20, 30, 60]] + [
    f'CNTP{w}' for w in [5, 10, 20, 30, 60]
] + [f'CNTN{w}' for w in [5, 10, 20, 30, 60]] + [f'CNTD{w}' for w in [5, 10, 20, 30, 60]] + [
    f'SUMP{w}' for w in [5, 10, 20, 30, 60]
] + [f'SUMN{w}' for w in [5, 10, 20, 30, 60]] + [f'SUMD{w}' for w in [5, 10, 20, 30, 60]] + [
    f'VMA{w}' for w in [5, 10, 20, 30, 60]
] + [f'VSTD{w}' for w in [5, 10, 20, 30, 60]] + [f'WVMA{w}' for w in [5, 10, 20, 30, 60]] + [
    f'VSUMP{w}' for w in [5, 10, 20, 30, 60]
] + [f'VSUMN{w}' for w in [5, 10, 20, 30, 60]] + [f'VSUMD{w}' for w in [5, 10, 20, 30, 60]]
