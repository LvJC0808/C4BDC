"""Alpha360 feature engineering (Qlib-style).

For each (stock, date) pair, stack the past `lookback` (=60) days of 6 raw fields
[open, high, low, close, volume, amount] and normalize each by the value on day t.

Produces a 360-dim dense vector per (date, stock). Used mainly as StockMixer input
but also useful as an additional GBDT feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm


ALPHA360_FIELDS = ['开盘', '最高', '最低', '收盘', '成交量', '成交额']
ALPHA360_COLUMNS = [f'alpha360_{i}' for i in range(360)]


def _normalize_window(window: np.ndarray) -> np.ndarray:
    """Normalize a (lookback, 6) window by day-t values (last row).

    - open/high/low/close: divide by close_t
    - volume: divide by volume_t + 1e-12
    - amount: divide by amount_t + 1e-12
    """
    close_t = window[-1, 3]
    volume_t = window[-1, 4]
    amount_t = window[-1, 5]
    out = np.empty_like(window, dtype=np.float32)
    out[:, 0] = window[:, 0] / (close_t + 1e-12)
    out[:, 1] = window[:, 1] / (close_t + 1e-12)
    out[:, 2] = window[:, 2] / (close_t + 1e-12)
    out[:, 3] = window[:, 3] / (close_t + 1e-12)
    out[:, 4] = window[:, 4] / (volume_t + 1e-12)
    out[:, 5] = window[:, 5] / (amount_t + 1e-12)
    return out


def engineer_alpha360(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Compute Alpha360 features per stock.

    Parameters
    ----------
    df : DataFrame
        Must contain 股票代码, 日期, and the 6 raw fields in ALPHA360_FIELDS.
    lookback : int
        Number of past days to stack. Default 60.

    Returns
    -------
    DataFrame with columns: instrument, datetime, alpha360_0..alpha360_359.
    """
    out_rows = []
    for code, g in tqdm(df.groupby('股票代码'), desc="alpha360"):
        g = g.sort_values('日期').reset_index(drop=True)
        if len(g) < lookback:
            continue
        raw = g[ALPHA360_FIELDS].astype(np.float32).values  # (T, 6)
        dates = pd.to_datetime(g['日期']).values  # (T,)

        # sliding windows (T-lookback+1, lookback, 6)
        n = len(g)
        # Use a Python loop; vectorized stride_tricks adds complexity without
        # dominating the cost in this competition-scale dataset.
        for t in range(lookback - 1, n):
            window = raw[t - lookback + 1: t + 1]  # (lookback, 6)
            norm = _normalize_window(window)        # (lookback, 6)
            flat = norm.reshape(-1)                 # 360
            out_rows.append((code, dates[t], flat))

    if not out_rows:
        return pd.DataFrame(columns=['instrument', 'datetime'] + ALPHA360_COLUMNS)

    codes = [r[0] for r in out_rows]
    dts = [r[1] for r in out_rows]
    mat = np.stack([r[2] for r in out_rows], axis=0).astype(np.float32)

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    feat_df = pd.DataFrame(mat, columns=ALPHA360_COLUMNS)
    feat_df.insert(0, 'datetime', pd.to_datetime(dts))
    feat_df.insert(0, 'instrument', codes)
    return feat_df
