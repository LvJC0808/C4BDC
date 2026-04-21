"""Cross-sectional neutralization + rank-gauss transform.

For each trading date independently:
  1. Industry demeaning (subtract group mean per industry)
  2. OLS residual against [log_mktcap, beta60]
  3. 3σ winsorization
  4. Rank → Φ⁻¹ (rank-gauss, mean 0 variance 1)

Labels (future returns) receive the same rank-gauss step.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


def winsorize_sigma(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Clip x to [mean-k*std, mean+k*std]."""
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or not np.isfinite(s):
        return x
    return np.clip(x, m - k * s, m + k * s)


def rank_gauss(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank → inverse CDF of standard normal."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.astype(np.float32)
    ranks = pd.Series(x).rank(method='average').values
    # scale to (0, 1) excluding endpoints to avoid ±inf
    u = (ranks - 0.5) / n
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return norm.ppf(u).astype(np.float32)


def _regress_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residual of OLS y on X (with intercept)."""
    if X.size == 0:
        return y
    Xb = np.column_stack([np.ones(len(y)), X])
    mask = np.isfinite(y) & np.all(np.isfinite(Xb), axis=1)
    if mask.sum() < Xb.shape[1] + 1:
        return y
    try:
        coef, *_ = np.linalg.lstsq(Xb[mask], y[mask], rcond=None)
    except np.linalg.LinAlgError:
        return y
    pred = Xb @ coef
    res = y - pred
    res[~np.isfinite(res)] = 0.0
    return res


def neutralize_cross_section(
    panel: pd.DataFrame,
    feature_cols: Iterable[str],
    industry_col: str = 'sw_industry',
    mktcap_col: str = 'log_mktcap',
    beta_col: str = 'beta60',
    date_col: str = 'datetime',
    do_winsorize: bool = True,
    do_rank_gauss: bool = True,
) -> pd.DataFrame:
    """Neutralize features cross-sectionally per date.

    Parameters
    ----------
    panel : DataFrame
        Long-form panel with columns [instrument, datetime, sw_industry,
        log_mktcap, beta60, <feature_cols>].
    feature_cols : iterable of str
        Columns to be neutralized. Other columns passed through.
    Returns
    -------
    A copy of `panel` with feature_cols replaced by neutralized values.
    """
    feature_cols = list(feature_cols)
    result = panel.copy()
    groups = result.groupby(date_col, sort=False, group_keys=False)

    out_frames = []
    for date, g in groups:
        sub = g.copy()
        # 1. industry demean
        if industry_col in sub.columns:
            ind_means = sub.groupby(industry_col)[feature_cols].transform('mean')
            sub[feature_cols] = sub[feature_cols].values - ind_means.values

        # 2. regress residual on [log_mktcap, beta60]
        X_cols = [c for c in (mktcap_col, beta_col) if c in sub.columns]
        if X_cols:
            X = sub[X_cols].values.astype(np.float64)
            for col in feature_cols:
                y = sub[col].values.astype(np.float64)
                sub[col] = _regress_residual(y, X)

        # 3. winsorize + 4. rank-gauss
        for col in feature_cols:
            arr = sub[col].values.astype(np.float64)
            if do_winsorize:
                arr = winsorize_sigma(arr, k=3.0)
            if do_rank_gauss:
                arr = rank_gauss(arr)
            sub[col] = arr

        out_frames.append(sub)

    return pd.concat(out_frames, ignore_index=False).loc[result.index]


def build_label(
    stock_df: pd.DataFrame,
    horizon_start: int = 1,
    horizon_end: int = 5,
) -> pd.DataFrame:
    """Label = open_{t+horizon_end} / open_{t+horizon_start} - 1.

    Returns DataFrame with columns [instrument, datetime, label_raw].
    Label is NaN for the last (horizon_end) days of each stock.
    """
    frames = []
    for code, g in stock_df.groupby('股票代码'):
        g = g.sort_values('日期').reset_index(drop=True)
        open_ = pd.to_numeric(g['开盘'], errors='coerce')
        buy = open_.shift(-horizon_start)
        sell = open_.shift(-horizon_end)
        label = (sell / buy - 1.0)
        frames.append(pd.DataFrame({
            'instrument': str(code).zfill(6),
            'datetime': pd.to_datetime(g['日期']).values,
            'label_raw': label.values.astype(np.float32),
        }))
    return pd.concat(frames, ignore_index=True)


def rank_gauss_label(label_panel: pd.DataFrame, date_col: str = 'datetime',
                     label_col: str = 'label_raw',
                     out_col: str = 'label') -> pd.DataFrame:
    """Apply rank-gauss to labels per date."""
    out = label_panel.copy()
    def _apply(g):
        g[out_col] = rank_gauss(g[label_col].values)
        return g
    out = out.groupby(date_col, group_keys=False).apply(_apply)
    return out
