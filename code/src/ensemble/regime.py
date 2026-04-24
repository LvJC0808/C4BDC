"""Regime signal: large-cap vs small-cap cumulative return over lookback window."""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_regime(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 10,
    large_quantile: float = 0.67,
    small_quantile: float = 0.33,
    threshold: float = 0.01,
) -> dict:
    """Classify regime by (large-cap mean return) - (small-cap mean return).

    panel must have columns: datetime, instrument, close, log_mktcap.
    Returns dict with keys: regime, r_large, r_small, diff, n_large, n_small.
    """
    as_of = pd.Timestamp(as_of)
    p = panel.copy()
    p["datetime"] = pd.to_datetime(p["datetime"])
    snap = p[p["datetime"] == as_of][["instrument", "log_mktcap"]].dropna()
    if snap.empty:
        return {"regime": "neutral", "r_large": np.nan, "r_small": np.nan,
                "diff": 0.0, "n_large": 0, "n_small": 0}

    q_hi = snap["log_mktcap"].quantile(large_quantile)
    q_lo = snap["log_mktcap"].quantile(small_quantile)
    large_ids = set(snap[snap["log_mktcap"] >= q_hi]["instrument"])
    small_ids = set(snap[snap["log_mktcap"] <= q_lo]["instrument"])

    window = p[(p["datetime"] <= as_of) &
               (p["datetime"] > as_of - pd.Timedelta(days=lookback_days * 2))]
    window = window.sort_values(["instrument", "datetime"])

    def _cum_ret(ids):
        sub = window[window["instrument"].isin(ids)]
        grp = sub.groupby("instrument")["close"]
        rets = grp.apply(lambda s: s.iloc[-1] / s.iloc[0] - 1 if len(s) >= 2 else np.nan)
        return float(rets.dropna().mean()) if len(rets.dropna()) else np.nan

    r_large = _cum_ret(large_ids)
    r_small = _cum_ret(small_ids)
    diff = (r_large - r_small) if (not np.isnan(r_large) and not np.isnan(r_small)) else 0.0
    regime = "large_cap_rotation" if diff > threshold else "neutral"
    return {"regime": regime, "r_large": r_large, "r_small": r_small,
            "diff": float(diff), "n_large": len(large_ids),
            "n_small": len(small_ids)}
