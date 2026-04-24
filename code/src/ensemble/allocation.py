"""Portfolio allocation strategies for Top-5 picks.

Each function takes a DataFrame with columns [stock_id, score_q] and returns
the same DataFrame with a `weight` column normalized to sum=1.0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def allocate_equal(picks: pd.DataFrame) -> pd.DataFrame:
    """Equal weight (0.2 each for k=5)."""
    out = picks.copy()
    out["weight"] = 1.0 / len(out)
    return out


def allocate_linear(picks: pd.DataFrame, floor: float = 0.05) -> pd.DataFrame:
    """Linear in score_q after min-shift, with floor.

    w_i ∝ (score_i - min_score), clipped to [floor, inf), then renormalized.
    floor=0.05 → no stock gets less than 5%
    floor=0.15 → no stock gets less than 15% (limits winner-take-all)
    """
    out = picks.copy()
    s = out["score_q"].astype(float)
    w = s - s.min()
    if w.sum() > 1e-12:
        w = w / w.sum()
        w = w.clip(lower=floor)
        w = w / w.sum()
    else:
        w = pd.Series([1.0 / len(out)] * len(out), index=out.index)
    out["weight"] = w.round(6).values
    residual = 1.0 - out["weight"].sum()
    out.loc[out["weight"].idxmax(), "weight"] += residual
    return out


def allocate_softmax(picks: pd.DataFrame, temperature: float = 0.3) -> pd.DataFrame:
    """Softmax over score_q / temperature.

    temperature=1.0 → close to equal
    temperature=0.3 → sharper, more toward top-1
    """
    out = picks.copy()
    s = out["score_q"].astype(float).values
    s = s - s.max()  # numerical stability
    w = np.exp(s / max(temperature, 1e-6))
    w = w / w.sum()
    out["weight"] = np.round(w, 6)
    residual = 1.0 - out["weight"].sum()
    out.loc[out["weight"].idxmax(), "weight"] += residual
    return out


def allocate_by_mode(picks: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Dispatch allocation by mode name."""
    if mode == "equal":
        return allocate_equal(picks)
    elif mode == "linear":
        return allocate_linear(picks, floor=0.05)
    elif mode == "linear_floor15":
        return allocate_linear(picks, floor=0.15)
    elif mode == "softmax_t03":
        return allocate_softmax(picks, temperature=0.3)
    elif mode == "softmax_t10":
        return allocate_softmax(picks, temperature=1.0)
    else:
        raise ValueError(f"unknown allocation mode: {mode}")
