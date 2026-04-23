import numpy as np
import pandas as pd
import pytest
from code.src.ensemble.blender import (
    rank_normalize_daily,
    compute_blend_ic_series,
    icir_objective,
)


def _synthetic_scores(n_days=30, n_stocks=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days)
    instruments = [f"s{i:03d}" for i in range(n_stocks)]
    rows = []
    for d in dates:
        for ins in instruments:
            rows.append((ins, d, rng.normal()))
    return pd.DataFrame(rows, columns=["instrument", "datetime", "score"])


def test_rank_normalize_daily_within_unit_interval():
    df = _synthetic_scores()
    out = rank_normalize_daily(df)
    assert out["score"].between(0.0, 1.0).all()
    for _, g in out.groupby("datetime"):
        assert abs(g["score"].mean() - 0.5) < 1e-6


def test_compute_blend_ic_series_shape_and_range():
    rng = np.random.default_rng(1)
    scores = {m: rank_normalize_daily(_synthetic_scores(seed=i)) for i, m in enumerate(["lgb", "master", "mixer"])}
    dates = scores["lgb"]["datetime"].unique()
    labels = []
    for d in dates:
        for ins in scores["lgb"][scores["lgb"]["datetime"] == d]["instrument"]:
            labels.append((ins, d, rng.normal()))
    labels = pd.DataFrame(labels, columns=["instrument", "datetime", "label"])
    ics = compute_blend_ic_series(scores, labels, weights={"lgb": 1/3, "master": 1/3, "mixer": 1/3})
    assert ics.shape == (len(dates),)
    assert ((-1.0 <= ics) & (ics <= 1.0)).all()


def test_icir_objective_equal_weight_kl_is_zero():
    # KL(w || 1/3) = 0 when w = 1/3
    rng = np.random.default_rng(2)
    n = 20
    scores = {m: rank_normalize_daily(_synthetic_scores(n_days=n, seed=i)) for i, m in enumerate(["lgb", "master", "mixer"])}
    dates = scores["lgb"]["datetime"].unique()
    labels = []
    for d in dates:
        for ins in scores["lgb"][scores["lgb"]["datetime"] == d]["instrument"]:
            labels.append((ins, d, rng.normal()))
    labels = pd.DataFrame(labels, columns=["instrument", "datetime", "label"])
    val_zero_lam = icir_objective([1/3, 1/3, 1/3], scores, labels, lam=0.0, model_order=["lgb", "master", "mixer"])
    val_nonzero_lam = icir_objective([1/3, 1/3, 1/3], scores, labels, lam=1.0, model_order=["lgb", "master", "mixer"])
    assert abs(val_zero_lam - val_nonzero_lam) < 1e-9  # KL=0 at equal weight
