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


def test_optimize_weights_icir_favors_strong_signal_model():
    """master 信号强 → w_master 应高于其他两者，但 KL 约束下 ≤ 0.7"""
    from code.src.ensemble.blender import optimize_weights_icir, rank_normalize_daily
    rng = np.random.default_rng(42)
    n_days, n_stocks = 60, 80
    dates = pd.date_range("2025-01-01", periods=n_days)
    instruments = [f"s{i:03d}" for i in range(n_stocks)]
    rows_label = []
    for d in dates:
        for ins in instruments:
            rows_label.append((ins, d, rng.normal()))
    labels = pd.DataFrame(rows_label, columns=["instrument", "datetime", "label"])

    def make_scores(correlation):
        rs = []
        for _, row in labels.iterrows():
            noise = rng.normal()
            rs.append((row["instrument"], row["datetime"], correlation * row["label"] + (1 - correlation) * noise))
        return rank_normalize_daily(pd.DataFrame(rs, columns=["instrument", "datetime", "score"]))

    scores = {
        "lgb": make_scores(0.05),
        "master": make_scores(0.30),
        "mixer": make_scores(0.02),
    }
    res = optimize_weights_icir(scores, labels, lam=0.2, model_order=["lgb", "master", "mixer"], seed=42)
    w = res["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= -1e-9 for v in w.values())
    assert w["master"] > w["lgb"] and w["master"] > w["mixer"]
    assert w["master"] <= 0.70  # KL shrink 应把极端解拉回
