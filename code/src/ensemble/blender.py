"""Ensemble rank-blend and grid-search (Task 12)."""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd


def rank_normalize(scores: pd.Series) -> pd.Series:
    """Cross-sectional rank normalization to [0, 1]. NaNs preserved."""
    mask = scores.notna()
    ranks = scores[mask].rank(method="average")
    n = len(ranks)
    if n <= 1:
        normed = pd.Series(0.5, index=ranks.index)
    else:
        normed = (ranks - 1.0) / (n - 1.0)
    out = pd.Series(np.nan, index=scores.index, dtype=float)
    out.loc[mask] = normed
    return out


def rank_normalize_panel(
    df: pd.DataFrame, score_col: str, date_col: str = "datetime"
) -> pd.Series:
    """Group by date and rank-normalize, aligned with df.index."""
    return df.groupby(date_col, group_keys=False)[score_col].apply(rank_normalize)


def blend_scores(
    model_scores: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    date_col: str = "datetime",
    id_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """Rank-normalize per date per model, outer-join, fill 0.5, weighted sum."""
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive.")
    norm_w = {k: v / total_w for k, v in weights.items()}

    merged = None
    for name, df in model_scores.items():
        sub = df[[id_col, date_col, score_col]].copy()
        sub["rank"] = rank_normalize_panel(sub, score_col, date_col=date_col)
        sub = sub[[id_col, date_col, "rank"]].rename(columns={"rank": f"rank_{name}"})
        merged = sub if merged is None else merged.merge(
            sub, on=[id_col, date_col], how="outer"
        )

    final = np.zeros(len(merged), dtype=float)
    for name, w in norm_w.items():
        col = merged[f"rank_{name}"].fillna(0.5).to_numpy()
        final += w * col
    out = merged[[id_col, date_col]].copy()
    out["final_score"] = final
    return out


def grid_search_weights(
    model_scores: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    top_k_list: List[int],
    date_col: str = "datetime",
    id_col: str = "instrument",
) -> dict:
    """Grid search simplex (step 0.1) over weights × K.

    Objective: sum over dates of mean(label) of top-K picks.
    """
    names = list(model_scores.keys())
    n_models = len(names)
    step = 0.1
    points = []
    grid = np.arange(0, 1.0 + 1e-9, step)
    for combo in itertools.product(grid, repeat=n_models):
        if abs(sum(combo) - 1.0) < 1e-6:
            points.append(combo)

    labels = labels.dropna(subset=["label"])
    best = {"weights": None, "top_k": None, "score": -np.inf}
    for combo in points:
        weights = {n: float(w) for n, w in zip(names, combo)}
        blended = blend_scores(
            model_scores, weights, date_col=date_col, id_col=id_col
        )
        merged = blended.merge(labels, on=[id_col, date_col], how="inner")
        for K in top_k_list:
            total = 0.0
            for _, g in merged.groupby(date_col):
                topk = g.nlargest(K, "final_score")
                m = topk["label"].mean()
                if np.isnan(m):
                    continue
                total += m
            if total > best["score"]:
                best = {"weights": weights, "top_k": int(K), "score": float(total)}
    if best["weights"] is None:
        # Fallback: equal weight across models
        best = {
            "weights": {n: 1.0 / n_models for n in names},
            "top_k": int(top_k_list[0]),
            "score": 0.0,
        }
    return best


def rolling_ic_weights(
    model_scores: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    window: int = 20,
    beta: float = 1.0,
    date_col: str = "datetime",
    id_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """Compute per-date per-model weights from rolling Spearman IC.

    Returns DataFrame[date, model, weight] where weights sum to 1 per date.
    For dates with insufficient history, returns uniform weights.
    """
    from scipy.stats import spearmanr

    labels = labels.dropna(subset=["label"])
    all_dates = sorted(set().union(*(set(v[date_col].unique())
                                     for v in model_scores.values())))
    names = list(model_scores.keys())
    n = len(names)

    per_date_ic: Dict[str, Dict] = {nm: {} for nm in names}
    for nm in names:
        df = model_scores[nm]
        merged = df.merge(labels, on=[id_col, date_col], how="inner")
        for dd, g in merged.groupby(date_col):
            if len(g) < 5:
                continue
            ic, _ = spearmanr(g[score_col], g["label"])
            if np.isfinite(ic):
                per_date_ic[nm][dd] = float(ic)

    rows = []
    for i, d in enumerate(all_dates):
        hist_dates = all_dates[max(0, i - window):i]
        if len(hist_dates) < 3:
            for nm in names:
                rows.append({"date": d, "model": nm, "weight": 1.0 / n})
            continue
        ics = {}
        for nm in names:
            day_ics = [per_date_ic[nm][dd] for dd in hist_dates if dd in per_date_ic[nm]]
            ics[nm] = float(np.mean(day_ics)) if day_ics else 0.0
        vals = np.array([ics[nm] for nm in names])
        exp_vals = np.exp(beta * (vals - vals.max()))
        w = exp_vals / exp_vals.sum()
        for j, nm in enumerate(names):
            rows.append({"date": d, "model": nm, "weight": float(w[j])})
    return pd.DataFrame(rows)


def blend_scores_dynamic(
    model_scores: Dict[str, pd.DataFrame],
    weight_df: pd.DataFrame,
    date_col: str = "datetime",
    id_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """Like blend_scores but with per-date weights from weight_df.

    weight_df has columns [date, model, weight]. Vectorized O(N) impl.
    """
    names = list(model_scores.keys())
    n_models = len(names)

    w_wide = weight_df.pivot_table(
        index="date", columns="model", values="weight", aggfunc="first"
    )
    for nm in names:
        if nm not in w_wide.columns:
            w_wide[nm] = np.nan
    w_wide = w_wide[names]

    merged = None
    for name, df in model_scores.items():
        sub = df[[id_col, date_col, score_col]].copy()
        sub["rank"] = rank_normalize_panel(sub, score_col, date_col=date_col)
        sub = sub[[id_col, date_col, "rank"]].rename(columns={"rank": f"rank_{name}"})
        merged = sub if merged is None else merged.merge(
            sub, on=[id_col, date_col], how="outer"
        )

    dates = merged[date_col]
    final = np.zeros(len(merged), dtype=float)
    for name in names:
        ranks = merged[f"rank_{name}"].fillna(0.5).to_numpy()
        w_map = w_wide[name].to_dict()
        ws = dates.map(w_map).fillna(1.0 / n_models).to_numpy()
        final += ws * ranks

    out = merged[[id_col, date_col]].copy()
    out["final_score"] = final
    return out


def _self_test() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    instruments = [f"S{i:02d}" for i in range(30)]
    rows = []
    for d in dates:
        for s in instruments:
            rows.append((s, d))
    base = pd.DataFrame(rows, columns=["instrument", "datetime"])
    labels = base.copy()
    labels["label"] = rng.normal(size=len(base))

    model_scores = {}
    for name in ["m1", "m2", "m3"]:
        df = base.copy()
        df["score"] = labels["label"] * rng.uniform(0.1, 0.5) + rng.normal(
            scale=1.0, size=len(base)
        )
        model_scores[name] = df

    blended = blend_scores(model_scores, {"m1": 1, "m2": 1, "m3": 1})
    assert set(blended.columns) == {"instrument", "datetime", "final_score"}
    assert len(blended) == len(base)
    assert blended["final_score"].between(0, 1).all()

    result = grid_search_weights(model_scores, labels, top_k_list=[3, 4, 5])
    assert set(result.keys()) == {"weights", "top_k", "score"}
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-6
    assert result["top_k"] in [3, 4, 5]
    print(f"best weights={result['weights']} K={result['top_k']} score={result['score']:.4f}")

    wdf = rolling_ic_weights(model_scores, labels, window=5, beta=1.0)
    assert list(wdf.columns) == ["date", "model", "weight"]
    dyn = blend_scores_dynamic(model_scores, wdf)
    assert "final_score" in dyn.columns and len(dyn) > 0

    print("OK")


if __name__ == "__main__":
    _self_test()
