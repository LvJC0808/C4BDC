"""Portfolio builder with confidence-based sizing (Task 13)."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .blender import rank_normalize


def compute_confidence(
    blended_scores_today: pd.Series,
    model_topk_sets: List[set],
    rolling_ic: float,
    top_k: int = 5,
) -> float:
    """Confidence = 0.4*c_disp + 0.4*c_ic + 0.2*c_agree, clipped to [0, 1]."""
    ranks = rank_normalize(blended_scores_today)
    topk_ranks = ranks.nlargest(top_k)
    top5_mean_rank = float(topk_ranks.mean()) if len(topk_ranks) else 0.5
    median_rank = 0.5
    c_disp = float(np.clip((top5_mean_rank - median_rank) / 0.5, 0.0, 1.0))

    c_ic = float(np.clip(rolling_ic / 0.05, 0.0, 1.0))

    if model_topk_sets:
        inter = set.intersection(*model_topk_sets) if len(model_topk_sets) > 1 else set(model_topk_sets[0])
        c_agree = len(inter) / float(top_k)
    else:
        c_agree = 0.0
    c_agree = float(np.clip(c_agree, 0.0, 1.0))

    c = 0.4 * c_disp + 0.4 * c_ic + 0.2 * c_agree
    return float(np.clip(c, 0.0, 1.0))


def build_portfolio(
    blended_scores_today: pd.DataFrame,
    confidence: float,
    alpha: float,
    top_k: int,
    min_position: float = 0.5,
) -> pd.DataFrame:
    """Build equal-weighted Top-K portfolio with confidence-scaled gross exposure."""
    total_position = min_position + alpha * (1.0 - min_position) * float(confidence)
    total_position = float(np.clip(total_position, 0.0, 1.0))
    topk = blended_scores_today.nlargest(top_k, "final_score").copy()
    weight = total_position / max(len(topk), 1)
    topk["weight"] = weight
    out = topk[["instrument", "weight"]].rename(columns={"instrument": "stock_id"})
    return out.reset_index(drop=True)


def grid_search_portfolio_params(
    blended_scores_panel: pd.DataFrame,
    labels: pd.DataFrame,
    top_k_candidates: List[int],
    alpha_candidates: List[float],
    confidence_fn: Optional[Callable] = None,
    date_col: str = "datetime",
    id_col: str = "instrument",
) -> dict:
    """Iterate (K, α) grid, simulate realized return, return best."""
    merged = blended_scores_panel.merge(labels, on=[id_col, date_col], how="inner")
    best = {"top_k": None, "alpha": None, "score": -np.inf}
    for K in top_k_candidates:
        for alpha in alpha_candidates:
            total = 0.0
            for _, g in merged.groupby(date_col):
                if confidence_fn is None:
                    conf = 1.0
                else:
                    conf = confidence_fn(g)
                today_df = g[[id_col, "final_score"]]
                port = build_portfolio(today_df, conf, alpha, K)
                picks = port.merge(
                    g[[id_col, "label"]].rename(columns={id_col: "stock_id"}),
                    on="stock_id",
                    how="left",
                )
                total += float((picks["weight"] * picks["label"]).sum())
            if total > best["score"]:
                best = {"top_k": int(K), "alpha": float(alpha), "score": float(total)}
    return best


def _self_test() -> None:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    instruments = [f"S{i:02d}" for i in range(20)]

    rows = []
    for d in dates:
        for s in instruments:
            rows.append((s, d))
    panel = pd.DataFrame(rows, columns=["instrument", "datetime"])
    panel["final_score"] = rng.normal(size=len(panel))
    labels = panel[["instrument", "datetime"]].copy()
    labels["label"] = rng.normal(size=len(panel))

    today = panel[panel["datetime"] == dates[0]].copy()
    scores_s = today.set_index("instrument")["final_score"]
    top_sets = [
        set(scores_s.nlargest(5).index.tolist()),
        set(scores_s.sample(5, random_state=1).index.tolist()),
        set(scores_s.nlargest(7).index.tolist()[:5]),
    ]
    c = compute_confidence(scores_s, top_sets, rolling_ic=0.03, top_k=5)
    assert 0.0 <= c <= 1.0, c

    port = build_portfolio(today[["instrument", "final_score"]], confidence=c, alpha=0.5, top_k=5)
    assert len(port) == 5
    assert port["weight"].sum() <= 1.0 + 1e-9
    assert list(port.columns) == ["stock_id", "weight"]
    print(f"confidence={c:.4f} port_sum={port['weight'].sum():.4f}")

    best = grid_search_portfolio_params(
        panel, labels, top_k_candidates=[3, 5], alpha_candidates=[0.3, 0.5, 0.7],
    )
    assert best["top_k"] in [3, 5]
    assert best["alpha"] in [0.3, 0.5, 0.7]
    print(f"best K={best['top_k']} alpha={best['alpha']} score={best['score']:.4f}")
    print("OK")


if __name__ == "__main__":
    _self_test()
