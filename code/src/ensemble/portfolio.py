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
    tradable_ids: set = None,
) -> pd.DataFrame:
    """Build equal-weighted Top-K portfolio with confidence-scaled gross exposure."""
    total_position = min_position + alpha * (1.0 - min_position) * float(confidence)
    total_position = float(np.clip(total_position, 0.0, 1.0))
    df = blended_scores_today
    if tradable_ids is not None:
        df = df[df["instrument"].isin(tradable_ids)]
    topk = df.nlargest(top_k, "final_score").copy()
    actual_k = len(topk)
    if actual_k == 0:
        return pd.DataFrame(columns=["stock_id", "weight"])
    weight = total_position / actual_k
    topk["weight"] = weight
    out = topk[["instrument", "weight"]].rename(columns={"instrument": "stock_id"})
    return out.reset_index(drop=True)


def deterministic_top_k(
    df: pd.DataFrame,
    k: int = 5,
    score_col: str = "score",
    id_col: str = "stock_id",
    quantize: float = 1e-4,
) -> pd.DataFrame:
    """Select a deterministic top-k slice with quantized score tie-breaking."""
    out = df.copy()
    out["score_q"] = (out[score_col] / quantize).round() * quantize
    out = out.sort_values(
        by=["score_q", id_col],
        ascending=[False, True],
        kind="mergesort",
    )
    return out.head(k).reset_index(drop=True)


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
    merged = merged.dropna(subset=["label"])
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
    if best["top_k"] is None:
        best = {"top_k": int(top_k_candidates[0]),
                "alpha": float(alpha_candidates[0]), "score": 0.0}
    return best


def mcap_constrained_topk(
    scores: pd.DataFrame,
    mktcap: pd.DataFrame,
    k: int = 5,
    candidate_k: int = 10,
    min_large_cap: int = 2,
    large_cap_quantile: float = 0.5,
    score_col: str = "score",
    id_col: str = "stock_id",
) -> pd.DataFrame:
    """Select k picks from top candidate_k, enforcing >= min_large_cap large-caps.

    Large-cap threshold = quantile of log_mktcap over the passed-in mktcap frame.
    If the top-candidate_k contains fewer than min_large_cap large-caps, FALL BACK
    to plain deterministic_top_k(k).
    """
    candidates = deterministic_top_k(
        scores, k=candidate_k, score_col=score_col, id_col=id_col
    )
    if min_large_cap <= 0 or len(candidates) <= k:
        return candidates.head(k).reset_index(drop=True)

    mcap = mktcap[[id_col, "log_mktcap"]].drop_duplicates(id_col)
    threshold = float(mcap["log_mktcap"].quantile(large_cap_quantile))
    merged = candidates.merge(mcap, on=id_col, how="left")
    merged["is_large"] = merged["log_mktcap"].fillna(-np.inf) >= threshold

    large_in_candidates = int(merged["is_large"].sum())
    if large_in_candidates < min_large_cap:
        return candidates.head(k).reset_index(drop=True)

    top_k_slice = merged.head(k).copy()
    need = min_large_cap - int(top_k_slice["is_large"].sum())
    if need <= 0:
        return top_k_slice.drop(columns=["log_mktcap", "is_large"]).reset_index(drop=True)

    remainder = merged.iloc[k:]
    add_large = remainder[remainder["is_large"]].head(need)
    smallcaps_in_top = top_k_slice[~top_k_slice["is_large"]]
    drop_ids = smallcaps_in_top.tail(need)[id_col].tolist()
    kept = top_k_slice[~top_k_slice[id_col].isin(drop_ids)]
    out = pd.concat([kept, add_large], ignore_index=True)
    out = out.sort_values(by=["score_q", id_col], ascending=[False, True],
                           kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["log_mktcap", "is_large"])


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
