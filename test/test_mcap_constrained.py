import pandas as pd
import pytest
from code.src.ensemble.portfolio import mcap_constrained_topk


def _make_scores(pairs):
    """pairs: list of (stock_id, score)."""
    return pd.DataFrame(pairs, columns=["stock_id", "score"])


def _make_mktcap(pairs):
    """pairs: list of (stock_id, log_mktcap)."""
    return pd.DataFrame(pairs, columns=["stock_id", "log_mktcap"])


def test_min_large_cap_zero_equivalent_to_top5():
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", float(i)) for i in range(10)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=0)
    assert out["stock_id"].tolist() == [f"{i:06d}" for i in range(5)]
    assert len(out) == 5


def test_all_large_cap_returns_top5_unchanged():
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", 20.0) for i in range(10)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2)
    assert out["stock_id"].tolist() == [f"{i:06d}" for i in range(5)]


def test_only_one_large_cap_triggers_fallback():
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", 1.0) for i in range(9)]
                           + [("000009", 20.0)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2)
    assert out["stock_id"].tolist() == [f"{i:06d}" for i in range(5)]


def test_two_large_caps_at_tail_replace_smallcaps():
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", 1.0) for i in range(8)]
                           + [("000008", 20.0), ("000009", 20.0)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2,
                                 large_cap_quantile=0.85)
    ids = out["stock_id"].tolist()
    assert ids[:3] == ["000000", "000001", "000002"]
    assert set(ids[3:]) == {"000008", "000009"}
    assert len(out) == 5


def test_output_length_and_order():
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", 20.0) for i in range(5)]
                           + [(f"{i:06d}", 1.0) for i in range(5, 10)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2)
    assert len(out) == 5
    scores_out = out["score_q"].tolist()
    assert scores_out == sorted(scores_out, reverse=True)
