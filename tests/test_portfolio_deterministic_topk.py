import pandas as pd

from code.src.ensemble.portfolio import deterministic_top_k


def test_deterministic_top_k_uses_stock_id_for_quantized_ties():
    df = pd.DataFrame(
        {
            "stock_id": ["s3", "s1", "s2"],
            "score": [0.100041, 0.100049, 0.100045],
        }
    )

    out = deterministic_top_k(df, k=3)

    assert out["stock_id"].tolist() == ["s1", "s2", "s3"]
    assert out["score_q"].nunique() == 1


def test_deterministic_top_k_prefers_stronger_quantized_bucket():
    df = pd.DataFrame(
        {
            "stock_id": ["s9", "s1", "s2", "s3"],
            "score": [0.10014, 0.10024, 0.10012, 0.10025],
        }
    )

    out = deterministic_top_k(df, k=3)

    assert out["stock_id"].tolist()[:2] == ["s1", "s3"]
    assert out.loc[0, "score_q"] == out.loc[1, "score_q"]
    assert out.loc[0, "score_q"] > out.loc[2, "score_q"]
