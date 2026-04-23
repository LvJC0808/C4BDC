from pathlib import Path

import pandas as pd

from code.src.verify_reproducibility import verify_topk_consistency


def _write_result(path: Path, rows) -> None:
    pd.DataFrame(rows, columns=["stock_id", "weight"]).to_csv(path, index=False)


def test_verify_topk_consistency_accepts_small_weight_noise(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_result(
        a,
        [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)],
    )
    _write_result(
        b,
        [("000001", 0.201), ("000002", 0.199), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)],
    )

    result = verify_topk_consistency(a, b, tol_weight=0.01, min_intersection=5)

    assert result["pass"] is True
    assert result["same_stocks"] is True
    assert result["intersection_size"] == 5


def test_verify_topk_consistency_accepts_cross_machine_overlap_four_of_five(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_result(
        a,
        [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)],
    )
    _write_result(
        b,
        [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000006", 0.2)],
    )

    result = verify_topk_consistency(a, b, tol_weight=0.01, min_intersection=4)

    assert result["pass"] is True
    assert result["same_stocks"] is False
    assert result["intersection_size"] == 4


def test_verify_topk_consistency_rejects_insufficient_overlap(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_result(
        a,
        [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)],
    )
    _write_result(
        b,
        [("000001", 0.2), ("000002", 0.2), ("000006", 0.2), ("000007", 0.2), ("000008", 0.2)],
    )

    result = verify_topk_consistency(a, b, tol_weight=0.01, min_intersection=4)

    assert result["pass"] is False
    assert result["intersection_size"] == 2
