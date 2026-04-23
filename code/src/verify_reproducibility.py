"""Verify Top-K reproducibility from saved result.csv artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def verify_topk_consistency(
    result_a: str | Path,
    result_b: str | Path,
    tol_weight: float = 0.01,
    min_intersection: int = 5,
) -> dict[str, Any]:
    a = pd.read_csv(result_a)
    b = pd.read_csv(result_b)

    set_a = set(a["stock_id"])
    set_b = set(b["stock_id"])
    merged = a.merge(b, on="stock_id", how="inner", suffixes=("_a", "_b"))
    intersection_size = len(set_a & set_b)
    max_wdiff = (
        float((merged["weight_a"] - merged["weight_b"]).abs().max())
        if not merged.empty
        else float("inf")
    )

    return {
        "pass": intersection_size >= min_intersection and max_wdiff <= tol_weight,
        "same_stocks": set_a == set_b,
        "intersection_size": intersection_size,
        "only_a": sorted(set_a - set_b),
        "only_b": sorted(set_b - set_a),
        "max_weight_diff": max_wdiff,
        "tol_weight": tol_weight,
        "min_intersection": min_intersection,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--single", nargs=2, metavar=("RUN1", "RUN2"))
    ap.add_argument("--cross", nargs=2, metavar=("MACHINE_A", "MACHINE_B"))
    ap.add_argument("--log", default="model/repro_check.log")
    ap.add_argument("--tol-weight", type=float, default=0.01)
    args = ap.parse_args()

    left, right = args.single or args.cross or (args.a, args.b)
    if not left or not right:
        raise SystemExit("must provide --single, --cross, or --a/--b")

    min_intersection = 5 if args.single else 4 if args.cross else 5
    result = verify_topk_consistency(
        left,
        right,
        tol_weight=args.tol_weight,
        min_intersection=min_intersection,
    )

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] {json.dumps(result, ensure_ascii=False)}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
