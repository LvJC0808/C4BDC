"""Offline recompute ICIR shrinkage weights from existing val_scores.

用法：
    .venv/bin/python scripts/recompute_icir_weights.py \
        --model_dir ./model \
        --data_path ./data \
        --temp_dir  ./temp \
        --out       ./model/ensemble_config.json

只读取 model/<name>/seed_*_fold_*/val_scores.parquet，不重训模型。
产出新的 ensemble_config.json，其中 weights 走 icir_shrink 路径。
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

from code.src import config
from code.src.features.build import build_feature_sets
from code.src.cv.walk_forward import generate_cv_splits
from code.src.ensemble.blender import (
    rank_normalize_daily, optimize_weights_icir,
    select_lambda_loo, bootstrap_weights, blend_scores,
)
from code.src.ensemble.portfolio import grid_search_portfolio_params

MODEL_NAMES = ("lgb", "master", "mixer")


def _load_val(model_dir: str, name: str, seeds, folds: int):
    """返回 {fold_idx: DataFrame(instrument,datetime,score)} ，seed 间做平均."""
    out = {}
    for fold in range(folds):
        parts = []
        for seed in seeds:
            p = os.path.join(model_dir, name, f"seed_{seed}_fold_{fold}", "val_scores.parquet")
            if os.path.exists(p):
                df = pd.read_parquet(p)[["instrument", "datetime", "score"]]
                parts.append(df)
            else:
                print(f"  [miss] {p}")
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        out[fold] = df.groupby(["instrument", "datetime"], as_index=False)["score"].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./model")
    ap.add_argument("--data_path", default="./data")
    ap.add_argument("--temp_dir", default="./temp")
    ap.add_argument("--out", default="./model/ensemble_config.json")
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "42,2024"))
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"[icir] seeds={seeds}  model_dir={args.model_dir}")

    # 1) Rebuild feature panel (cached) to get labels
    print("[icir] building features (cached)…")
    feature_sets = build_feature_sets(args.data_path, args.temp_dir)
    panel = feature_sets["lgb"]["panel"][["instrument", "datetime", "label"]].copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])

    # 2) CV plan (same rule as training)
    dates = pd.Series(sorted(panel["datetime"].unique()))
    plan = generate_cv_splits(dates, n_folds=config.CV_FOLDS,
                              holdout_days=config.HOLDOUT_DAYS,
                              embargo=config.CV_EMBARGO)
    print(f"[icir] CV folds={len(plan.splits)}  holdout={plan.holdout_start.date()}..{plan.holdout_end.date()}")

    # 3) Load val_scores per fold per model
    scores_by_fold = []
    for split in plan.splits:
        fold_dict = {}
        for name in MODEL_NAMES:
            df_all = _load_val(args.model_dir, name, seeds, folds=len(plan.splits))
            if split.fold in df_all:
                fold_dict[name] = rank_normalize_daily(df_all[split.fold])
        scores_by_fold.append(fold_dict)

    # Holdout: re-use last fold's val as pseudo-holdout? Actually holdout preds should come from refit.
    # For pure CV-ICIR it's fine to skip holdout fold; here we just use the 3 CV folds.

    all_scores = {
        m: pd.concat([f[m] for f in scores_by_fold if m in f], ignore_index=True)
        for m in MODEL_NAMES
    }
    for m, df in all_scores.items():
        print(f"  [{m}] rows={len(df)}")

    # 4) λ selection via LOO-CV
    print("[icir] selecting lambda via LOO over CV folds…")
    lam_res = select_lambda_loo(
        scores_by_fold, panel,
        lam_grid=config.ICIR_LAMBDA_GRID,
        model_order=list(MODEL_NAMES),
        seed=config.SEED,
    )
    print(f"[icir] lambda*={lam_res['lambda']}  mean_icir={lam_res['mean_icir']:.4f}")
    print(f"        all: {lam_res['all']}")

    # 5) Optimize weights on all folds
    opt_res = optimize_weights_icir(
        all_scores, panel, lam=lam_res["lambda"],
        model_order=list(MODEL_NAMES), seed=config.SEED,
    )
    weights = opt_res["weights"]
    print(f"[icir] weights={weights}  icir={opt_res['icir']:.4f}")

    # 6) Bootstrap CI
    boot = bootstrap_weights(
        all_scores, panel, lam=lam_res["lambda"],
        model_order=list(MODEL_NAMES),
        n=int(os.environ.get("ICIR_BOOTSTRAP_N", "100")),
        seed=config.SEED,
    )
    print(f"[icir] bootstrap CI_low={boot['ci_low']}  CI_high={boot['ci_high']}")

    # 7) Find best top_k/alpha on the last fold treated as holdout-ish
    last_fold = scores_by_fold[-1]
    blended_last = blend_scores(last_fold, weights)
    port = grid_search_portfolio_params(
        blended_last, panel,
        top_k_candidates=config.TOP_K_CANDIDATES,
        alpha_candidates=config.POSITION_ALPHAS,
    )
    print(f"[icir] best portfolio: top_k={port['top_k']} alpha={port['alpha']} score={port['score']:.4f}")

    # 8) Write new ensemble_config.json (preserve existing refit_epochs if any)
    existing = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)

    cfg = {
        "method": "icir_shrink",
        "weights": {k: float(v) for k, v in weights.items()},
        "legacy_weights": existing.get("weights"),
        "lambda": float(lam_res["lambda"]),
        "icir": float(opt_res["icir"]),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "top_k": int(port["top_k"] or config.TOP_K_CANDIDATES[0]),
        "alpha": float(port["alpha"] or config.POSITION_ALPHAS[0]),
        "refit_epochs": existing.get("refit_epochs", {}),
        "feature_version": "v1.0",
        "seeds": seeds,
        "cv_folds": int(config.CV_FOLDS),
    }
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[icir] wrote {args.out}")


if __name__ == "__main__":
    main()
