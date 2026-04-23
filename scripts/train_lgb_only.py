"""Path B: LGB-only 快速训练 + ensemble_config 生成。

复用 pipeline 的 CV 计划、特征构建、DoubleEnsemble 模型。只跑 lgb，权重固定 {lgb:1.0, master:0.0, mixer:0.0}，生成 model_lgb_only/ 目录。
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from code.src import config
from code.src.features.build import build_feature_sets
from code.src.cv.walk_forward import generate_cv_splits, get_refit_epochs, filter_panel
from code.src.models.lgb_de import DoubleEnsembleModel
from code.src.pipeline import (
    set_global_seed, _ensure_dirs, _sorted_union_dates, _slice,
    _fit_lgb, _fit_lgb_refit, _lgb_xy,
)

DATA = "./data"
MODEL_DIR = "./model_lgb_only"
TEMP = "./temp"

def main():
    set_global_seed(config.SEED)
    _ensure_dirs(MODEL_DIR, TEMP)

    print("[lgb-only] build features")
    fsets = build_feature_sets(DATA, TEMP, use_cache=True)

    dates = _sorted_union_dates(fsets)
    plan = generate_cv_splits(dates, n_folds=config.CV_FOLDS, embargo=config.CV_EMBARGO,
                              val_days=config.CV_VAL_DAYS, holdout_days=config.HOLDOUT_DAYS,
                              fold_gap=config.CV_FOLD_GAP)
    print(f"[lgb-only] holdout {plan.holdout_start.date()}..{plan.holdout_end.date()}")

    fset = fsets["lgb"]
    panel = fset["panel"]
    fcols = fset["feature_cols"]

    best_iters = []
    for split in plan.splits:
        tr = _slice(panel, split.train_start, split.train_end)
        vl = _slice(panel, split.val_start, split.val_end)
        for seed in config.SEEDS:
            set_global_seed(seed)
            out = os.path.join(MODEL_DIR, "lgb", f"seed_{seed}_fold_{split.fold}")
            _ensure_dirs(out)
            print(f"[lgb-only] fit seed={seed} fold={split.fold}")
            mdl, bi = _fit_lgb(tr, vl, fcols, seed)
            mdl.save(out)
            best_iters.append(bi)

    refit_epochs = int(get_refit_epochs(best_iters))
    print(f"[lgb-only] refit_rounds={refit_epochs}")

    full_start = dates.iloc[0]
    embargo_idx = max(0, len(dates) - config.HOLDOUT_DAYS - config.CV_EMBARGO - 1)
    full_end = dates.iloc[embargo_idx]
    tr = _slice(panel, full_start, full_end)
    for seed in config.SEEDS:
        set_global_seed(seed)
        out = os.path.join(MODEL_DIR, "lgb", f"seed_{seed}_refit")
        _ensure_dirs(out)
        print(f"[lgb-only] refit seed={seed}")
        mdl = _fit_lgb_refit(tr, fcols, seed, num_rounds=refit_epochs)
        mdl.save(out)

    cfg = {
        "method": "lgb_only",
        "weights": {"lgb": 1.0, "master": 0.0, "mixer": 0.0},
        "legacy_weights": {"lgb": 1.0, "master": 0.0, "mixer": 0.0},
        "top_k": 5,
        "alpha": 0.7,
        "refit_epochs": {"lgb": refit_epochs, "master": 0, "mixer": 0},
        "seeds": list(config.SEEDS),
        "cv_folds": int(config.CV_FOLDS),
    }
    with open(os.path.join(MODEL_DIR, "ensemble_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[lgb-only] wrote {MODEL_DIR}/ensemble_config.json")

if __name__ == "__main__":
    main()
