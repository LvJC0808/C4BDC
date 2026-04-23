"""Unified LGB-only train/predict pipeline entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from . import config
    from .cv.walk_forward import CVPlan, filter_panel, generate_cv_splits, get_refit_epochs
    from .ensemble.portfolio import deterministic_top_k
    from .features.build import build_feature_sets
except ImportError:  # script-style
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent.parent))
    from code.src import config  # type: ignore
    from code.src.cv.walk_forward import (  # type: ignore
        CVPlan,
        filter_panel,
        generate_cv_splits,
        get_refit_epochs,
    )
    from code.src.ensemble.portfolio import deterministic_top_k  # type: ignore
    from code.src.features.build import build_feature_sets  # type: ignore


MODEL_NAMES = ("lgb",)


def _double_ensemble_model_cls():
    try:
        from .models.lgb_de import DoubleEnsembleModel
    except ImportError:  # script-style
        from code.src.models.lgb_de import DoubleEnsembleModel  # type: ignore
    return DoubleEnsembleModel


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def _sorted_union_dates(feature_sets: dict) -> pd.Series:
    all_dates = set()
    for name in MODEL_NAMES:
        panel = feature_sets[name]["panel"]
        all_dates.update(pd.to_datetime(panel["datetime"]).unique())
    return pd.Series(sorted(all_dates))


def _slice(panel: pd.DataFrame, start, end) -> pd.DataFrame:
    panel = panel.copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    return filter_panel(panel, pd.Timestamp(start), pd.Timestamp(end))


def _lgb_xy(panel: pd.DataFrame, feature_cols: List[str]):
    X = panel[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = panel["label"].to_numpy(dtype=np.float32) if "label" in panel.columns else None
    return X, y


def _fit_lgb(train_panel, val_panel, feature_cols, seed):
    DoubleEnsembleModel = _double_ensemble_model_cls()
    X, y = _lgb_xy(train_panel, feature_cols)
    Xv, yv = _lgb_xy(val_panel, feature_cols) if val_panel is not None else (None, None)
    mdl = DoubleEnsembleModel(seed=seed)
    if Xv is None:
        n = len(X)
        cut = max(1, int(n * 0.9))
        mdl.fit(X[:cut], y[:cut], X[cut:], y[cut:])
    else:
        mdl.fit(X, y, Xv, yv)
    best_iter = mdl.submodels[0].best_iteration_ if mdl.submodels else 0
    return mdl, int(best_iter or 0)


def _fit_lgb_refit(train_panel, feature_cols, seed, num_rounds: int):
    DoubleEnsembleModel = _double_ensemble_model_cls()
    X, y = _lgb_xy(train_panel, feature_cols)
    n = len(X)
    cut = max(1, int(n * 0.95))
    mdl = DoubleEnsembleModel(seed=seed, num_rounds=max(1, int(num_rounds)))
    mdl.fit(X[:cut], y[:cut], X[cut:], y[cut:])
    return mdl


def _save_model(name: str, model, path: str) -> None:
    if name != "lgb":
        raise ValueError(f"Unsupported model for mainline pipeline: {name}")
    _ensure_dirs(path)
    model.save(path)


def _load_model(name: str, path: str):
    if name != "lgb":
        raise ValueError(f"Unsupported model for mainline pipeline: {name}")
    DoubleEnsembleModel = _double_ensemble_model_cls()
    return DoubleEnsembleModel.load(path)


def _predict_scores(name: str, model, panel: pd.DataFrame, feature_cols: List[str], market_df=None) -> pd.DataFrame:
    del market_df
    if name != "lgb":
        raise ValueError(f"Unsupported model for mainline pipeline: {name}")
    X, _ = _lgb_xy(panel, feature_cols)
    out = panel[["instrument", "datetime"]].copy()
    out["score"] = model.predict(X)
    return out


def cmd_train(args) -> None:
    set_global_seed(config.SEED)
    _ensure_dirs(args.model_dir, args.temp_dir)

    print(f"[train] building features from {args.data_path}")
    feature_sets = build_feature_sets(args.data_path, args.temp_dir, use_cache=True)
    fset = feature_sets["lgb"]
    panel = fset["panel"]
    feature_cols = fset["feature_cols"]

    dates = _sorted_union_dates({"lgb": fset})
    plan: CVPlan = generate_cv_splits(
        dates,
        n_folds=config.CV_FOLDS,
        embargo=config.CV_EMBARGO,
        val_days=config.CV_VAL_DAYS,
        holdout_days=config.HOLDOUT_DAYS,
        fold_gap=config.CV_FOLD_GAP,
    )
    print(
        f"[train] CV plan: {len(plan.splits)} folds, holdout "
        f"{plan.holdout_start.date()}..{plan.holdout_end.date()}"
    )

    best_iters: List[int] = []
    for split in plan.splits:
        tr = _slice(panel, split.train_start, split.train_end)
        vl = _slice(panel, split.val_start, split.val_end)
        for seed in config.SEEDS:
            set_global_seed(seed)
            out_dir = os.path.join(args.model_dir, "lgb", f"seed_{seed}_fold_{split.fold}")
            _ensure_dirs(out_dir)
            print(f"[train] fit lgb seed={seed} fold={split.fold}")
            mdl, best_iter = _fit_lgb(tr, vl, feature_cols, seed)
            _save_model("lgb", mdl, out_dir)
            best_iters.append(best_iter)
            val_scores = _predict_scores("lgb", mdl, vl, feature_cols)
            val_scores.assign(fold=split.fold, seed=seed).to_parquet(
                os.path.join(out_dir, "val_scores.parquet")
            )

    refit_rounds = max(200, int(get_refit_epochs(best_iters)))
    print(f"[train] refit_rounds={refit_rounds}")

    full_start = dates.iloc[0]
    embargo_idx = max(0, len(dates) - config.HOLDOUT_DAYS - config.CV_EMBARGO - 1)
    full_end = dates.iloc[embargo_idx]
    full_train = _slice(panel, full_start, full_end)

    for seed in config.SEEDS:
        set_global_seed(seed)
        out_dir = os.path.join(args.model_dir, "lgb", f"seed_{seed}_refit")
        _ensure_dirs(out_dir)
        print(f"[train] refit lgb seed={seed}")
        mdl = _fit_lgb_refit(full_train, feature_cols, seed, num_rounds=refit_rounds)
        _save_model("lgb", mdl, out_dir)

    ensemble_cfg = {
        "method": "lgb_only",
        "weights": {"lgb": 1.0},
        "top_k": 5,
        "alpha": 1.0,
        "refit_epochs": {"lgb": refit_rounds},
        "feature_version": "v1.0",
        "seeds": list(config.SEEDS),
        "cv_folds": int(config.CV_FOLDS),
    }
    cfg_path = os.path.join(args.model_dir, "ensemble_config.json")
    with open(cfg_path, "w") as f:
        json.dump(ensemble_cfg, f, indent=2)
    print(f"[train] wrote {cfg_path}")


def cmd_predict(args) -> None:
    set_global_seed(config.SEED)
    _ensure_dirs(args.temp_dir, os.path.dirname(args.output_path) or ".")

    cfg_path = os.path.join(args.model_dir, "ensemble_config.json")
    with open(cfg_path) as f:
        ens_cfg = json.load(f)

    top_k = int(ens_cfg.get("top_k", 5))
    seeds = [int(seed) for seed in ens_cfg.get("seeds", config.SEEDS)]

    feature_sets = build_feature_sets(args.data_path, args.temp_dir, use_cache=True)
    fset = feature_sets["lgb"]
    panel = fset["panel"].copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    feature_cols = fset["feature_cols"]

    dates = _sorted_union_dates({"lgb": fset})
    target_date = pd.Timestamp(args.date) if getattr(args, "date", None) else pd.Timestamp(dates.iloc[-1])
    print(f"[predict] target date = {target_date.date()}")

    today_panel = panel[panel["datetime"] == target_date].copy()
    if today_panel.empty:
        raise RuntimeError(f"No LGB rows found for target date {target_date.date()}")

    seed_scores: List[pd.DataFrame] = []
    for seed in seeds:
        model_dir = os.path.join(args.model_dir, "lgb", f"seed_{seed}_refit")
        if not os.path.exists(model_dir):
            print(f"[predict] skip missing model {model_dir}")
            continue
        mdl = _load_model("lgb", model_dir)
        seed_scores.append(_predict_scores("lgb", mdl, today_panel, feature_cols))

    if not seed_scores:
        raise RuntimeError("No LGB refit models found for prediction.")

    avg_scores = (
        pd.concat(seed_scores, ignore_index=True)
        .groupby(["instrument", "datetime"], as_index=False)["score"]
        .mean()
    )
    picks = deterministic_top_k(
        avg_scores.rename(columns={"instrument": "stock_id"}),
        k=top_k,
        score_col="score",
        id_col="stock_id",
    )[["stock_id"]].copy()

    if picks.empty:
        result = pd.DataFrame(columns=["stock_id", "weight"])
    else:
        picks["weight"] = 1.0 / len(picks)
        result = picks

    _ensure_dirs(os.path.dirname(args.output_path) or ".")
    result.to_csv(args.output_path, index=False)
    print(f"[predict] wrote {args.output_path} ({len(result)} rows)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train CV + refit")
    pt.add_argument("--data_path", default=config.DATA_PATH)
    pt.add_argument("--model_dir", default=config.MODEL_DIR)
    pt.add_argument("--temp_dir", default=config.TEMP_DIR)
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="predict and write result.csv")
    pp.add_argument("--data_path", default=config.DATA_PATH)
    pp.add_argument("--model_dir", default=config.MODEL_DIR)
    pp.add_argument("--temp_dir", default=config.TEMP_DIR)
    pp.add_argument("--output_path", default=os.path.join(config.OUTPUT_DIR, "result.csv"))
    pp.add_argument("--date", default=None, help="override target date YYYY-MM-DD")
    pp.set_defaults(func=cmd_predict)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
