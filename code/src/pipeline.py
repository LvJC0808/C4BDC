"""Unified train/predict pipeline entrypoint (Task 14).

Subcommands:
    train    -> CV + refit + holdout ensemble search, writes ensemble_config.json
    predict  -> load refit models, score target date, write output/result.csv
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from . import config
    from .features.build import build_feature_sets
    from .cv.walk_forward import (
        generate_cv_splits, get_refit_epochs, filter_panel, CVPlan,
    )
    from .models.lgb_de import DoubleEnsembleModel
    from .models.master import MasterTrainer
    from .models.stockmixer import MixerTrainer
    from .ensemble.blender import (
        blend_scores, grid_search_weights, rank_normalize,
        blend_scores_dynamic, rolling_ic_weights,
    )
    from .ensemble.portfolio import (
        compute_confidence, build_portfolio, grid_search_portfolio_params,
    )
    from .ensemble.tradability import get_tradable_ids
except ImportError:  # script-style
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent.parent))
    from code.src import config  # type: ignore
    from code.src.features.build import build_feature_sets  # type: ignore
    from code.src.cv.walk_forward import (  # type: ignore
        generate_cv_splits, get_refit_epochs, filter_panel, CVPlan,
    )
    from code.src.models.lgb_de import DoubleEnsembleModel  # type: ignore
    from code.src.models.master import MasterTrainer  # type: ignore
    from code.src.models.stockmixer import MixerTrainer  # type: ignore
    from code.src.ensemble.blender import (  # type: ignore
        blend_scores, grid_search_weights, rank_normalize,
        blend_scores_dynamic, rolling_ic_weights,
    )
    from code.src.ensemble.portfolio import (  # type: ignore
        compute_confidence, build_portfolio, grid_search_portfolio_params,
    )
    from code.src.ensemble.tradability import get_tradable_ids  # type: ignore


MODEL_NAMES = ("lgb", "master", "mixer")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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
    for k in MODEL_NAMES:
        panel = feature_sets[k]["panel"]
        all_dates.update(pd.to_datetime(panel["datetime"]).unique())
    return pd.Series(sorted(all_dates))


def _slice(panel: pd.DataFrame, start, end) -> pd.DataFrame:
    panel = panel.copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    return filter_panel(panel, pd.Timestamp(start), pd.Timestamp(end))


def _best_epoch(trainer) -> Optional[int]:
    for attr in ("best_epoch_", "best_epoch"):
        v = getattr(trainer, attr, None)
        if isinstance(v, int) and v >= 0:
            return v
    return None


def _lgb_xy(panel: pd.DataFrame, feature_cols: List[str]):
    X = panel[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = panel["label"].to_numpy(dtype=np.float32) if "label" in panel.columns else None
    return X, y


# --------------------------------------------------------------------------- #
# Per-model fit/predict/save/load adapters
# --------------------------------------------------------------------------- #
def _fit_lgb(train_panel, val_panel, feature_cols, seed):
    X, y = _lgb_xy(train_panel, feature_cols)
    Xv, yv = _lgb_xy(val_panel, feature_cols) if val_panel is not None else (None, None)
    mdl = DoubleEnsembleModel(seed=seed)
    if Xv is None:
        # no val: slice last 10% as pseudo-val for early stopping
        n = len(X)
        cut = max(1, int(n * 0.9))
        mdl.fit(X[:cut], y[:cut], X[cut:], y[cut:])
    else:
        mdl.fit(X, y, Xv, yv)
    best_iter = mdl.submodels[0].best_iteration_ if mdl.submodels else 0
    return mdl, int(best_iter or 0)


def _fit_lgb_refit(train_panel, feature_cols, seed, num_rounds: int):
    X, y = _lgb_xy(train_panel, feature_cols)
    # Use 5% tail as val just so early_stopping callback has a set; num_rounds caps training.
    n = len(X)
    cut = max(1, int(n * 0.95))
    mdl = DoubleEnsembleModel(seed=seed, num_rounds=max(1, int(num_rounds)))
    mdl.fit(X[:cut], y[:cut], X[cut:], y[cut:])
    return mdl


def _save_model(name: str, model, path: str) -> None:
    _ensure_dirs(path)
    if name == "lgb":
        model.save(path)
    elif name == "master":
        model.save(path)
    elif name == "mixer":
        model.save(os.path.join(path, "mixer.pt"))


def _load_model(name: str, path: str):
    if name == "lgb":
        return DoubleEnsembleModel.load(path)
    if name == "master":
        return MasterTrainer().load(path)
    if name == "mixer":
        return MixerTrainer().load(os.path.join(path, "mixer.pt"))
    raise ValueError(name)


def _predict_scores(name: str, model, panel: pd.DataFrame, feature_cols: List[str],
                    market_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return DataFrame[instrument, datetime, score]."""
    if name == "lgb":
        X, _ = _lgb_xy(panel, feature_cols)
        s = model.predict(X)
        out = panel[["instrument", "datetime"]].copy()
        out["score"] = s
        return out
    if name == "master":
        return model.predict(panel, market_df, feature_cols)
    if name == "mixer":
        s = model.predict(panel, market_df, feature_cols)
        out = panel[["instrument", "datetime"]].copy()
        out["score"] = np.asarray(s)
        return out
    raise ValueError(name)


def _fit_trainer(name: str, feature_sets: dict, tr_panel, vl_panel, seed,
                 override_epochs: Optional[int] = None):
    if name == "master":
        cfg = dict(config.MASTER_CONFIG)
        if override_epochs is not None:
            cfg["epochs"] = int(override_epochs)
        t = MasterTrainer(config=cfg, seed=seed)
        t.fit(tr_panel, feature_sets["master"]["market"],
              feature_sets["master"]["feature_cols"],
              val_panel=vl_panel,
              val_market=feature_sets["master"]["market"] if vl_panel is not None else None)
        return t
    if name == "mixer":
        cfg = dict(config.MIXER_CONFIG)
        if override_epochs is not None:
            cfg["epochs"] = int(override_epochs)
        t = MixerTrainer(config=cfg)
        t.fit(tr_panel, feature_sets["mixer"].get("market"),
              feature_sets["mixer"]["feature_cols"],
              val_panel=vl_panel,
              val_market=feature_sets["mixer"].get("market") if vl_panel is not None else None)
        return t
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# TRAIN
# --------------------------------------------------------------------------- #
def cmd_train(args) -> None:
    set_global_seed(config.SEED)
    _ensure_dirs(args.model_dir, args.temp_dir)

    print(f"[train] building features from {args.data_path}")
    feature_sets = build_feature_sets(
        args.data_path, args.temp_dir, use_cache=True,
    )

    dates = _sorted_union_dates(feature_sets)
    plan: CVPlan = generate_cv_splits(
        dates,
        n_folds=config.CV_FOLDS,
        embargo=config.CV_EMBARGO,
        val_days=config.CV_VAL_DAYS,
        holdout_days=config.HOLDOUT_DAYS,
        fold_gap=config.CV_FOLD_GAP,
    )
    print(f"[train] CV plan: {len(plan.splits)} folds, holdout "
          f"{plan.holdout_start.date()}..{plan.holdout_end.date()}")

    best_iters: Dict[str, List[int]] = {n: [] for n in MODEL_NAMES}
    # Collect holdout predictions: per model, list[(seed, DataFrame)]
    holdout_preds: Dict[str, List[pd.DataFrame]] = {n: [] for n in MODEL_NAMES}

    # ---------------- CV ---------------- #
    for name in MODEL_NAMES:
        fset = feature_sets[name]
        panel = fset["panel"]
        fcols = fset["feature_cols"]
        mkt = fset.get("market")
        for split in plan.splits:
            tr = _slice(panel, split.train_start, split.train_end)
            vl = _slice(panel, split.val_start, split.val_end)
            for seed in config.SEEDS:
                set_global_seed(seed)
                print(f"[train] fit {name} seed={seed} fold={split.fold}")
                out_dir = os.path.join(args.model_dir, name, f"seed_{seed}_fold_{split.fold}")
                _ensure_dirs(out_dir)
                if name == "lgb":
                    mdl, bi = _fit_lgb(tr, vl, fcols, seed)
                    _save_model(name, mdl, out_dir)
                    best_iters[name].append(bi)
                    val_scores = _predict_scores(name, mdl, vl, fcols, mkt)
                else:
                    t = _fit_trainer(name, feature_sets, tr, vl, seed)
                    _save_model(name, t, out_dir)
                    be = _best_epoch(t)
                    if be is None:
                        be = config.MASTER_CONFIG["epochs"] if name == "master" else config.MIXER_CONFIG["epochs"]
                    best_iters[name].append(int(be))
                    val_scores = _predict_scores(name, t, vl, fcols, mkt)
                val_scores = val_scores.assign(fold=split.fold, seed=seed)
                val_scores.to_parquet(os.path.join(out_dir, "val_scores.parquet"))

    # ---------------- Refit on full training data ---------------- #
    refit_epochs: Dict[str, int] = {
        n: int(get_refit_epochs(best_iters[n])) for n in MODEL_NAMES
    }
    _REFIT_FLOOR = {"lgb": 200, "master": 20, "mixer": 30}
    for _n, _floor in _REFIT_FLOOR.items():
        if _n in refit_epochs and refit_epochs[_n] < _floor:
            print(f"[train] refit_epochs[{_n}]={refit_epochs[_n]} < floor {_floor}, raising")
            refit_epochs[_n] = _floor
    print(f"[train] refit epochs: {refit_epochs}")

    # Full train range = [earliest, holdout_start - 1 - embargo]
    full_start = dates.iloc[0]
    embargo_idx = max(0, len(dates) - config.HOLDOUT_DAYS - config.CV_EMBARGO - 1)
    full_end = dates.iloc[embargo_idx]

    for name in MODEL_NAMES:
        fset = feature_sets[name]
        panel = fset["panel"]
        fcols = fset["feature_cols"]
        mkt = fset.get("market")
        tr = _slice(panel, full_start, full_end)
        holdout = _slice(panel, plan.holdout_start, plan.holdout_end)
        for seed in config.SEEDS:
            set_global_seed(seed)
            out_dir = os.path.join(args.model_dir, name, f"seed_{seed}_refit")
            _ensure_dirs(out_dir)
            print(f"[train] refit {name} seed={seed}")
            if name == "lgb":
                mdl = _fit_lgb_refit(tr, fcols, seed, num_rounds=refit_epochs[name])
                _save_model(name, mdl, out_dir)
                preds = _predict_scores(name, mdl, holdout, fcols, mkt)
            else:
                t = _fit_trainer(name, feature_sets, tr, None, seed,
                                 override_epochs=refit_epochs[name])
                _save_model(name, t, out_dir)
                preds = _predict_scores(name, t, holdout, fcols, mkt)
            preds = preds.assign(seed=seed)
            holdout_preds[name].append(preds)

    # ---------------- Holdout ensemble grid search ---------------- #
    model_scores_mean: Dict[str, pd.DataFrame] = {}
    for name in MODEL_NAMES:
        df = pd.concat(holdout_preds[name], ignore_index=True)
        agg = (df.groupby(["instrument", "datetime"])["score"]
                 .mean().reset_index())
        model_scores_mean[name] = agg

    # Labels from any panel (use lgb; labels match across)
    lgb_panel = feature_sets["lgb"]["panel"].copy()
    lgb_panel["datetime"] = pd.to_datetime(lgb_panel["datetime"])
    ho_labels = lgb_panel[(lgb_panel["datetime"] >= plan.holdout_start) &
                          (lgb_panel["datetime"] <= plan.holdout_end)][
        ["instrument", "datetime", "label"]
    ].copy()

    weight_best = grid_search_weights(model_scores_mean, ho_labels,
                                      top_k_list=config.TOP_K_CANDIDATES)
    print(f"[train] best weights: {weight_best}")

    blended = blend_scores(model_scores_mean, weight_best["weights"])
    blended = blended.rename(columns={"final_score": "final_score"})
    port_best = grid_search_portfolio_params(
        blended, ho_labels,
        top_k_candidates=config.TOP_K_CANDIDATES,
        alpha_candidates=config.POSITION_ALPHAS,
    )
    print(f"[train] best portfolio params: {port_best}")

    ensemble_cfg = {
        "weights": weight_best["weights"],
        "top_k": int(port_best["top_k"] or weight_best["top_k"]),
        "alpha": float(port_best["alpha"] or config.POSITION_ALPHAS[0]),
        "refit_epochs": refit_epochs,
        "feature_version": "v1.0",
        "seeds": list(config.SEEDS),
        "cv_folds": int(config.CV_FOLDS),
    }
    cfg_path = os.path.join(args.model_dir, "ensemble_config.json")
    with open(cfg_path, "w") as f:
        json.dump(ensemble_cfg, f, indent=2)
    print(f"[train] wrote {cfg_path}")


# --------------------------------------------------------------------------- #
# PREDICT
# --------------------------------------------------------------------------- #
def _rolling_ic(blended_panel: pd.DataFrame, labels: pd.DataFrame,
                target_date, window: int = 20) -> float:
    from scipy.stats import spearmanr
    merged = blended_panel.merge(labels, on=["instrument", "datetime"], how="inner")
    merged = merged[merged["datetime"] < pd.Timestamp(target_date)]
    if merged.empty:
        return 0.0
    uniq = sorted(merged["datetime"].unique())[-window:]
    ics = []
    for d in uniq:
        g = merged[merged["datetime"] == d]
        if len(g) < 3:
            continue
        rho, _ = spearmanr(g["final_score"], g["label"])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else 0.0


def cmd_predict(args) -> None:
    set_global_seed(config.SEED)
    _ensure_dirs(args.temp_dir, os.path.dirname(args.output_path) or ".")

    cfg_path = os.path.join(args.model_dir, "ensemble_config.json")
    with open(cfg_path) as f:
        ens_cfg = json.load(f)
    weights = ens_cfg["weights"]
    top_k = int(ens_cfg["top_k"])
    alpha = float(ens_cfg["alpha"])
    blend_mode = str(ens_cfg.get("blend_mode", "static"))
    rolling_window = int(ens_cfg.get("rolling_window", 20))
    rolling_beta = float(ens_cfg.get("rolling_beta", 1.0))

    feature_sets = build_feature_sets(
        args.data_path, args.temp_dir, use_cache=True,
    )

    dates = _sorted_union_dates(feature_sets)
    target_date = pd.Timestamp(args.date) if getattr(args, "date", None) else dates.iloc[-1]
    print(f"[predict] target date = {target_date.date()}")

    # Per-model scoring, averaged across seeds
    per_model_today: Dict[str, pd.DataFrame] = {}
    per_model_topk: List[set] = []
    # For rolling IC we need historical predictions too; use a tail window
    hist_window_start = dates.iloc[max(0, len(dates) - 80)]
    per_model_hist: Dict[str, pd.DataFrame] = {}

    for name in MODEL_NAMES:
        fset = feature_sets[name]
        panel = fset["panel"].copy()
        panel["datetime"] = pd.to_datetime(panel["datetime"])
        fcols = fset["feature_cols"]
        mkt = fset.get("market")
        slice_panel = panel[(panel["datetime"] >= hist_window_start) &
                            (panel["datetime"] <= target_date)]
        if slice_panel.empty:
            continue
        seed_scores: List[pd.DataFrame] = []
        for seed in config.SEEDS:
            mdir = os.path.join(args.model_dir, name, f"seed_{seed}_refit")
            if not os.path.exists(mdir):
                continue
            mdl = _load_model(name, mdir)
            s = _predict_scores(name, mdl, slice_panel, fcols, mkt)
            seed_scores.append(s)
        if not seed_scores:
            continue
        concat = pd.concat(seed_scores, ignore_index=True)
        agg = (concat.groupby(["instrument", "datetime"])["score"]
                     .mean().reset_index())
        per_model_hist[name] = agg
        today = agg[agg["datetime"] == target_date].copy()
        per_model_today[name] = today
        if not today.empty:
            top = set(today.nlargest(top_k, "score")["instrument"].tolist())
            per_model_topk.append(top)

    if not per_model_today:
        raise RuntimeError("No per-model predictions generated.")

    # Blend today (+ history for IC)
    lgb_panel = feature_sets["lgb"]["panel"].copy()
    lgb_panel["datetime"] = pd.to_datetime(lgb_panel["datetime"])
    labels = lgb_panel[["instrument", "datetime", "label"]]

    if blend_mode == "dynamic":
        weight_df = rolling_ic_weights(
            per_model_hist, labels,
            window=rolling_window, beta=rolling_beta,
        )
        blended_today = blend_scores_dynamic(per_model_today, weight_df)
        blended_hist = blend_scores_dynamic(per_model_hist, weight_df)
    else:
        blended_today = blend_scores(per_model_today, weights)
        blended_hist = blend_scores(per_model_hist, weights)

    # Rolling IC vs labels
    roll_ic = _rolling_ic(blended_hist, labels, target_date, window=20)
    print(f"[predict] rolling_ic = {roll_ic:.4f}")

    today_df = blended_today[blended_today["datetime"] == target_date].copy()
    scores_series = today_df.set_index("instrument")["final_score"]
    confidence = compute_confidence(scores_series, per_model_topk, roll_ic, top_k=top_k)
    print(f"[predict] confidence = {confidence:.3f}")

    raw_stock = pd.read_csv(os.path.join(args.data_path, "stock_data.csv"))
    tradable = get_tradable_ids(raw_stock, target_date)
    print(f"[predict] tradable count = {len(tradable)}")

    portfolio = build_portfolio(today_df, confidence, alpha=alpha, top_k=top_k,
                                min_position=config.MIN_POSITION,
                                tradable_ids=tradable)
    _ensure_dirs(os.path.dirname(args.output_path) or ".")
    portfolio[["stock_id", "weight"]].to_csv(args.output_path, index=False)
    print(f"[predict] wrote {args.output_path} ({len(portfolio)} rows)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train CV + refit + ensemble search")
    pt.add_argument("--data_path", default=config.DATA_PATH)
    pt.add_argument("--model_dir", default=config.MODEL_DIR)
    pt.add_argument("--temp_dir", default=config.TEMP_DIR)
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="predict and write result.csv")
    pp.add_argument("--data_path", default=config.DATA_PATH)
    pp.add_argument("--model_dir", default=config.MODEL_DIR)
    pp.add_argument("--temp_dir", default=config.TEMP_DIR)
    pp.add_argument("--output_path",
                    default=os.path.join(config.OUTPUT_DIR, "result.csv"))
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
