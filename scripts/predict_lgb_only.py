"""LGB-only prediction entrypoint."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from code.src import config
from code.src.ensemble.portfolio import deterministic_top_k
from code.src.features.build import build_feature_sets
from code.src.pipeline import _ensure_dirs, _load_model, _predict_scores, _sorted_union_dates


DATA = "./data"
MODEL_DIR = "./model_lgb_only"
TEMP = "./temp"
OUTPUT_PATH = "./output/result.csv"


def main() -> None:
    _ensure_dirs(TEMP, os.path.dirname(OUTPUT_PATH) or ".")

    cfg_path = os.path.join(MODEL_DIR, "ensemble_config.json")
    with open(cfg_path) as f:
        ens_cfg = json.load(f)

    top_k = int(ens_cfg.get("top_k", 5))
    seeds = [int(seed) for seed in ens_cfg.get("seeds", config.SEEDS)]

    print("[lgb-only] build features")
    feature_sets = build_feature_sets(DATA, TEMP, use_cache=True)
    fset = feature_sets["lgb"]
    panel = fset["panel"].copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    fcols = fset["feature_cols"]

    dates = _sorted_union_dates({"lgb": fset})
    target_date = pd.Timestamp(dates.iloc[-1])
    today_panel = panel[panel["datetime"] == target_date].copy()
    if today_panel.empty:
        raise RuntimeError(f"No LGB rows found for target date {target_date.date()}")
    print(f"[lgb-only] target date = {target_date.date()}")

    seed_scores = []
    for seed in seeds:
        mdir = os.path.join(MODEL_DIR, "lgb", f"seed_{seed}_refit")
        if not os.path.exists(mdir):
            print(f"[lgb-only] skip missing model {mdir}")
            continue
        mdl = _load_model("lgb", mdir)
        scored = _predict_scores("lgb", mdl, today_panel, fcols, market_df=None)
        seed_scores.append(scored)

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

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"[lgb-only] wrote {OUTPUT_PATH} ({len(result)} rows)")


if __name__ == "__main__":
    main()
