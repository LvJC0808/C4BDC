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
from code.src.ensemble.portfolio import deterministic_top_k, mcap_constrained_topk
from code.src.ensemble.regime import detect_regime
from code.src.ensemble.allocation import allocate_by_mode
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
    if os.environ.get("TARGET_DATE"):
        target_date = pd.Timestamp(os.environ["TARGET_DATE"])
    else:
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
    )[["stock_id", "score_q"]].copy()

    mcap_min_large = int(os.environ.get("MCAP_MIN_LARGE", "0"))
    mcap_cand_k = int(os.environ.get("MCAP_CAND_K", "10"))
    mcap_large_q = float(os.environ.get("MCAP_LARGE_Q", "0.5"))
    if mcap_min_large > 0 and "log_mktcap" in today_panel.columns:
        scores_for_pick = avg_scores.rename(columns={"instrument": "stock_id"})
        mktcap_snap = today_panel[["instrument", "log_mktcap"]].rename(
            columns={"instrument": "stock_id"}
        )
        picks = mcap_constrained_topk(
            scores_for_pick, mktcap_snap,
            k=top_k, candidate_k=mcap_cand_k,
            min_large_cap=mcap_min_large,
            large_cap_quantile=mcap_large_q,
        )[["stock_id", "score_q"]].copy()
        print(f"[lgb-only] mcap_constrained Top-{top_k} "
              f"(min_large={mcap_min_large}, cand_k={mcap_cand_k}, q={mcap_large_q})",
              file=sys.stderr)

    try:
        close_col = "close" if "close" in panel.columns else (
            "$close" if "$close" in panel.columns else None
        )
        if close_col and "log_mktcap" in panel.columns:
            p_slim = panel[["datetime", "instrument", close_col, "log_mktcap"]].rename(
                columns={close_col: "close"}
            )
            reg = detect_regime(p_slim, as_of=target_date)
            print(f"[lgb-only] regime={reg['regime']} diff={reg['diff']:+.4f} "
                  f"r_large={reg['r_large']:+.4f} r_small={reg['r_small']:+.4f}",
                  file=sys.stderr)
        else:
            print("[lgb-only] regime=unavailable (no close/log_mktcap col)", file=sys.stderr)
    except Exception as e:
        print(f"[lgb-only] regime detection failed: {e}", file=sys.stderr)

    if picks.empty:
        result = pd.DataFrame(columns=["stock_id", "weight"])
    else:
        alloc_mode = os.environ.get("ALLOC_MODE", "equal")
        result = allocate_by_mode(picks, alloc_mode)[["stock_id", "weight"]]

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"[lgb-only] wrote {OUTPUT_PATH} ({len(result)} rows)")


if __name__ == "__main__":
    main()
