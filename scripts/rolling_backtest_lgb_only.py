"""Path B: LGB-only rolling backtest wrapper with configurable allocation.

Set env var ALLOC_MODE=equal|linear|linear_floor15|softmax_t03|softmax_t10 to
override the default equal-weight allocation produced by build_portfolio.

Monkey-patch MODEL_NAMES to ("lgb",), then invoke rolling_backtest.main().
All argparse args forwarded via sys.argv.
"""
import os
import sys
import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "rolling_backtest", str(REPO / "test" / "rolling_backtest.py")
)
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)
rb.MODEL_NAMES = ("lgb",)

from code.src.ensemble.portfolio import mcap_constrained_topk

MCAP_MIN_LARGE = int(os.environ.get("MCAP_MIN_LARGE", "0"))
MCAP_CAND_K = int(os.environ.get("MCAP_CAND_K", "10"))
MCAP_LARGE_Q = float(os.environ.get("MCAP_LARGE_Q", "0.5"))

_PANEL_CACHE = {"panel": None}


def _load_mktcap_panel():
    if _PANEL_CACHE["panel"] is not None:
        return _PANEL_CACHE["panel"]
    from code.src.features.build import build_feature_sets
    fs = build_feature_sets(str(REPO / "data"), str(REPO / "temp"), use_cache=True)
    panel = fs["lgb"]["panel"][["instrument", "datetime", "log_mktcap"]].copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    _PANEL_CACHE["panel"] = panel
    return panel

# Wrap _load_ensemble_cfg to drop zero-weight models so blend_scores won't look up rank_master etc.
_orig_load = rb._load_ensemble_cfg
def _patched_load(model_dir):
    ens = _orig_load(model_dir)
    if "weights" in ens:
        ens["weights"] = {k: v for k, v in ens["weights"].items() if k in rb.MODEL_NAMES}
    if "legacy_weights" in ens:
        ens["legacy_weights"] = {k: v for k, v in ens["legacy_weights"].items() if k in rb.MODEL_NAMES}
    return ens
rb._load_ensemble_cfg = _patched_load


# Monkey-patch build_portfolio to re-allocate weights by ALLOC_MODE env var.
_orig_build_portfolio = rb.build_portfolio
def _patched_build_portfolio(*args, **kwargs):
    # Stage 0: market-cap constrained top-K (applied BEFORE build_portfolio).
    today_scores = args[0] if args else kwargs.get("blended_scores_today")
    top_k = kwargs.get("top_k", args[3] if len(args) >= 4 else 5)
    if (MCAP_MIN_LARGE > 0 and today_scores is not None
            and "datetime" in today_scores.columns):
        try:
            panel = _load_mktcap_panel()
            as_of = pd.to_datetime(today_scores["datetime"]).max()
            mktcap_snap = panel[panel["datetime"] == as_of][
                ["instrument", "log_mktcap"]
            ].rename(columns={"instrument": "stock_id"})
            if not mktcap_snap.empty:
                scores_in = today_scores.rename(
                    columns={"instrument": "stock_id", "final_score": "score"}
                )[["stock_id", "score"]].copy()
                scores_in["stock_id"] = scores_in["stock_id"].astype(str).str.zfill(6)
                mktcap_snap["stock_id"] = mktcap_snap["stock_id"].astype(str).str.zfill(6)
                picked = mcap_constrained_topk(
                    scores_in, mktcap_snap, k=top_k,
                    candidate_k=MCAP_CAND_K,
                    min_large_cap=MCAP_MIN_LARGE,
                    large_cap_quantile=MCAP_LARGE_Q,
                )
                keep = set(picked["stock_id"].astype(str).str.zfill(6))
                today_zfill = today_scores["instrument"].astype(str).str.zfill(6)
                mask = today_zfill.isin(keep)
                today_scores = today_scores[mask].copy()
                if args:
                    args = (today_scores,) + args[1:]
                else:
                    kwargs["blended_scores_today"] = today_scores
        except Exception as e:
            print(f"[rb-lgb-only] mcap constraint skipped: {e}", file=sys.stderr)

    port = _orig_build_portfolio(*args, **kwargs)
    mode = os.environ.get("ALLOC_MODE", "equal")
    if mode == "equal" or port is None or len(port) == 0:
        return port

    from code.src.ensemble.allocation import allocate_by_mode
    # build_portfolio returns stock_id + weight (after confidence scaling).
    # We reallocate by score. The original `today_scores` isn't exposed here,
    # so we use the returned order & weights as rank proxy.
    # Better: the first argument is the scores DataFrame.
    today_scores = args[0] if args else kwargs.get("blended_scores_today")
    if today_scores is None:
        return port
    # today_scores has ['instrument', 'final_score']
    order = today_scores.rename(columns={"instrument": "stock_id",
                                          "final_score": "score_q"})
    order = order[order["stock_id"].astype(str).str.zfill(6).isin(
        port["stock_id"].astype(str).str.zfill(6))].copy()
    order["stock_id"] = order["stock_id"].astype(str).str.zfill(6)
    order = order.sort_values("score_q", ascending=False).reset_index(drop=True)

    # Apply target allocation, scaled by total_position (confidence).
    total_position = float(port["weight"].sum())
    realloc = allocate_by_mode(order[["stock_id", "score_q"]], mode)
    realloc["weight"] = realloc["weight"] * total_position
    return realloc[["stock_id", "weight"]].reset_index(drop=True)

rb.build_portfolio = _patched_build_portfolio


if __name__ == "__main__":
    print(f"[rb-lgb-only] ALLOC_MODE = {os.environ.get('ALLOC_MODE', 'equal')}", file=sys.stderr)
    rb.main()
