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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "rolling_backtest", str(REPO / "test" / "rolling_backtest.py")
)
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)
rb.MODEL_NAMES = ("lgb",)

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
    port = _orig_build_portfolio(*args, **kwargs)
    mode = os.environ.get("ALLOC_MODE", "equal")
    if mode == "equal" or port is None or len(port) == 0:
        return port

    from code.src.ensemble.allocation import allocate_by_mode
    import pandas as pd
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
