"""Path B: LGB-only rolling backtest wrapper.

Monkey-patch MODEL_NAMES to ("lgb",), then invoke rolling_backtest.main().
All argparse args forwarded via sys.argv.
"""
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

if __name__ == "__main__":
    rb.main()
