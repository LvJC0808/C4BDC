"""Feature-build validation entrypoint (required by competition harness)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from code.src import config  # type: ignore
    from code.src.features.build import build_feature_sets  # type: ignore
else:
    from . import config
    from .features.build import build_feature_sets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="featurework")
    parser.add_argument("--data_path", default=config.DATA_PATH)
    parser.add_argument("--temp_dir", default=config.TEMP_DIR)
    args = parser.parse_args(argv)
    fs = build_feature_sets(args.data_path, args.temp_dir, use_cache=True)
    for k in ("lgb", "master", "mixer"):
        sub = fs[k]
        panel = sub["panel"]
        print(f"[featurework] {k}: rows={len(panel)} feats={len(sub['feature_cols'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
