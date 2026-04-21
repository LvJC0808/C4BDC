"""Prediction entrypoint (thin wrapper around pipeline.cmd_predict)."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from code.src.pipeline import build_parser, cmd_predict  # type: ignore
else:
    from .pipeline import build_parser, cmd_predict


def main(argv=None) -> int:
    parser = build_parser()
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv or argv[0] != "predict":
        argv = ["predict", *argv]
    args = parser.parse_args(argv)
    cmd_predict(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
