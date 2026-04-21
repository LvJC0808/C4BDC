"""Run pipeline predict twice, compare MD5 of result.csv.

Usage:
    python verify_reproducibility.py \
        --data_path /app/data \
        --model_dir /app/model \
        --temp_dir /app/temp
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_predict(data_path: str, model_dir: str, temp_dir: str, output_path: str) -> None:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.py")
    cmd = [
        sys.executable, script, "predict",
        "--data_path", data_path,
        "--model_dir", model_dir,
        "--temp_dir", temp_dir,
        "--output_path", output_path,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--temp_dir", required=True)
    args = ap.parse_args()

    temp = Path(args.temp_dir)
    temp.mkdir(parents=True, exist_ok=True)
    out1 = str(temp / "result_1.csv")
    out2 = str(temp / "result_2.csv")

    run_predict(args.data_path, args.model_dir, args.temp_dir, out1)
    run_predict(args.data_path, args.model_dir, args.temp_dir, out2)

    h1, h2 = md5sum(out1), md5sum(out2)
    log_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "repro_check.log"
    ok = h1 == h2
    with open(log_path, "a") as f:
        f.write(
            f"[{datetime.utcnow().isoformat()}Z] result_1={h1} result_2={h2} "
            f"match={ok}\n"
        )
    print(f"md5(result_1)={h1}")
    print(f"md5(result_2)={h2}")
    print(f"match={ok}  log={log_path}")
    assert ok, "Reproducibility check FAILED: predictions differ"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
