#!/usr/bin/env bash
# Run three rolling backtests: baseline / M10-2 / M10-3
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

EVAL_START="${EVAL_START:-2025-06-03}"
EVAL_END="${EVAL_END:-2026-04-16}"

echo "=== Baseline (MCAP_MIN_LARGE=0) ==="
MCAP_MIN_LARGE=0 python scripts/rolling_backtest_lgb_only.py \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --out test/rolling_lgb_baseline.csv 2>&1 | tail -20

echo "=== M10-2 (MCAP_MIN_LARGE=2) ==="
MCAP_MIN_LARGE=2 python scripts/rolling_backtest_lgb_only.py \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --out test/rolling_lgb_mcap2.csv 2>&1 | tail -20

echo "=== M10-3 (MCAP_MIN_LARGE=3) ==="
MCAP_MIN_LARGE=3 python scripts/rolling_backtest_lgb_only.py \
  --eval_start "$EVAL_START" --eval_end "$EVAL_END" \
  --out test/rolling_lgb_mcap3.csv 2>&1 | tail -20

echo "=== 近 30 日 / 213 天对比 ==="
python - <<'PY'
import pandas as pd
for tag, f in [("Baseline", "test/rolling_lgb_baseline.csv"),
               ("M10-2",    "test/rolling_lgb_mcap2.csv"),
               ("M10-3",    "test/rolling_lgb_mcap3.csv")]:
    d = pd.read_csv(f)
    tcol = "T" if "T" in d.columns else "datetime"
    d[tcol] = pd.to_datetime(d[tcol])
    d = d.sort_values(tcol).reset_index(drop=True)
    last30 = d.tail(30)
    print(f"{tag:10s}  N={len(d):3d}  "
          f"all={d['portfolio_return'].mean()*100:+6.3f}%  "
          f"last30={last30['portfolio_return'].mean()*100:+6.3f}%  "
          f"HS300_30={last30['baseline_avg'].mean()*100:+6.3f}%  "
          f"Δ30={((last30['portfolio_return']-last30['baseline_avg']).mean())*100:+6.3f}%")
PY
