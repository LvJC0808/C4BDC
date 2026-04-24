#!/bin/bash
set -e

# Ensure runtime dirs exist (in case mounts are empty)
mkdir -p /app/model /app/output /app/temp

# ==============================================================================
# Data self-healing: judges mount their own /app/data (train.csv + test.csv only).
# Our pipeline needs stock_data.csv + 5 auxiliary CSVs. If missing, reconstruct.
# ==============================================================================

need_heal=0
required_files=(
    "stock_data.csv"
    "industry_map.csv"
    "hs300_history.csv"
    "csi300_index.csv"
    "stock_basic.csv"
    "trade_calendar.csv"
)
for f in "${required_files[@]}"; do
    if [ ! -f "/app/data/$f" ]; then
        need_heal=1
        break
    fi
done

if [ "$need_heal" = "1" ]; then
    echo "[init] /app/data/ missing required CSVs; self-healing from /app/data_bundled/"

    # 1. If judges provided train.csv + test.csv, merge them into stock_data.csv
    if [ -f "/app/data/train.csv" ] && [ -f "/app/data/test.csv" ] && [ ! -f "/app/data/stock_data.csv" ]; then
        echo "[init] merging /app/data/{train,test}.csv -> /app/data/stock_data.csv"
        python - <<'PY'
import pandas as pd, os
tr = pd.read_csv("/app/data/train.csv")
te = pd.read_csv("/app/data/test.csv")
full = pd.concat([tr, te], ignore_index=True)
full = full.drop_duplicates(subset=["股票代码", "日期"]).sort_values(["股票代码", "日期"])
full.to_csv("/app/data/stock_data.csv", index=False, encoding="utf-8-sig")
print(f"[init] merged stock_data.csv: {len(full)} rows")
PY
    fi

    # 2. Copy auxiliary files from bundled backup if still missing
    for f in "${required_files[@]}"; do
        if [ ! -f "/app/data/$f" ] && [ -f "/app/data_bundled/$f" ]; then
            echo "[init] restoring $f from /app/data_bundled/"
            cp "/app/data_bundled/$f" "/app/data/$f"
        fi
    done

    # 3. Also restore hs300_stock_list.csv (used by fetch_all.py, non-critical)
    if [ -f "/app/data_bundled/hs300_stock_list.csv" ] && [ ! -f "/app/data/hs300_stock_list.csv" ]; then
        cp /app/data_bundled/hs300_stock_list.csv /app/data/hs300_stock_list.csv
    fi
fi

# Final dependency sanity check
python -c "import lightgbm, numpy, pandas, scipy, sklearn, pyarrow, talib; print('[init] deps OK')"
echo "[init] init.sh complete"
