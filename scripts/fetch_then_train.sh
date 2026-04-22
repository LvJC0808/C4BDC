#!/bin/bash
# fetch_then_train.sh
#
# 队友一键脚本：
#   1. 抓取 baostock 6 个 CSV（~35 min，需联网，带代理）
#   2. 校验数据完整性（含 peTTM 等估值列）
#   3. 启动完整训练（~6-8 h）
#   4. 训练完成后立即跑一次滚动回测验证
#
# 用法：
#   ./scripts/fetch_then_train.sh                  # 默认 3 seeds
#   SEEDS=42,2024 ./scripts/fetch_then_train.sh    # 2 seeds（4060 友好）
#
# 日志位置：/tmp/bdc2026_fetch.log / /tmp/bdc2026_train.log

set -euo pipefail

# ---- 0. 进入仓库根 ----
cd "$(dirname "$0")/.."
REPO=$(pwd)
echo "[$(date '+%F %T')] repo: $REPO"

# ---- 1. 代理（按需配置，抓取和装依赖用，训练时不需要） ----
export https_proxy="${HTTPS_PROXY:-http://u-UE25Z3:tXGJgV92@10.255.128.102:3128}"
export http_proxy="${HTTP_PROXY:-http://u-UE25Z3:tXGJgV92@10.255.128.102:3128}"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,localhost"

# ---- 2. 环境自检 ----
if [[ ! -x .venv/bin/python ]]; then
    echo "[ERR] .venv/bin/python not found. Run 'uv sync' first."
    exit 1
fi
.venv/bin/python -c "import lightgbm, torch, talib, numpy, pandas, pyarrow, baostock" \
    || { echo "[ERR] missing deps. Run 'uv sync' or pip install lightgbm torch ta-lib pyarrow baostock"; exit 1; }

# ---- 3. 抓取数据 ----
mkdir -p data model temp output
FETCH_LOG=/tmp/bdc2026_fetch.log
TRAIN_LOG=/tmp/bdc2026_train.log

echo "[$(date '+%F %T')] fetching baostock data → $FETCH_LOG (~35 min)"
.venv/bin/python -u code/src/data_prep/fetch_all.py \
    --start 2018-01-01 --end 2026-04-22 --out ./data \
    > "$FETCH_LOG" 2>&1
echo "[$(date '+%F %T')] fetch done."

# ---- 4. 校验：必须含 peTTM 等估值列 ----
.venv/bin/python - <<'PY'
import pandas as pd, sys
df = pd.read_csv('data/stock_data.csv', nrows=5)
required = {'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM', '开盘', '收盘', '涨跌幅'}
missing = required - set(df.columns)
if missing:
    print(f'[ERR] stock_data.csv missing columns: {sorted(missing)}')
    sys.exit(1)
print(f'[OK] stock_data.csv has {len(df.columns)} columns, valuation fields present.')
for f in ['industry_map.csv', 'stock_basic.csv', 'hs300_history.csv',
          'csi300_index.csv', 'trade_calendar.csv']:
    try:
        pd.read_csv(f'data/{f}', nrows=1)
        print(f'[OK] data/{f}')
    except Exception as e:
        print(f'[ERR] data/{f}: {e}'); sys.exit(1)
PY

# ---- 5. 打印 MD5（方便多机对齐） ----
echo "[$(date '+%F %T')] data MD5:"
md5sum data/stock_data.csv data/industry_map.csv data/stock_basic.csv \
       data/hs300_history.csv data/csi300_index.csv data/trade_calendar.csv \
    | tee data/data.md5

# ---- 6. 启动训练 ----
# 代理关掉，训练纯离线
unset https_proxy http_proxy

SEEDS="${SEEDS:-42,2024,7}"
echo "[$(date '+%F %T')] starting training (SEEDS=$SEEDS) → $TRAIN_LOG"
SEEDS="$SEEDS" .venv/bin/python -u code/src/pipeline.py train \
    --data_path ./data --model_dir ./model --temp_dir ./temp \
    > "$TRAIN_LOG" 2>&1
echo "[$(date '+%F %T')] training done."

# ---- 7. 校验训练产物 ----
if [[ ! -f model/ensemble_config.json ]]; then
    echo "[ERR] ensemble_config.json missing after training — check $TRAIN_LOG"
    exit 1
fi
echo "[OK] ensemble_config.json:"
cat model/ensemble_config.json

# ---- 8. 自测：推理 + score_self ----
echo "[$(date '+%F %T')] running inference (test.sh)"
bash test.sh 2>&1 | tail -20
echo "[OK] output/result.csv:"
cat output/result.csv

# ---- 9. （可选）跑 20 天快速回测 ----
if [[ "${RUN_BACKTEST:-1}" == "1" ]]; then
    echo "[$(date '+%F %T')] running short rolling backtest (recent 30 days)"
    LAST_DATE=$(.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('data/stock_data.csv')['日期']
d = pd.to_datetime(d)
print((d.max() - pd.Timedelta(days=40)).date(), (d.max() - pd.Timedelta(days=14)).date())
")
    START=$(echo "$LAST_DATE" | awk '{print $1}')
    END=$(echo "$LAST_DATE"   | awk '{print $2}')
    .venv/bin/python test/rolling_backtest.py \
        --eval_start "$START" --eval_end "$END" \
        --tradable_filter --cost_bps 3 \
        --out test/rolling_backtest_auto.csv \
        2>&1 | tail -30
fi

echo "[$(date '+%F %T')] ALL DONE."
echo "  training log : $TRAIN_LOG"
echo "  result.csv   : $(wc -l < output/result.csv) lines"
echo "  models saved : model/{lgb,master,mixer}/seed_*_refit/"
