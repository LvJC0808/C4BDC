#!/bin/bash
# 每日数据更新脚本 —— 爬取 baostock hs300 日线数据到今日
# 用法：./scripts/fetch_today.sh [end_date]
#   end_date: 默认 = 今日 (date +%Y-%m-%d)，可传 "2026-04-24" 等
#
# 推荐晚上 19:00 之后运行（收盘+清算数据上 baostock）
# 若网络不通自动套代理

set -e

cd "$(dirname "$0")/.."

END_DATE="${1:-$(date +%Y-%m-%d)}"
echo "=== fetch_today.sh ==="
echo "target end_date: $END_DATE"

# 代理（仅在网络受限时启用）
if ! curl -sI --max-time 5 https://pypi.org >/dev/null 2>&1; then
    echo "[info] direct network fails, exporting proxy"
    export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
    export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
    export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
fi

# 临时改写 get_stock_data.py 里的 end_date
# （更稳妥：用环境变量方式；这里直接 sed in-place）
BACKUP=$(mktemp)
cp get_stock_data.py "$BACKUP"
sed -i "s|end_date = \"[0-9-]*\"|end_date = \"$END_DATE\"|" get_stock_data.py

LOG="temp/fetch_$(date +%Y%m%d_%H%M%S).log"
mkdir -p temp

echo "[info] running get_stock_data.py, log -> $LOG"
PYTHONUNBUFFERED=1 .venv/bin/python get_stock_data.py 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

# 恢复 get_stock_data.py
mv "$BACKUP" get_stock_data.py

if [ "$RC" -ne 0 ]; then
    echo "[error] fetch failed with exit $RC"
    exit "$RC"
fi

# 汇报结果
.venv/bin/python - <<'PY'
import pandas as pd
d = pd.read_csv("data/stock_data.csv")
d["日期"] = pd.to_datetime(d["日期"])
print(f"=== data/stock_data.csv ===")
print(f"rows    : {len(d):,}")
print(f"stocks  : {d['股票代码'].nunique()}")
print(f"date    : {d['日期'].min().date()}  ..  {d['日期'].max().date()}")
PY

echo "=== done ==="
