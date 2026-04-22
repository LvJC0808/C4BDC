# 跨硬件训练指南

支持在多台不同 GPU 的机器上训练，产物按硬件归档，推理时用 `scripts/activate_model.sh` 切换激活版本。

## 目录约定

```
model/
├── 5090/                       # 当前已有：5090 完整训练（3 seeds）
│   ├── lgb/master/mixer/
│   └── ensemble_config.json
├── 4060/                       # 新训练：4060 产物（2 seeds）
│   └── ...
├── lgb/ master/ mixer/         # "激活版本" —— activate_model.sh 的硬链接
└── ensemble_config.json        # 同上

scripts/activate_model.sh 5090  # 切换激活到 5090 版本
scripts/activate_model.sh 4060  # 切换激活到 4060 版本
```

`pipeline.py predict` 和 `test.sh` 始终读 `model/` 顶层目录，与归档目录解耦。

## SEEDS 环境变量

`code/src/config.py` 的 `SEEDS` 从环境变量 `SEEDS` 读取，默认 `42,2024,7`。
短训可用 `SEEDS=42,2024` 跑 2 个种子。

## 4060 训练全流程

### 1. 同步代码 + 数据到 4060 机器

```bash
# on 4060 machine:
git clone <repo> THU-BDC2026
cd THU-BDC2026
git checkout feat/ensemble-v1

# 装依赖（需网络；uv 已在 pyproject.toml 配置）
uv sync      # 或：.venv/bin/pip install -r requirements.txt

# 同步数据（避免重新 baostock 抓取 35min）
rsync -avz --progress <5090-host>:/path/THU-BDC2026/data/ ./data/
```

### 2. 启动训练（2 seeds，输出到 model/4060/）

```bash
mkdir -p model/4060
SEEDS=42,2024 \
    .venv/bin/python -u code/src/pipeline.py train \
      --data_path ./data \
      --model_dir ./model/4060 \
      --temp_dir ./temp \
      > /tmp/train_4060.log 2>&1 &
```

### 3. 监控

```bash
# 进度 + 显存
watch -n 5 'nvidia-smi; echo ---; tail -5 /tmp/train_4060.log | grep -v "it/s"'

# 如显存告警（4060 只有 8 GB），降低 batch 或减 d_model
```

### 4. 完成后切到 4060 版本做推理

```bash
./scripts/activate_model.sh 4060      # 激活 4060 产物
bash test.sh                          # 或：python code/src/pipeline.py predict ...
.venv/bin/python test/score_self.py   # 自测得分

# 跑滚动回测
.venv/bin/python test/rolling_backtest.py \
    --eval_start 2025-11-01 --eval_end 2026-03-13 \
    --feature_end 2026-03-20 \
    --out test/rolling_backtest_4060.csv
```

### 5. 对比 5090 vs 4060

```bash
# 切回 5090 做基准
./scripts/activate_model.sh 5090
# 看两个回测 CSV 的 mean/std/t-test 差异
python -c "
import pandas as pd
for tag in ['', '_4060']:
    df = pd.read_csv(f'test/rolling_backtest_full{tag}.csv' if not tag else f'test/rolling_backtest{tag}.csv')
    print(tag or '5090', 'mean', df['portfolio_return'].mean(), 'win', (df['portfolio_return']>0).mean())
"
```

## 跨硬件复现性说明

即使 `torch.use_deterministic_algorithms(True)`，**不同 GPU 架构（sm_89 vs sm_120）浮点结果可能小幅不同**。
同硬件 + 同代码 + 同 SEEDS → 字节级一致；跨硬件 → 统计分布一致但具体权重 md5 不同。
赛题在评委机器上评测，本地跨机差异仅用于自检。

## 预算估计（4060 vs 5090）

| 项 | 5090 实测 | 4060 估计（2 seeds） |
|---|---|---|
| 特征工程 | 2.5 min | ~3 min |
| LGB +DE × 2 seeds × 3 folds + refit | ~30 min | ~30 min（CPU） |
| MASTER ×2×3+refit | ~2.5 h | **~4-5 h** |
| Mixer ×2×3+refit | ~40 min | **~1.5 h** |
| **合计** | **~6.3 h** | **~6-7 h** ✅ |

2 seeds 配置足以保持在 8 h 预算内。
