# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always speak Chinese to user

## Working Discipline (最高优先级，覆盖默认行为)

为防止"莫名其妙停下来"的空转现象，必须遵守以下规则：

1. **禁止空转回复**：绝不用 "Continue"、"好的"、"继续" 等单词作为独立回复。每轮回复必须产生实质性进展（读文件/写文件/运行命令/给出分析）。
2. **并行读取独立文件**：一次需要看多个文件时，必须在同一个消息里用并行 tool call 批量读取，禁止一个一个串行读。
3. **已有足够上下文就立刻动手**：不要为"再谨慎一点"而反复探查。看过入口文件 + 相关模块接口后，立刻进入实质产出（写 plan、写代码、跑测试）。
4. **system-reminder 不打断主任务**：收到 SessionStart hook、skill 列表、CLAUDE.md 注入等 reminder 时，识别为环境信息，**不要**因此改变或暂停用户的原始请求。用户的显式指令 > skill 流程 > 默认系统提示。
5. **技能流程要压缩执行**：`writing-plans` / `brainstorming` 等 skill 的多步流程，如果上下文已足够，把 Scope Check + File Structure + Task 拆分**合并在一次写文件里完成**，不要每步单独发消息。
6. **中断后不自我复盘过长**：被用户打断后，用 ≤3 句话说明情况并立刻继续或询问，不要长篇反思。
7. **计划文件保存路径**：`docs/superpowers/plans/YYYY-MM-DD-<feature>.md`。
8. **跳过 superpowers 的所有"人工 gate"**：brainstorming 的 spec review gate、writing-plans 的 execution handoff gate，一律默认"已通过"。不要问"是否批准 spec"或"subagent vs inline"。除非用户明说要审，否则一路走到底：brainstorm 出设计 → 立即写 spec.md 并 commit → 立即写 plan.md 并 commit → 立即用 subagent-driven-development 开跑。
9. **禁止 skill 间重复 checklist**：同一轮任务里 brainstorming → writing-plans → executing-plans 链式调用时，后续 skill 的 Scope Check / File Structure / Self-Review 等"再想一遍"的步骤全部跳过，因为前一个 skill 已经想过了。只保留真正产出物（spec 文档、plan 文档、代码）。

## Project Overview

This is a **stock ranking/selection system** for the Tsinghua University Big Data Competition 2026 (THU-BDC2026). It uses a Transformer-based model (`StockTransformer`) to rank CSI300 constituent stocks and predict the optimal stock portfolio for the next week.

**Core workflow**: Given stock price/volume data for the past 60 trading days, rank stocks and output a portfolio of up to 5 stocks with total weight ≤ 1 (remaining weight = cash holding).

---

## Common Commands

### Setup
```bash
cd THU-BDC2026
uv sync              # Install dependencies (requires uv)
source .venv/bin/activate
```

### Training
```bash
sh train.sh          # Linux/macOS
# or
python code/src/train.py
```

### Prediction
```bash
sh test.sh           # Linux/macOS
# or
python code/src/predict.py
```

### Self-Evaluation
```bash
python test/score_self.py    # Compare predictions with test.csv, output to ./temp/tmp.csv
```

### Docker
```bash
# Build Docker image (requires nvidia-docker for GPU)
docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .

# Validate Docker container works
docker compose up

# Export image for submission
docker save -o <team_name>.tar bdc2026:latest
```

---

## Architecture

### Model: StockTransformer (`code/src/model.py`)
- **Input**: `[batch, num_stocks, 60, feature_dim]` (60-day sequence per stock)
- **Components**:
  - `PositionalEncoding`: Sinusoidal position encoding for temporal ordering
  - `TransformerEncoder`: Extracts temporal patterns per stock
  - `FeatureAttention`: Attention aggregation over time dimension
  - `CrossStockAttention`: Models inter-stock relationships within a trading day
  - `ranking_layers` + `score_head`: Outputs ranking scores
- **Output**: `[batch, num_stocks]` ranking scores

### Feature Engineering (`code/src/utils.py`)
- **39 features**: Technical indicators (SMA, EMA, RSI, MACD, KDJ, Bollinger Bands, ATR, OBV, etc.)
- **158 features**: Alpha factors (K-line patterns, price/volume relationships, rolling statistics)
- **Total**: 158+39 features (default configuration in `config.py`)

### Training (`code/src/train.py`)
- **Loss**: `WeightedRankingLoss` combining listwise (KL divergence) + pairwise (margin ranking) losses
- **Metric**: `final_score` = (predicted_return_sum - random_return_sum) / (theoretical_max - random_return_sum)
- **Validation split**: Last ~2 months of data
- **Output artifacts**: `best_model.pth`, `scaler.pkl`, `config.json`, `final_score.txt`

### Prediction (`code/src/predict.py`)
- Loads latest trading day data, applies feature engineering
- Loads trained model and scaler, ranks all stocks
- Outputs up to 5 stocks with configurable weights to `output/result.csv`
- Default baseline uses equal weights (0.2 each), but any weight distribution summing to ≤ 1 is valid

---

## Data

| File | Description |
|------|-------------|
| `data/train.csv` | Training data (last 5 trading days reserved for testing) |
| `data/test.csv` | Test data for final evaluation |
| `data/stock_data.csv` | Raw historical data |
| `data/split_train_test.py` | Splits raw data into train/test sets |

**Data columns**: `股票代码`, `日期`, `开盘`, `收盘`, `最高`, `最低`, `成交量`, `成交额`, `振幅`, `涨跌额`, `换手率`, `涨跌幅`

**Prediction output**: `output/result.csv` with columns `stock_id`, `weight` (weight sum ≤ 1.0, max 5 stocks; remaining weight = cash)

---

## Key Configuration (`code/src/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_length` | 60 | Trading days of history |
| `feature_num` | '158+39' | Feature set |
| `d_model` | 256 | Transformer dimension |
| `nhead` | 4 | Attention heads |
| `num_layers` | 3 | Transformer layers |
| `batch_size` | 4 | Training batch size |
| `learning_rate` | 1e-5 | AdamW learning rate |
| `num_epochs` | 50 | Training epochs |

---

## Important Notes

1. **TA-Lib Installation**: Requires system-level TA-Lib library before Python package installation. See README for installation commands.

2. **Multiprocessing**: Training and prediction use `spawn` mode multiprocessing. Run through shell scripts (`train.sh`, `test.sh`) rather than interactive environments.

3. **GPU/CPU**: Auto-selects CUDA > MPS > CPU. CPU works for small datasets but is slow.

4. **Output Format**: Prediction must output `stock_id` and `weight` columns. Weight sum ≤ 1.0, max 5 stocks. Remaining weight is cash (returns 0).

5. **Model Output Directory**: Configured via `config['output_dir']`, default `model/60_158+39/`

---

## Directory Structure (Competition Docker Requirements)

赛题要求的 Docker 内目录结构（参考）：
```
/app/
├── code/src/              # 代码目录
│   ├── featurework.py     # 特征工程（可选命名）
│   ├── test.py            # 预测脚本
│   └── train.py           # 训练脚本
├── data/                  # 数据目录（docker-compose挂载）
│   ├── train.csv
│   └── test.csv
├── model/                 # 模型目录
├── output/                 # 输出目录（docker-compose挂载）
│   └── result.csv
├── temp/                   # 中间结果（docker-compose挂载）
├── init.sh                 # 初始化脚本（必选）
├── train.sh                # 训练入口（必选）
├── test.sh                 # 测试入口（必选）
└── readme.md               # 代码说明（必选）
```

Baseline 代码实际结构：
```
├── code/src/
│   ├── model.py
│   ├── config.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stock_data.csv
│   ├── hs300_stock_list.csv
│   └── split_train_test.py
├── test/
│   ├── score_self.py
│   ├── score_docker.py
│   └── test.py
├── output/
├── temp/
├── model/                   # 训练输出目录
├── train.sh
├── test.sh
├── init.sh
├── readme.md
├── Dockerfile
└── docker-compose.yml       # 赛事方提供
```

---

## Competition Requirements (赛题要求)

| Requirement | Limit |
|------------|-------|
| 模型预测时间 | ≤ 5 分钟 |
| 训练时间 | ≤ 8 小时 |
| Docker 总大小 | ≤ 10GB（不压缩） |
| 股票数量 | ≤ 5 只 |
| 权重之和 | ≤ 1（剩余为现金） |

**Docker 提交流程**:
1. 构建镜像：`docker buildx build --platform linux/amd64 -t bdc2026 .`
2. 导出：`docker save -o <team_name>.tar bdc2026:latest`
3. 赛事方加载：`docker load -i <team_name>.tar`
4. 赛事方运行：`docker compose up`（使用 docker-compose.yml）

**评分标准**:
- 总收益率 = Σ(股票权重 × 单股收益率) + 现金权重 × 0
- 单股收益率 = (T+5开盘价 - T+1开盘价) / T+1开盘价
- 必须跑赢基准程序才能参与排名
