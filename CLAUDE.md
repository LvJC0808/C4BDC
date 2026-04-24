# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always speak Chinese to user

请使用第一性原理思考。不要默认我清楚自己的需求或达成路径。请从原始问题出发，保持审慎：
如果我的动机或目标不清晰，请停下来与我讨论；
如果目标清晰但你的方案并非最短路径，请直接指出，并给出更优建议。
除非我明确要求，否则不要调用 brainstorming/writing-plans skills，直接实现
读或写文件时超过180行必须分批读取或写入

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

## 代理（网络受限时使用）
```bash
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
```
