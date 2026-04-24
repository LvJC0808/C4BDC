# THU-BDC2026 - LGB-only 主线方案（W1 提交版）

本 README 按 THU 大数据竞赛提交要求编写。当前主线是 **LGB-only + M10-3 防守补丁**：训练只保留 LightGBM + DoubleEnsemble × 3 seeds；推理增加"市值硬约束 Top-K"（`MCAP_MIN_LARGE=3`）。入口由 `bash train.sh` 与 `bash test.sh` 调度到 `scripts/train_lgb_only.py` 与 `scripts/predict_lgb_only.py`。

**W1 提交元数据**：
- 分支：`feat/ensemble-v1` · commit `77a8d61`
- 提交日：2026-04-25
- Top-5 持仓（target = 2026-04-23）：`300308, 600023, 688187, 688256, 002714`（各 0.2）
- result.csv MD5：`f13034946c0aaea5cb1e3f2d0d6ad692`（本机双跑一致）

---

## 1. 环境配置

### 1.1 基础镜像

- 基础镜像：`python:3.12-slim-bookworm`
- C 库：源码编译安装 `ta-lib 0.4.0`（见 `Dockerfile`）
- 包管理：[`uv`](https://github.com/astral-sh/uv)
- 虚拟环境：`/app/.venv`，由 `uv sync --frozen` 基于 `pyproject.toml` + `uv.lock` 固定版本构建

### 1.2 主要 Python 依赖（见 `pyproject.toml`）

| 包 | 版本 | 用途 |
| --- | --- | --- |
| `lightgbm` | `>=4.6` | LGB-only 主模型 |
| `numpy` / `pandas` / `scipy` | 常规 | 数据处理 |
| `scikit-learn` | `>=1.7` | CV、标准化、线性残差中性化 |
| `pyarrow` | `>=16` | Parquet 缓存 |
| `TA-Lib` | `>=0.6.8` | 技术指标 |
| `baostock` | `>=0.8.9` | 离线拉取行情（仅本地使用） |
| `tqdm` | `>=4.67` | 进度条 |
| `torch` | `>=2.6.0` | 仅保留为 legacy 代码依赖，不是主线必须 |

### 1.3 镜像与脚本

竞赛标准四脚本：

- `Dockerfile`：构建镜像，创建 `/app/{model,output,temp}`。
- `init.sh`：容器启动健康检查，确认依赖可导入。
- `train.sh`：调用 `python scripts/train_lgb_only.py`。
- `test.sh`：调用 `python scripts/predict_lgb_only.py`。

---

## 2. 数据来源

- 标的池：沪深 300 (HS300) 成分股。
- 源头：[`baostock`](http://baostock.com/)，通过本地脚本 `get_stock_data.py` 拉取。
- 时间范围：训练与回测截止 **2026-04-01**。
- 字段：日频量价（open/high/low/close/volume/amount）+ 基本面（行业、市值、beta）。
- 存储：CSV + Parquet 缓存于 `data/`（镜像内 `/app/data`）。
- 预训练模型：**无**；全部参数均在镜像内通过 `train.sh` 从零训练。

---

## 3. 算法整体思路

总体为 **LGB-only 排序主线 + DoubleEnsemble 重加权**：

1. 同一套日频因子只馈入 LightGBM + DoubleEnsemble。
2. 采用 walk-forward + embargo 切分，按固定随机种子重复训练，保留 refit 模型。
3. 推理阶段对目标交易日截面排序，直接输出 Top-5 股票及其权重。

### 3.1 主线特点

- **单模型主线**：去掉 MASTER / StockMixer，只保留 CPU 可稳定复现的 LGB 路径。
- **DoubleEnsemble**（Han et al. 2020）：对困难样本重加权，提升 LGB 稳健性。
- **中性化 + rank-gauss**：用 industry / log-mktcap / beta 做线性残差，对标签与特征 `rank -> Φ⁻¹` 高斯化。
- **确定性推理**：最终 Top-5 由稳定的排序规则生成，避免浮点抖动改变提交文件。
- **W1 · 市值硬约束 Top-K（M10-3）**：从 Top-10 候选中强制选出**至少 3 只大盘股**（log_mktcap ≥ HS300 中位数），对冲近期小盘 → 大盘风格轮动风险。实现见 `code/src/ensemble/portfolio.py::mcap_constrained_topk`。

### 3.2 213 天 rolling 回测结果（2025-06-03 ~ 2026-04-16）

| 指标 | **M10-3（提交）** | Baseline-LGB | HS300 等权 | Mom-TopK |
|---|---|---|---|---|
| Mean 5d return | **+2.95%** | +2.91% | +0.37% | +0.35% |
| 胜率 (>0) | **87%** | 85% | 65% | 55% |
| t-stat vs HS300 | **+12.0** | — | — | — |
| 近 30 d Δ vs HS300 | **+0.38 pp** | +0.30 pp | — | — |

### 3.3 对比赛方 baseline（6 天重叠，2026-04-01 ~ 04-09）

| 档位 | mean 5d | 赛方 baseline | 超额 | 胜率 |
|---|---|---|---|---|
| **M10-3** | **+0.76%** | −0.88% | **+1.64 pp** | 5/6 (83%) |

赛方 baseline（StockTransformer，权重 MD5 `d9c56a6a...`）永远固定押 5 大金融蓝筹（600919/601658/601169/601939/601328 等），6 天均值 −0.881%。

---

## 4. 模型说明

源码位于 `code/src/models/`。

### 4.1 LGB (`code/src/models/lgb_de.py`)

- `boosting_type="gbdt"`，`objective="regression_l1"`，`deterministic=True`，`num_threads` 固定。
- DoubleEnsemble：两轮，按首轮残差与 feature-shuffle loss 计算样本权重。

---

## 5. 损失函数

- **LGB**：`regression_l1`（MAE），对收益率分布的厚尾更鲁棒；再叠加 DoubleEnsemble 样本权重。

---

## 6. 数据扩增

本方案 **不做显式数据扩增**（不引入合成样本）。替代手段：

- **Cross-sectional neutralize**：对特征和标签，分别回归 `industry + log_mktcap + beta`，取残差。
- **Rank-Gauss transform**：`rank / N -> Φ⁻¹`，稳定分布、消除重尾。

---

## 7. Deterministic Top-5

最终提交使用固定的 Top-5 规则，确保同一输入得到同一输出：

1. 先对当天所有候选股票的 score 做 **1e-4 量化**，即将分数四舍五入到 `0.0001` 精度。
2. 若量化后出现相同 score，则按 `stock_id` 的**字典序**做 tie-break。
3. 取前 5 只股票后，权重固定为 **0.2**，五只股票等权。

这意味着只要候选股票集合不变，结果文件的行顺序与权重都稳定。

---

## 8. 训练流程

入口：`bash train.sh` -> `python scripts/train_lgb_only.py`。

1. **特征构建**：`featurework.py` + `features/` 生成 158+39 因子，做中性化 + rank-gauss。
2. **交叉验证**：Walk-forward + embargo，3 折 × 3 随机种子 = 9 次训练；记录 OOF。
3. **Refit**：用完整历史重训 LGB，保存到 `/app/model_lgb_only/`。
4. 输出 `ensemble_config.json`，记录 `method=lgb_only`、`top_k=5`、种子与 CV 配置。
5. 训练日志写入 `/app/temp/train.log`，`config.py` 集中管理超参数与随机种子。

---

## 9. 推理流程

入口：`bash test.sh` -> `python scripts/predict_lgb_only.py`。

1. 读取最新 `/app/data`，构建与训练一致的特征。
2. 加载 LGB refit checkpoint（`model_lgb_only/lgb/seed_{42,2024,7}_refit/`）；对目标交易日截面打分。
3. 对 3 个 seed 的 score 取均值，作为融合后的最终 score。
4. **Deterministic Top-10** 先取候选（quantize 1e-4 + stock_id 字典序 tie-break）。
5. **Market-cap constrained 选 5**：从 Top-10 中贪心挑 5 只，强制大盘 ≥ `MCAP_MIN_LARGE`；若 Top-10 大盘不足则降级回原 Top-5。
6. 五只股票等权 `0.2`，输出 `/app/output/result.csv`（`stock_id,weight`）。

### 9.1 环境变量（`test.sh` 已设默认值）

| 变量 | 默认值 | 含义 |
|---|---|---|
| `MCAP_MIN_LARGE` | `3` | Top-5 中大盘股最低数量（W1 提交值） |
| `MCAP_CAND_K` | `10` | 候选池大小（Top-N 再做市值约束） |
| `MCAP_LARGE_Q` | `0.5` | 大盘 log_mktcap 分位阈值（0.5 = 中位数） |
| `TARGET_DATE` | 最新交易日 | 可选，覆盖预测目标日（YYYY-MM-DD） |

设 `MCAP_MIN_LARGE=0` 可完全禁用防守补丁，行为退回原 Deterministic Top-5。

---

## 10. 可复现性校验

提交前使用以下命令检查结果稳定性：

```bash
python code/src/verify_reproducibility.py --single temp/result_run1.csv temp/result_run2.csv
python code/src/verify_reproducibility.py --cross /tmp/result_5090.csv /tmp/result_4060.csv
```

判定标准：

- Top-5 交集 `>= 4/5`
- 共享股票的最大绝对权重差 `<= 0.01`

---

## 11. 复现性与其他注意事项

- **固定种子**：`config.SEED`，传播至 `numpy` / `lightgbm` / `sklearn`。
- **Deterministic**：
  - LightGBM 设 `deterministic=True, force_row_wise=True, num_threads` 固定。
  - 环境变量 `PYTHONHASHSEED=0`。
- **复现性自检**：`code/src/verify_reproducibility.py` 支持单机与跨机比对，建议在提交前先跑一遍。
- **数据截止**：训练与 baostock 拉取均止于 `2026-04-01`，保证与评测口径一致。
- **镜像体积**：通过 `apt` 清理、移除主线 torch、删除 ta-lib 源码，目标控制在 2 GB 以内。

---

## 12. 代码结构速查

```
code/src/
├── pipeline.py              # train / predict 顶层 CLI
├── config.py                # 超参数与路径
├── featurework.py           # 因子构建主流程
├── features/                # 单因子实现
├── data_prep/               # 数据加载、清洗、缓存
├── models/                  # lgb_de / legacy master / stockmixer
├── cv/                      # walk-forward + embargo
├── ensemble/                # 排序、组合与仓位逻辑
├── train.py / test.py       # legacy 训练 / 推理步骤
├── predict.py               # legacy 被 test.sh 间接调用
└── verify_reproducibility.py
```

提交文件：`Dockerfile`、`docker-compose.yml`、`init.sh`、`train.sh`、`test.sh`、`readme.md`、`code/`、`scripts/`、`data/`、`model/`（赛方 baseline 权重）、`model_lgb_only/`（LGB-only 主线权重）。

---

## 13. Docker 提交与运行

### 13.1 构建镜像（本地）

```bash
docker buildx build --platform linux/amd64 \
  --build-arg IMAGE_NAME=nvidia/cuda \
  -t bdc2026 .
```

首次构建 ~15-25 分钟（主要是 TA-Lib C 库编译 + uv sync）。

### 13.2 本地 dry-run 验证

```bash
docker compose up
```

`docker-compose.yml` 默认调用 `/app/test.sh`，会挂载当前 `./data`、`./output`、`./temp`，推理结束后在 `./output/result.csv` 看到结果。

### 13.3 导出提交包

```bash
docker save -o <team_name>.tar bdc2026:latest
ls -lh <team_name>.tar   # 应 < 10 GB
```

### 13.4 赛方加载与评测（供赛方参考）

```bash
docker load -i <team_name>.tar
docker compose up   # 使用赛方自己的 docker-compose.yml 挂载评测数据
cat output/result.csv
```

---

## 14. 方法论文档索引

详细的方法演化、失败实验与决策理由，见以下文档：

| 文件 | 内容 |
|---|---|
| `docs/reports/2026-04-24-path-to-now-v2.md` | v2 阶段完整路径报告（3.15 → 4.24） |
| `docs/reports/2026-04-24-w1-ab-decision.md` | W1 三档 AB（Baseline / M10-2 / M10-3）决策记录 |
| `docs/reports/2026-04-24-stage-report.md` | 2026-04-24 阶段性日报（数据更新、Regime 诊断） |
| `docs/reference/baseline-authoritative.md` | 赛方 baseline 权威记录（代码/权重 MD5/表现数据） |
| `docs/superpowers/specs/2026-04-24-w1-defensive-patch-design.md` | W1 防守补丁设计 spec |
| `docs/slides/2026-04-24-w1-defense.pptx` | 10 分钟答辩 PPT |
