# THU-BDC2026 - LGB-only 主线方案

本 README 按 THU 大数据竞赛提交要求编写，说明环境、数据、训练、推理与复现性保障。当前主线是 **LGB-only**：训练与推理都只保留 LightGBM + DoubleEnsemble，入口由 `bash train.sh` 和 `bash test.sh` 调度到 `python scripts/train_lgb_only.py` 与 `python scripts/predict_lgb_only.py`。

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
2. 加载 LGB refit checkpoint；对目标交易日截面打分。
3. 采用上面的 Deterministic Top-5 规则选出 5 只股票。
4. 五只股票等权 `0.2`，输出 `/app/output/result.csv`（`stock_id,weight`）。

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

提交文件：`Dockerfile`、`init.sh`、`train.sh`、`test.sh`、`readme.md`、`code/`、`data/`、`model/`。
