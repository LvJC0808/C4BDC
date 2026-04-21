# THU-BDC2026 —— 集成排序选股方案 (Ensemble v1)

本 README 按 THU 大数据竞赛提交要求编写，说明环境、数据、模型、训练与推理流程，以及复现性保障。对应实现分支 `feat/ensemble-v1`，核心入口 `code/src/pipeline.py`。

---

## 1. 环境配置

### 1.1 基础镜像

- 基础镜像：`python:3.12-slim-bookworm`
- C 库：源码编译安装 `ta-lib 0.4.0`（见 `Dockerfile`）
- 包管理：[`uv`](https://github.com/astral-sh/uv)（从 `ghcr.io/astral-sh/uv:latest` 拷贝二进制）
- 虚拟环境：`/app/.venv`，由 `uv sync --frozen` 基于 `pyproject.toml` + `uv.lock` 固定版本构建

### 1.2 主要 Python 依赖（见 `pyproject.toml`）

| 包 | 版本 | 用途 |
| --- | --- | --- |
| `torch` | `>=2.6.0` | MASTER / StockMixer 神经网络 |
| `lightgbm` | `>=4.6` | GBDT + DoubleEnsemble |
| `numpy` / `pandas` / `scipy` | 常规 | 数据处理 |
| `scikit-learn` | `>=1.7` | CV、标准化、线性残差中性化 |
| `pyarrow` | `>=16` | Parquet 缓存 |
| `TA-Lib` | `>=0.6.8` | 技术指标 |
| `baostock` | `>=0.8.9` | 离线拉取行情（仅本地使用） |
| `tqdm` | `>=4.67` | 进度条 |

### 1.3 镜像与脚本

竞赛标准四脚本：

- `Dockerfile`：构建镜像，创建 `/app/{model,output,temp}`。
- `init.sh`：容器启动健康检查，确认依赖可导入。
- `train.sh`：调用 `python pipeline.py train`，输入 `/app/data`，输出 `/app/model`。
- `test.sh`：调用 `python pipeline.py predict`，输出 `/app/output/result.csv`。

预估镜像大小 < 10 GB（CPU torch 轮 + lightgbm + ta-lib，约 3–4 GB 层）。

---

## 2. 数据来源

- 标的池：沪深 300 (HS300) 成分股。
- 源头：[`baostock`](http://baostock.com/)，通过本地脚本 `get_stock_data.py` 拉取。
- 时间范围：训练 & 回测截止 **2026-04-01**（cutoff）。
- 字段：日频量价（open/high/low/close/volume/amount）+ 基本面（行业、市值、beta）。
- 存储：CSV + Parquet 缓存于 `data/`（镜像内 `/app/data`）。
- 预训练模型：**无**；全部参数均在镜像内通过 `train.sh` 从零训练。

---

## 3. 算法整体思路

总体为 **三模型排序集成 + 置信度加权建仓**：

1. 同一套日频因子馈入三个异质模型：
   - **LGB + DoubleEnsemble**：两轮子样本重加权 GBDT。
   - **MASTER**：FiLM 市场门控 + 股票内时序注意力 + 股票间横截面注意力的 Transformer。
   - **StockMixer**：时间 / 特征 / 股票三路 MLP-Mixer。
2. 每日截面对三模型得分做 **rank-normalize**，在 holdout 网格搜索得到融合权重 `w*`。
3. 按融合分数取 TopK，再由 **置信度自适应仓位** 决定是否降低仓位或持现金。

### 3.1 创新点

- **三模型异质集成**：GBDT + 时序注意力 + Mixer，分别捕捉非线性、时序依赖、横截面关系。
- **DoubleEnsemble**（Han et al. 2020）：对困难样本重加权，提升 LGB 稳健性。
- **中性化 + rank-gauss**：用 industry / log-mktcap / beta 做线性残差，对标签与特征 `rank → Φ⁻¹` 高斯化。
- **置信度自适应建仓**：当 TopK 的融合分数扩散度（std）低于阈值时缩仓，避免低信号日过度交易。

---

## 4. 网络结构

源码位于 `code/src/models/`。

### 4.1 MASTER (`code/src/models/master.py`)
- 输入：每股 60 个交易日 × F 个特征序列，以及市场因子 `m_t`。
- FiLM 门：`γ, β = MLP(m_t)`，对 token 做 `x ← γ ⊙ x + β`。
- Intra-stock Transformer：标准多头自注意力沿时间维建模。
- Inter-stock Attention：同一日股票间多头注意力。
- 输出头：线性层 → 每股当期得分。

### 4.2 StockMixer (`code/src/models/stockmixer.py`)
- 三路 MLP-Mixer：Time-mix、Feature-mix、Stock-mix（交替应用于时序与横截面）。
- 每块 `LayerNorm → Linear → GELU → Linear`，残差连接。
- 最终 pooling + 线性 head 输出分数。

### 4.3 LGB (`code/src/models/lgb_de.py`)
- `boosting_type="gbdt"`，`objective="regression_l1"`，`deterministic=True`，`num_threads` 固定。
- DoubleEnsemble：两轮，按首轮残差与 feature-shuffle loss 计算样本权重。

---

## 5. 损失函数

- **神经网络 (MASTER / StockMixer)**：复合损失
  ```
  L = λ1 * MSE(ŷ, y) + λ2 * (1 − RankIC(ŷ, y)) + λ3 * TopK_margin(ŷ, y)
  ```
  其中 `RankIC` 采用 Spearman 排序相关，`TopK_margin` 鼓励预测 Top-K 与真实 Top-K 对齐。
- **LGB**：`regression_l1`（MAE），对收益率分布的厚尾更鲁棒；再叠加 DoubleEnsemble 样本权重。

---

## 6. 数据扩增

本方案 **不做显式数据扩增**（不引入合成样本）。替代手段：

- **Cross-sectional neutralize**：对特征和标签，分别回归 `industry + log_mktcap + beta`，取残差。
- **Rank-Gauss transform**：`rank / N → Φ⁻¹`，稳定分布、消除重尾。
- 轻度 dropout / weight decay 控制 NN 过拟合。

---

## 7. 模型集成

- 每日截面内将三模型预测 `rank → [0, 1]` 归一化。
- 以 holdout 窗口（最近若干月）网格搜索权重 `(w_lgb, w_master, w_mixer)`（`sum=1, step=0.05`），
  目标函数 `0.7*RankIC + 0.3*TopK收益`。
- 最优 `w*` 保存到 `model/blend_weights.json`，推理直接复用。

---

## 8. 训练流程

入口：`bash train.sh` → `python pipeline.py train`（对应 `code/src/train.py` 与 `code/src/cv/`）。

1. **特征构建**：`featurework.py` + `features/` 生成 158+39 因子，做中性化 + rank-gauss。
2. **交叉验证**：Walk-forward + embargo，3 折 × 3 随机种子 = 9 次训练；记录 OOF。
3. **Ensemble 搜索**：对 OOF 做 `rank-blend` 网格搜索。
4. **Refit**：用完整历史重训三模型，权重保存到 `/app/model/`：
   - `lgb_refit.txt`、`master_refit.pt`、`stockmixer_refit.pt`、`blend_weights.json`、`feature_scaler.pkl`。
5. 训练日志写入 `/app/temp/train.log`，`config.py` 集中管理超参数与随机种子。

---

## 9. 推理流程

入口：`bash test.sh` → `python pipeline.py predict`（对应 `code/src/predict.py` 与 `code/src/test.py`）。

1. 读取最新 `/app/data`，构建与训练一致的特征（读取 `feature_scaler.pkl`）。
2. 加载三模型 refit checkpoint；对目标交易日截面分别打分。
3. 每模型分数 rank-normalize；按 `blend_weights.json` 加权融合。
4. 选 TopK（默认 K=5），按 **置信度自适应仓位** 计算权重；总仓位不足时补现金。
5. 输出 `/app/output/result.csv`（`stock_id,weight`）。

---

## 10. 复现性与其他注意事项

- **固定种子**：`config.SEED`，传播至 `numpy` / `torch` / `lightgbm` / `sklearn`。
- **Deterministic**：
  - `torch.use_deterministic_algorithms(True)`，`torch.backends.cudnn.deterministic=True`，`cudnn.benchmark=False`。
  - LightGBM 设 `deterministic=True, force_row_wise=True, num_threads` 固定。
  - 环境变量 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，`PYTHONHASHSEED=0`。
- **复现性自检**：`code/src/verify_reproducibility.py` 连续两次执行推理，比对 `result.csv` 的 MD5，结果写入 `model/repro_check.log`。
- **数据截止**：训练与 baostock 拉取均止于 `2026-04-01`，保证与评测口径一致。
- **镜像体积**：通过 `apt` 清理、CPU torch、删除 ta-lib 源码，控制在 10 GB 以内。

---

## 11. 代码结构速查

```
code/src/
├── pipeline.py              # train / predict 顶层 CLI
├── config.py                # 超参数与路径
├── featurework.py           # 因子构建主流程
├── features/                # 单因子实现
├── data_prep/               # 数据加载、清洗、缓存
├── models/                  # master / stockmixer / lgb_de
├── cv/                      # walk-forward + embargo
├── ensemble/                # rank-blend 网格搜索
├── train.py / test.py       # 训练 / 推理具体步骤
├── predict.py               # 被 test.sh 间接调用
└── verify_reproducibility.py
```

提交文件：`Dockerfile`、`init.sh`、`train.sh`、`test.sh`、`readme.md`、`code/`、`data/`、`model/`。
