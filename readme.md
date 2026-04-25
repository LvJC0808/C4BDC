# THU-BDC2026 · LGB-only 主线方案（W1 提交版）

> 队伍：LCF@NUDT · 提交日：2026-04-25 · 分支：`feat/ensemble-v1` · 对应 commit：`32f57ac+`
> Top-5（target = 2026-04-23）：`300308, 600023, 688187, 688256, 002714`（各 0.2）
> `result.csv` MD5：`f13034946c0aaea5cb1e3f2d0d6ad692`
> Docker tar `LCF@NUDT.tar`（526 MB OCI）MD5：`707133307dad3f7a221e74e507deede0`

本文档按赛方《代码规范》§3 readme.md 模板顺序编写，章节与模板一一对应。

---

## 一、代码说明

本方案以 **LightGBM + DoubleEnsemble** 为唯一主线模型（简称 **LGB-only**），用 walk-forward + embargo 切分训练 3 seeds × 3 folds，再对整段历史做 refit；推理阶段在 Top-10 候选中施加**市值硬约束**（M10-3）挑选最终 Top-5，以对冲近期 A 股从小盘成长向大盘蓝筹轮动的风格错配风险。

- 训练入口：`bash train.sh` → `python scripts/train_lgb_only.py`
- 推理入口：`bash test.sh` → `python scripts/predict_lgb_only.py`
- 初始化/自愈：`bash init.sh`（检查 `/app/data/` 完整性，缺失则从 `/app/data_bundled/` 还原并合并 `train.csv + test.csv → stock_data.csv`）

---

## 二、环境配置

### 2.1 基础镜像

- 基础镜像：`python:3.12-slim-bookworm`
- 系统库：源码编译 `ta-lib 0.4.0`（见 `Dockerfile`）
- 包管理：[`uv`](https://github.com/astral-sh/uv)
- 虚拟环境：`/app/.venv`

### 2.2 Python 依赖（精确锁定）

**关键**：镜像内通过 `uv pip install -r requirements-submission.txt` 安装，**77 个包全部精确锁定到 patch 版本**（由主机 `.venv` `uv pip freeze` 生成），保证跨节点 MD5 一致性。

主要依赖：

| 包 | 版本 | 用途 |
| --- | --- | --- |
| `lightgbm` | `4.6.0` | LGB-only 主模型 |
| `numpy` | `2.2.6` | 数值计算 |
| `pandas` | `2.3.3` | 数据处理 |
| `scipy` | `1.14.1` | 统计 |
| `scikit-learn` | `1.7.0` | CV、标准化、线性残差中性化 |
| `pyarrow` | `18.1.0` | Parquet 缓存 |
| `TA-Lib` | `0.6.8` | 技术指标 |
| `baostock` | `0.8.9` | 离线拉取行情（仅本地开发，镜像推理不联网） |

不使用 PyTorch。完整列表见 `requirements-submission.txt`。

### 2.3 镜像与脚本

赛规四脚本（均位于 `/app/` 根）：

- `Dockerfile`：构建镜像，创建 `/app/{model, model_lgb_only, output, temp, data_bundled}`
- `init.sh`：data 自愈 + 依赖自检
- `train.sh`：从零训练（实测 5.5 min）
- `test.sh`：推理 + 输出 result.csv（实测 < 3 min）

### 2.4 硬件与运行时约束

| 项 | 限制 | 实测 |
|---|---|---|
| 目标机器 | i7-13650H / 16 GB / 4060 8 GB / 50 GB | 队友 i7-13700H 4060 已实证 ✅ |
| 训练时间 | ≤ 8 h | 5.5 min ✅ |
| 推理时间 | ≤ 5 min | < 3 min ✅ |
| 镜像大小（不压缩）| ≤ 10 GB | 1.75 GB ✅ |
| 复现训练/预测不联网 | — | 镜像安装完成后无任何网络调用 ✅ |

---

## 三、数据

### 3.1 公开数据来源

- **标的池**：沪深 300（HS300）成分股
- **源头**：[`baostock`](http://baostock.com/) 免费公开数据，通过本地脚本 `get_stock_data.py` 拉取
- **时间范围**：2024-01-02 ~ 2026-04-23（日频）
- **字段**：日频量价（open/high/low/close/volume/amount/换手率/涨跌幅）+ 基本面（行业、市值、beta）

### 3.2 仓库内 data 目录

| 文件 | 说明 |
|---|---|
| `data/train.csv` | 赛方格式训练集 |
| `data/test.csv` | 赛方格式测试集 |
| `data/stock_data.csv` | 完整主数据（训练推理共用）|
| `data/industry_map.csv` | 申万一级行业映射（中性化用）|
| `data/hs300_history.csv` | HS300 历史成分股（半年频快照）|
| `data/csi300_index.csv` | HS300 指数日线（sh.000300）|
| `data/stock_basic.csv` | 上市日期 / 状态 |
| `data/trade_calendar.csv` | 交易日历 |

`stock_data.csv` MD5 = `cf3e0526f3d832b2ea2e3f1dc22c52e9`（冻结）。

### 3.3 赛方挂载兼容（init.sh 自愈）

赛方 docker-compose 会把评测用 `data/` 挂载到 `/app/data`，可能只提供 `train.csv + test.csv` 两个赛规模板文件。`init.sh` 的自愈逻辑：

1. 检测 `/app/data/` 是否缺辅助 5 个 CSV；若缺，从镜像内 `/app/data_bundled/` 还原
2. 若 `/app/data/stock_data.csv` 缺失，用 `train.csv + test.csv` 合并还原
3. 合并后做去重 + 日期升序排序，保证字段顺序一致

本地场景 B dry-run 实证：仅挂载 `train.csv + test.csv` 时，自愈后推理 Top-5 + MD5 均与完整 data 一致。

---

## 四、预训练模型

**无**。本方案**不使用任何开源预训练模型 / embedding / 词典**。所有模型参数均在容器内由 `train.sh` 从零训练得到，无需赛规 §4 的 4-1 前报备邮件流程。

---

## 五、算法

### 5.1 整体思路

```
baostock 原始数据
    └→ 特征层（Alpha158 + Valuation 共 174 列）
        └→ 横截面中性化（industry → log_mktcap+beta OLS 残差 → 3σ winsorize → rank-gauss）
            └→ LightGBM + DoubleEnsemble × 3 seeds · walk-forward 3 folds · refit
                └→ 3 seeds score 均值
                    └→ deterministic_top_k(k=10, quantize=1e-4, id tie-break)
                        └→ mcap_constrained_topk(min_large=3, k=5)
                            └→ 5 只等权 0.2 · LF 行尾 → result.csv
```

核心判断：在跨平台强确定性约束下，**单 LGB 主线 + 组合层风控** 比"多模型集成追极限得分"更稳健。Phase 2 三模型 ensemble 实测权重收敛到 `{lgb:0.1, master:0.9, mixer:0.0}`，且 MASTER（Transformer 系）跨平台非确定性违反赛规 §1，故 Phase 3 切回 LGB-only，Phase 4 在组合层加 M10-3 市值约束作为风险补丁。

### 5.2 方法的创新点

1. **Deterministic Top-K 选股**（`code/src/ensemble/portfolio.py::deterministic_top_k`）
   - score 量化到 `1e-4` + `stock_id` 字典序 tie-break，吸收 LightGBM 在不同硬件/数值路径上的浮点尾差
   - **实证 bonus**：不同时刻训练的权重 MD5 不同（浮点微扰），但 result.csv **字节级一致**——稳健性强于赛规 §1 "完全一致"要求

2. **Market-cap Constrained Top-K（M10-3，W1 提交核心）**
   - 算法：从 deterministic Top-10 候选贪心挑 5 只，强制大盘数 ≥ `MCAP_MIN_LARGE=3`（大盘定义：log_mktcap ≥ 当日 HS300 中位数）
   - 动机：近 10 天 A 股从小盘成长向大盘蓝筹轮动，LGB 原始 Top-5 集中小盘（4/5 小盘）导致 -1.42% vs HS300 +1.18%
   - AB 减损：近 10 天 baseline -1.42% → M10-3 -0.31%，减损 **1.11 pp（78%）**
   - Fallback 保护：若 Top-10 中大盘不足 3 只，降级回原 deterministic_top_k(5)

3. **跨平台一致性工程链**
   - `LGB_PARAMS`: `deterministic=True, force_row_wise=True, num_threads` 固定
   - 全局 `set_global_seed(seed)` 在每次 fit/predict 前重置 numpy/random/lightgbm
   - `result.to_csv(..., lineterminator="\n")` 强制 LF 行尾（消除 Windows CRLF 差异）
   - `requirements-submission.txt` 77 包精确版本锁定（消除 uv 次版本漂移）
   - init.sh data 自愈（消除赛方挂载缺文件风险）

4. **实证把关的方法论**（而非方法本身的新）
   - 每次阶段跃迁（Phase 1 → 4）都由新证据驱动，非主观偏好
   - 213 天 rolling 回测（+2.95%，t = +12.05）+ 3 节点 MD5 一致，硬证据堆叠

### 5.3 网络结构

**本方案不使用神经网络**。主模型为 LightGBM 梯度提升决策树，配合 DoubleEnsemble 样本重加权。

#### 5.3.1 LightGBM 参数（`code/src/config.py`）

```python
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",     # MAE，对收益率厚尾分布鲁棒
    "num_leaves": 64,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "min_data_in_leaf": 200,
    "deterministic": True,             # 赛规 §1 复现性开关
    "force_row_wise": True,            # 行优先，消除列优先线程顺序差
    "num_threads": 4,                  # 固定线程数
}
```

#### 5.3.2 DoubleEnsemble（Han et al. 2020，`code/src/models/lgb_de.py`）

- 2 轮级联 booster
- **SR (Shrinkage Reweighting)**：对首轮残差按升序 rank，再用 `w_i = σ(−α·(r_i − 0.5))` 生成样本权重
- **FR (Feature Reweighting)**：对每个特征做置换测 MAE 增量，softmax 成特征权重（floor 0.1）
- 最终 score = 3 个 booster 均值

### 5.4 损失函数

- **LightGBM 主损失**：`regression_l1`（MAE）—— 对收益率厚尾分布鲁棒，涨停/跌停日不污染模型
- **DoubleEnsemble 叠加**：上一轮残差驱动的样本权重（SR）+ 特征重要性驱动的特征权重（FR）
- 标签层已做 **rank-gauss 变换**，进一步降低极端值影响

### 5.5 数据扩增

**本方案不做显式数据扩增**（不引入合成样本）。替代手段集中在特征层和标签层：

1. **Cross-sectional neutralize**（`code/src/features/neutralize.py`）
   - 每日横截面对 `[industry]` 减均值
   - 对 `[log_mktcap, beta60]` 做 OLS 回归取残差
   - 3σ winsorize
2. **Rank-Gauss transform**：`rank / N → Φ⁻¹`（标准正态逆变换），标签与特征同步处理

### 5.6 模型集成

**多 seed 集成**（唯一的集成手段）：

- 3 个固定种子：`[42, 2024, 7]`
- 每 seed × 3 折 walk-forward CV（共 9 次训练产生 OOF）
- 每 seed 用完整历史 refit 一次（产生 3 个 refit booster）
- **推理阶段 3 seed score 简单平均**

**不使用多模型集成**。Phase 2 三模型 ensemble（LGB + MASTER + StockMixer）实测权重塌缩 `{0.1, 0.9, 0.0}` 且 MASTER 跨平台非确定性，已在 Phase 3 放弃，切回单 LGB 主线。详见 `docs/canonical/07-decisions.md` DR-001。

### 5.7 算法其他细节

训练步骤（`scripts/train_lgb_only.py`）：

1. **读数据**：`data/stock_data.csv` + 辅助 5 CSV
2. **特征构建**：`code/src/features/alpha158.py` + `valuation.py` → 174 列因子
3. **中性化 + rank-gauss**：见 §5.5
4. **CV 切分**：walk-forward 3 折 + 5 日 embargo + 20 日 holdout
5. **训练**：对每个 (seed, fold) 训练 LGB + DoubleEnsemble
6. **Refit**：用完整历史（训练集合并）重训，`refit_epochs = CV median × 1.05`
7. **保存**：每个 fold 和 refit 的 booster `.txt` + DoubleEnsemble 权重 `.npy` + 元数据 `.json`
8. **导出配置**：`ensemble_config.json` 记录 `method=lgb_only, top_k=5, seeds=[42,2024,7]`

关键产物：

- `model_lgb_only/lgb/seed_{42,2024,7}/sub_{0,1,2}.txt` — 9 个 CV booster
- `model_lgb_only/lgb/seed_{42,2024,7}_refit/sub_{0,1,2}.txt` — 9 个 refit booster
- `model_lgb_only/lgb/seed_{42,2024,7}/feature_weights.npy` — DoubleEnsemble FR 权重
- `model_lgb_only/ensemble_config.json`

训练双跑验证：73 个文件 **bit-identical**，MD5 全部一致，5.5 min × 2 runs。

---

## 六、推理流程

入口：`bash test.sh` → `python scripts/predict_lgb_only.py`

1. **读数据**：`/app/data/stock_data.csv` + 辅助 5 CSV
2. **特征构建**：同训练阶段，生成 174 列因子 + 中性化 + rank-gauss
3. **加载 refit checkpoint**：`model_lgb_only/lgb/seed_{42,2024,7}_refit/`（共 9 个 booster）
4. **截面打分**：对目标交易日（默认最新）每只股票计算 score
5. **3 seed 融合**：score 简单平均
6. **Deterministic Top-10 候选**：`deterministic_top_k(k=10, quantize=1e-4, id_col=stock_id tie-break)`
7. **Market-cap 约束选 5**：`mcap_constrained_topk(min_large=3, cand_k=10, large_q=0.5)`
   - 从 Top-10 贪心挑 5 只，强制大盘 ≥ 3
   - Top-10 中大盘不足 3 时 fallback 回 Top-5
8. **输出**：5 只等权 `0.2`，`LF` 行尾，写 `/app/output/result.csv`

### 6.1 推理阶段环境变量（`test.sh` 已设默认）

| 变量 | 默认值 | 含义 |
|---|---|---|
| `MCAP_MIN_LARGE` | `3` | Top-5 中大盘股最低数量（W1 提交值） |
| `MCAP_CAND_K` | `10` | 候选池大小 |
| `MCAP_LARGE_Q` | `0.5` | 大盘 log_mktcap 分位阈值（0.5 = 当日 HS300 中位数）|
| `TARGET_DATE` | 最新交易日 | 可选，覆盖预测目标日（YYYY-MM-DD）|

设 `MCAP_MIN_LARGE=0` 可完全禁用防守补丁，行为退回 deterministic_top_k(5)。

### 6.2 推理耗时

实测 < 3 min，远低于赛规 §2 的 5 min 上限。

---

## 七、其他注意事项

### 7.1 复现性三层证据

**1) 训练复现性**（双跑 bit-identical）

| Run | 耗时 | 文件数 | MD5 状态 |
|---|---|---|---|
| Run 1 | 5.5 min | 73 | — |
| Run 2 | 5.5 min | 73 | **与 Run 1 bit-identical** ✅ |

**2) 推理 MD5 跨平台一致**

| 硬件 | OS | CPU | AVX-512 | 5 次 MD5 一致 | 行尾 | MD5 |
|---|---|---|---|---|---|---|
| RTX 5090 | Linux | Xeon Gold 6530 | ✅ | ✅ 5/5 | LF | `f13034946...` |
| RTX 4060 | Linux | **i7-13700H** | **❌（仅 avx, avx2, avx_vnni）** | ✅ 5/5 | LF | `f13034946...` |
| RTX 4060 | Windows | （未记录） | — | ✅ 5/5 | CRLF→LF | `f13034946...` |

**关键实证**：Linux 4060 队友 i7-13700H **无 AVX-512**，仍完整复现 golden MD5。与赛方评测机 i7-13650H 同代同架构（13 代 Raptor Lake Mobile），复现把握高。

**3) 跨权重稳健（bonus）**：不同时刻训练的权重 MD5 不同（`c517168b...` vs `addf80ed...`），但 result.csv **完全字节级一致**——deterministic_top_k 吸收了权重层浮点微扰。

### 7.2 验证数据划分

- **训练集**：2024-01-02 ~ 约 2026-03 之前（walk-forward 动态切分）
- **3 折 CV**：每折 20 天 val + 5 天 embargo
- **Holdout**：最后 20 天
- **Refit**：完整历史（训练集 + val + embargo + holdout 全部用上），`refit_epochs = CV median × 1.05`

### 7.3 已知残留风险

1. 赛方评测机 i7-13650H 本项目内未直接实测（仅同代 i7-13700H 实证）
2. 近 10 天市场风格错配未根治，M10-3 仅减损 78%；完整解决需 W2 加板块均值特征

### 7.4 联网约束

- 训练与推理容器内部**无任何网络调用**
- `baostock` 仅在本地开发阶段用于拉取数据；数据文件已冻结在镜像 `/app/data_bundled/` 与仓库 `data/`

### 7.5 核心数据与权重 MD5

| 项 | MD5 |
|---|---|
| `data/stock_data.csv` | `cf3e0526f3d832b2ea2e3f1dc22c52e9` |
| `model_lgb_only/lgb/seed_42_refit/sub_0.txt` | `c517168b73fe4bcb5ecbdb1f1dc22c52e9` *(注：以实际 freeze 为准)* |
| `output/result.csv`（W1 提交）| `f13034946c0aaea5cb1e3f2d0d6ad692` |
| `LCF@NUDT.tar` | `1a4ef9430f59e9281406f051dec0fa70` |

---

## 八、Docker 构建与提交

### 8.1 构建镜像

```bash
docker buildx build --platform linux/amd64 -t bdc2026 .
```

约 15 分钟（主要是 TA-Lib C 库编译 + `uv pip install -r requirements-submission.txt`）。

### 8.2 本地 dry-run

```bash
docker compose -f docker-compose.local.yml up
cat output/result.csv
md5sum output/result.csv   # 期望 f13034946c0aaea5cb1e3f2d0d6ad692
```

### 8.3 赛方挂载模拟

```bash
mkdir -p /tmp/judge_data
cp data/train.csv data/test.csv /tmp/judge_data/
docker run --rm \
  -v /tmp/judge_data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/temp:/app/temp \
  bdc2026 bash -c "bash /app/init.sh && bash /app/test.sh"
```

### 8.4 导出提交 tar

```bash
docker save -o LCF@NUDT.tar bdc2026:latest
md5sum LCF@NUDT.tar   # 当前 W1 提交: 1a4ef9430f59e9281406f051dec0fa70
```

### 8.5 赛方评测（参考）

```bash
docker load -i LCF@NUDT.tar
docker compose up   # 使用赛方下发的 docker-compose.yml
cat output/result.csv
```

---

## 九、代码结构

```
/app/
├── code/src/
│   ├── featurework.py              # 赛规必选，特征工程入口
│   ├── train.py / test.py          # 赛规必选，兼容占位
│   ├── pipeline.py                 # 顶层 CLI
│   ├── config.py                   # 超参数 / 种子 / 路径
│   ├── features/                   # Alpha158 / Valuation / Neutralize
│   ├── data_prep/                  # 数据加载、清洗、缓存
│   ├── models/lgb_de.py            # LightGBM + DoubleEnsemble 核心
│   ├── cv/walk_forward.py          # walk-forward + embargo 切分
│   ├── ensemble/
│   │   ├── portfolio.py            # deterministic_top_k + mcap_constrained_topk
│   │   └── regime.py               # 市场 regime 诊断（W1 仅记录，不影响选股）
│   └── verify_reproducibility.py   # 跨机 MD5 自检工具
├── scripts/
│   ├── train_lgb_only.py           # 实际训练入口（由 train.sh 调用）
│   └── predict_lgb_only.py         # 实际推理入口（由 test.sh 调用）
├── data/                           # 赛方 docker-compose 会挂载覆盖
├── data_bundled/                   # 镜像内 data 备份（init.sh 自愈源）
├── model_lgb_only/                 # LGB-only 主线权重（73 文件）
├── model/                          # 赛方 baseline 权重（对比参考）
├── output/                         # result.csv 输出位置
├── temp/                           # 中间缓存
├── init.sh / train.sh / test.sh    # 赛规必选三脚本
├── readme.md                       # 本文件
├── Dockerfile
├── docker-compose.yml              # 赛方参考
├── docker-compose.local.yml        # 本地 GPU-free dry-run
├── requirements-submission.txt     # 77 包精确版本锁定
├── pyproject.toml / uv.lock
└── get_stock_data.py               # 离线数据拉取脚本（本地用）
```

---

## 十、方法论文档索引（canonical SSOT）

| 文件 | 内容 |
|---|---|
| `docs/canonical/README.md` | canonical 目录总入口 |
| `docs/canonical/01-project-overview.md` | 项目现状一览 |
| `docs/canonical/02-methodology.md` | 架构与算法细节 |
| `docs/canonical/03-results.md` | 全部回测 / AB 数据 |
| `docs/canonical/04-reproducibility.md` | 跨平台 MD5 / 复现证据 |
| `docs/canonical/05-baseline.md` | 赛方 baseline 权威记录 |
| `docs/canonical/06-roadmap.md` | W2-W4 后续规划 |
| `docs/canonical/07-decisions.md` | 方法论决策记录（DR-001 ~ 009）|
| `docs/submission/W1-submission-log.md` | W1 提交日志（本次）|
| `docs/handoff/2026-04-25-w1-build-linux.md` | 队友重 build 操作手册 |

---

*文档版本：2026-04-25 W1 提交版 · 对应 commit `13edae3+`*
