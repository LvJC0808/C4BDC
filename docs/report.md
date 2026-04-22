# BDC2026 阶段性报告

> 2026 中国高校计算机大赛—大数据挑战赛
> 方案代号 **A4+ / B2 / D3 / E4 / F3 / Z1**
> 分支 `feat/ensemble-v1`，截至 2026-04-22

---

## 摘要

我方针对沪深 300 股价收益预测赛题构建了基于 **LightGBM+DoubleEnsemble / MASTER / StockMixer** 的三模型集成系统。在 2025-11-03 至 2026-03-13 共 **87 个交易日**的严格滚动回测中，组合 5 日开盘-开盘收益率均值达 **+1.01%**，显著优于等权 HS300 baseline 的 +0.11%（**t=+4.78, p<0.0001**）。胜率 **75.86%**，跑赢 baseline 天数占比 **70.11%**。

---

## 1. 赛题与约束

| 项 | 要求 |
|---|---|
| 预测目标 | 沪深 300 成分股 T+1 开盘买入、T+5 开盘卖出的 Top-K 组合收益 |
| 输出 | `result.csv`：`stock_id, weight`；≤5 只；权重和 ≤1（现金补足） |
| 硬件 | i7-13650H / 16 GB / RTX 4060 8 GB |
| 训练时长 | ≤ 8 h |
| 推理时长 | ≤ 5 min |
| 运行时 | **离线**（无网络）、Docker 镜像 ≤ 10 GB |
| 可复现 | 固定随机种子两次训练/推理权重与输出 MD5 一致 |
| 数据截止 | 2026-04-01 前公开资源 |

---

## 2. 整体架构

```
data/                           # baostock 离线爬取（Z1 扩展，~85 MB）
├── stock_data.csv              # 行情+估值
├── industry_map.csv            # 申万行业
├── hs300_history.csv           # 动态成分股
├── csi300_index.csv            # 市场指数
├── stock_basic.csv / trade_calendar.csv
       │
       ▼
code/src/features/              # B2 + Z1 特征工程
├── alpha158.py  ← Qlib Alpha158（TA-Lib）
├── alpha360.py  ← Qlib Alpha360（OHLCV×60d）
├── valuation.py ← PE/PB/PS/PCF + 市场 63 维 + Beta60 + log_mktcap
├── neutralize.py← 行业/市值/Beta 中性化 + rank-gauss
└── build.py     ← 统一入口，parquet 缓存
       │
       ▼
code/src/cv/walk_forward.py     # E4：3 段 walk-forward + 5 天 embargo + 20 天 holdout
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
 models/lgb_de.py  models/master.py models/stockmixer.py
 (Alpha158+估值)    (Alpha158+市场63维) (Alpha360 原始)
       │             │             │
       └─────────────┴──────┬──────┘
                            ▼
code/src/ensemble/          # D3：rank-blend + 置信度自适应仓位
├── blender.py   ← 横截面 rank 归一 → holdout 单纯形网格权重
└── portfolio.py ← Top-K + confidence 动态仓位
                            ▼
                    output/result.csv
```

**单一入口**：`code/src/pipeline.py {train|predict}`，另有 `train.sh / test.sh` 薄包装。

---

## 3. 算法亮点

### 3.1 三模型集成（A4+）

| 模型 | 输入 | 结构要点 | 训练规模 |
|---|---|---|---|
| **LightGBM + DoubleEnsemble** | Alpha158 + 估值，中性化 | L1 回归，3 轮样本/特征重加权（SR+FR） | 3 folds × 3 seeds |
| **MASTER**（AAAI 2024） | Alpha158 + 市场 63 维 | FiLM 市场引导门控 → 股内 attn → 股间 attn | 3 folds × 3 seeds |
| **StockMixer**（AAAI 2024） | Alpha360 原始 OHLCV | Time / Feature / Stock 三路 MLP-Mixer | 3 folds × 3 seeds |

融合方式：每模型对横截面做 rank 归一至 [0,1] → 跨模型加权求和；**holdout 上单纯形网格搜索最优权重** + Top-K + α。

**实际搜索结果**：`{lgb: 0.1, master: 0.9, mixer: 0.0}`, `top_k=4`, `α=0.7`。

### 3.2 特征工程（B2 + Z1）

- **Alpha158**：9 K 线 + 4 价比 + 5 窗口 × 29 技术因子，TA-Lib 向量化
- **Alpha360**：过去 60 日 × 6 字段（OHLCV + amount）除以 close_t 归一化
- **估值因子**：peTTM / pbMRQ / psTTM / pcfNcfTTM + 20/60 日 z-score + 行业内 rank
- **中性化**（每日横截面独立执行）：
  1. 行业内减均值（申万一级）
  2. log(成交额代理市值) + Beta60 线性回归取残差
  3. 3σ winsorize
  4. 横截面 rank → Φ⁻¹ 得 rank-gauss

标签同样做 rank-gauss，保证模型学习的是**相对排名**而非绝对收益，抵御市场整体 beta 冲击。

### 3.3 验证策略（E4）

```
段1: train[D_start .. D_end−150]  embargo 5  val[D_end−144 .. D_end−125]
段2: train[D_start .. D_end−100]  embargo 5  val[D_end− 94 .. D_end− 75]
段3: train[D_start .. D_end− 50]  embargo 5  val[D_end− 44 .. D_end− 25]
holdout:                                         [D_end− 20 .. D_end]
```

- **5 天 embargo**：标签跨 5 日必不可少，否则验证集泄漏
- **3 seed 集成**：`[42, 2024, 7]`，每段每 seed 独立重训
- **refit epochs = CV median × 1.05**：全量重训时用 CV 得到的经验 epoch 数
- **holdout 只用一次**：集成权重、Top-K、α 网格搜索

### 3.4 置信度自适应仓位（D3）

```
c_disp  = clip((Top-K 平均 rank − 0.5) / 0.5, 0, 1)
c_ic    = clip(rolling_RankIC_last_20d / 0.05, 0, 1)
c_agree = |三模型 Top-K 交集| / K
c       = 0.4·c_disp + 0.4·c_ic + 0.2·c_agree
total_position = 0.5 + α · 0.5 · c         # α = 0.7
w_i     = total_position / K                # Top-K 内部等权
```

**直觉**：模型越自信（分散度高、近期 IC 强、三模型共识强）就上越重仓位；否则留现金。Baseline 无仓位调节。

### 3.5 DoubleEnsemble 实现要点

- **SR (Shrinkage Reweighting)**：残差 L 升序 rank → `w_i = σ(−α·(r_i−0.5))`，让**低残差（易学）样本**获得更高权重，后续 booster 聚焦稳定结构而非噪声
- **FR (Feature Reweighting)**：逐特征 shuffle 置换，测量 MAE 增量 Δ_j；softmax 成特征权重，floor 0.1 避免零化
- **3 轮级联**，最终分数为三个 booster 的均值

---

## 4. 实验结果

### 4.1 滚动回测（N=87 交易日）

**设置**：2025-11-03 ~ 2026-03-13 每个交易日 T 做一次预测，按生成的组合 T+1 开盘买入 / T+5 开盘卖出，记录当日 score。**训练数据截止远早于回测窗口，无泄漏**。

| 指标 | 我方 | 等权 HS300 | 5 日动量 Top-K |
|---|---|---|---|
| **平均 5 日收益** | **+1.01%** | +0.11% | −0.76% |
| **中位数** | +0.79% | +0.31% | −1.00% |
| **标准差** | 1.54% | 1.45% | 6.28% |
| **胜率 (>0)** | **75.86%** | 60.92% | 45.98% |
| **vs 等权胜天比** | **70.11%** | — | — |

### 4.2 显著性检验

| 对比 | 统计量 | p 值 | N |
|---|---|---|---|
| 我方 − 等权 HS300 | **t = +4.78** | **p < 0.0001** | 87 |
| 我方 − 动量 Top-K | t = +2.67 | p = 0.009 | 87 |

### 4.3 极值样本

| 类型 | 日期 | 组合收益 | 选股 |
|---|---|---|---|
| 最佳 | 2026-02-13 | +5.01% | 600039 / 000538 / 000708 / 002001 |
| 最佳 | 2026-02-10 | +4.91% | 000538 / 000708 / 688009 / 300394 |
| 最差 | 2026-01-27 | −2.90% | 300408 / 300014 / 600926 / 688187 |
| 最差 | 2025-11-19 | −2.28% | 000895 / 601919 / 000708 / 000166 |

最大回撤日 −2.90%，最佳日 +5.01%，**尾部偏正（正偏度）**。平均置信度 0.767，仓位平均约 76.7% × α = 0.5 + 0.7×0.5×0.767 ≈ **0.77**。

### 4.4 推理一致性

三模型独立给分 → rank-blend → confidence → Top-K，从特征到 `result.csv` **全流程 deterministic**：`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`LightGBM deterministic=True, force_row_wise=True`。

---

## 5. 工程交付

### 5.1 Dockerfile

- 基础：`python:3.12-slim-bookworm`
- 编译安装 TA-Lib C 库 + `uv` 快速装 Python 依赖
- 关键包：`torch 2.10 (cu128)`, `lightgbm 4.6`, `pyarrow 24`, `TA-Lib`, `scikit-learn`
- 镜像估算 **~3.5 GB + 模型 0.5 GB + 数据 0.1 GB ≈ 4 GB** ✅（限额 10 GB）

### 5.2 Shell 脚本

```bash
# init.sh  ── 依赖自检
python -c "import lightgbm, torch, talib, numpy, pandas; print('deps OK')"

# train.sh ── 一键训练
python pipeline.py train --data_path /app/data --model_dir /app/model --temp_dir /app/temp

# test.sh  ── 一键推理
python pipeline.py predict --data_path /app/data --model_dir /app/model --temp_dir /app/temp \
                           --output_path /app/output/result.csv
```

### 5.3 可复现自检

`code/src/verify_reproducibility.py`：连跑两次 predict，MD5 对比 `result.csv`，结果写 `model/repro_check.log`。

### 5.4 预算对账

| 阶段 | 预算 | 实测 |
|---|---|---|
| 特征工程（一次性） | 0.2 h | **0.04 h** (2.5 min) |
| 训练（3 模型 × 3 fold × 3 seed + refit） | 8 h | **6.3 h** |
| 推理 | 5 min | **<3 min** |
| 镜像大小 | 10 GB | ~4 GB |

---

## 6. 数据来源与合规（7/18 前报备清单）

| 资源 | URL | 状态 |
|---|---|---|
| baostock | github.com/baostock/baostock | 离线快照 |
| Qlib (Alpha158/360) | github.com/microsoft/qlib | commit 固定 |
| MASTER | github.com/SJTU-DMTai/MASTER | commit 固定 |
| StockMixer | github.com/SJTU-DMTai/StockMixer | commit 固定 |
| LightGBM / TA-Lib | 官方 release | 版本固定 |
| 本队爬取 6 CSV | 不上传，逐文件 MD5 | 已记录 |

所有数据**截止 2026-04-01 前公开**，符合赛规。

---

## 7. 未竟事项与风险

| 项 | 现状 | 缓解 |
|---|---|---|
| Ensemble 权重偏向 master (0.9) | holdout 10 天搜索，可能过拟合 | 计划引入 CV 内 IC 作为次级加权约束 |
| 样本仍 N=87 | 非万级，可能低估尾部风险 | 继续扩窗到 120~200 天 |
| 未建模交易摩擦 | 涨停不能买、冲击成本、手续费全部忽略 | 赛题评测同样忽略，但实盘需加 |
| 动态成分股 | hs300_history 半年一次快照 | 已在特征阶段做样本池过滤 |

---

## 8. 关键超参汇总

| 模块 | 关键超参 |
|---|---|
| LightGBM | `num_leaves=64, lr=0.02, feature_fraction=0.7, bagging=0.7, min_data=200, objective=regression_l1` |
| DoubleEnsemble | `n_rounds=3, α_sr=1.0, β_fr=1.0, floor=0.1` |
| MASTER | `lookback=8, d_model=256, n_head=4, dropout=0.5, β=5, epochs=40, lr=1e-4, patience=10` |
| StockMixer | `lookback=16, hidden=128, scale=3, market_scale=3, epochs=40, lr=1e-4` |
| Loss (NN) | `0.5·MSE + 0.5·(1 − BatchRankIC) + 0.1·TopK` |
| CV | `folds=3, embargo=5, val_days=20, holdout_days=20, seeds=[42,2024,7]` |
| Ensemble | `weights={lgb:0.1, master:0.9, mixer:0.0}, top_k=4, α=0.7, min_pos=0.5` |

---

## 9. 关键 Commit 与交付物

| Task | Commit | 说明 |
|---|---|---|
| 设计文档 | `474fe45` | `specs/2026-04-21-bdc2026-ensemble-design.md` |
| 数据抓取 | `4325def` | `data_prep/fetch_all.py` (Z1 扩展) |
| 特征工程 | `a94e844 / f70f816 / 35b4c95 / a79531e / 87c1256` | 158/360/估值/中性化/统一入口 |
| CV | `aea6804` | walk-forward + embargo |
| 模型 | `ba3eea7 / b06aec7 / 6f00747` | LGB+DE / MASTER / StockMixer |
| 集成 | `22eac2c / 1ab7356` | blender / portfolio |
| Pipeline | `bdcacc3` | train/predict 总入口 |
| Docker & 交付 | `451761e` | Dockerfile + 脚本 + readme + 复现自检 |
| 修复 | `ea96a3d / b08589f` | 短序列跳过、NaN-safe grid search、滚动回测 |

全部位于 `feat/ensemble-v1` 分支。

---

## 10. 结论

- **pipeline 端到端跑通**，Docker 包可直接交付
- **87 天滚动回测 p<0.0001 显著跑赢等权 baseline**，超额收益 ~90 bp / 5 日
- **训练 6.3 h、推理 <3 min、镜像 ~4 GB**，全部在赛题硬约束内
- 下一步聚焦：扩大回测窗口、稳健化集成权重、冲击成本敏感性分析、最终镜像打包上传
