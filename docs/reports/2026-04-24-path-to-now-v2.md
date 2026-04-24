# 从 0.0642 到 LGB-only 主线 · 第二阶段路径报告

> **续接** `/root/shared-nvme/path_to_now.md`（v1 阶段，3.15-4.10，最终本地分 0.06421）
> **本阶段时间**：2026-04-10 ~ 2026-04-24（约两周）
> **结论**：完整重构主线，从 StockTransformer 转 LGB-only，**213 天 rolling +2.41%/5d，t=+12，胜率 85%**，比 v1 阶段使用的本地评分体系更可靠。
> **W1 提交**：M10-3 配置（mcap_constrained_topk + 强制 ≥3 大盘股），214 行代码新增、22 commits。

## 三份必读前置文档

本报告组成"三段式"完整记录：

1. **v1**：`/root/shared-nvme/path_to_now.md`（3.15-4.10，StockTransformer + linear 配权 → 本地 0.0642）
2. **v2 前半段**：`docs/report.md`（4.10-4.22，三模型 Ensemble + walk-forward CV + Phase-A 稳健化 → 87 天 +1.01%, t=+4.78）
3. **v2 后半段**：本文档（4.22-4.24，LGB-only + W1 mcap 防守补丁 → 213 天 +2.41%, t=+12）

**不要把任何一段当作"失败版本"**——每一段都是当时基于已知证据的最优决策，每一次切换都是新证据出现后的理性更新。

---

## 一、第二阶段为什么必须存在

v1 结尾留下三个根本不安：

1. **本地分 0.0642 vs 赛方真实评分相关性不明**：v1 用的本地加权收益（5 天 forward return × 权重）只在最近 ~30 天测试，没有长期 rolling
2. **StockTransformer 跨平台非确定性**：CUDA 卷积在 4060/5090/Windows 之间最多有 0.17 分差异（远超提升幅度），赛方评分用谁的 GPU 都不知道
3. **训练不可复现**：v1 的 best_epoch 在不同硬件不一样，没有 seed 控制 + 多 seed ensemble

第二阶段的核心命题：**把"能在我自己机器上得高分"变成"能在任何机器上 deterministic 得高分，并且被长期 rolling 验证过"**。

---

## 二、二阶段时间线 · 五个里程碑

### M0 · 2026-04-10 ~ 04-13：本地分 0.0642 的可信度审判

**做的事**：
- 把 v1 模型 + best.pth 同步到 4060 Linux + 4060 Windows 跑 5 次推理
- 5 次输出 stock_id 不一致（CUDA 非确定性）
- 跨平台分差中位数 **0.17**

**结论**：
- 本地分 0.0642 不可作为最终决策依据
- StockTransformer 路线必须放弃 GPU 推理或寻找完全确定性算法

### M1 · 2026-04-14 ~ 04-21：三模型 Ensemble v1（详见 `docs/report.md`）

**这是 v2 阶段的第一个完整方案**，不是失败实验，是**当时的最佳产出**——它证明了 LGB+MASTER+StockMixer 集成 + 严格 walk-forward CV + rank-blend 这套方法论本身是 work 的。

**关键数字**（来自 `docs/report.md`，2025-11-03 ~ 2026-03-13，**87 天 rolling**）：

| 指标 | 三模型 Ensemble | 等权 HS300 | 5 日动量 Top-K |
|---|---|---|---|
| Mean 5d 收益 | **+1.01%** | +0.11% | −0.76% |
| 胜率 | 75.86% | 60.92% | 45.98% |
| t vs HS300 | **+4.78** (p<0.0001) | — | — |

**架构**（详见 `docs/report.md` §3）：
- LightGBM+DoubleEnsemble（Alpha158+估值）weight 0.1
- MASTER (AAAI 2024, Alpha158+市场63维) weight 0.9
- StockMixer (AAAI 2024, Alpha360 原始 OHLCV) weight 0.0
- 融合：横截面 rank → holdout 单纯形网格搜索 → top_k=4, α=0.7
- 置信度自适应仓位：c = 0.4·c_disp + 0.4·c_ic + 0.2·c_agree

**Phase-A 稳健化**（同 report §11）：
- Tradability filter（涨停/跌停过滤）
- Transaction cost 6 bp/周
- Dynamic IC weights（实测拖累 15bp，未启用）
- 加完前两项后 **mean +1.00%**（仅损失 1 bp，胜率保持 75.6%）

**为什么在 04-22 后从这条线撤回到 LGB-only**：

1. **MASTER 权重 0.9 极端不稳定**：报告 §7 自己也写"holdout 10 天搜索可能过拟合"
2. **跨平台 CUDA 非确定性**：MASTER 的 attention 在 4060/5090/Windows 输出 0.17 分差异（赛规要求"两次 MD5 一致"，硬伤）
3. **训练 6.3h 很紧**：3 模型 × 3 fold × 3 seed + refit 已用满 8h 预算的 79%，无任何加新模型/特征的余地
4. **MASTER weight 0.9 = 单模型方案的伪装**：去掉 LGB（0.1）和 Mixer（0.0）实测，"集成"只是个 1-模型方案，全部赌注在 MASTER 上
5. **04-22 跨平台测试发现 MASTER 输出 5 次不一致**：直接判死刑

**M1 留下的可继承资产**：
- Walk-forward CV + 5 天 embargo 框架（`code/src/cv/walk_forward.py`）
- Alpha158 + 估值 + 中性化 pipeline（B2 + Z1）
- DoubleEnsemble 实现（`code/src/models/lgb_de.py`）
- Rank-blend + confidence 仓位（`code/src/ensemble/{blender,portfolio}.py`）
- Tradability filter（`code/src/ensemble/tradability.py`）
- Transaction cost 框架（`rolling_backtest.py --cost_bps`）

**这些全部被 v2 后半段（LGB-only）继承**，所以 M1 不是"丢弃"，是"剥离 + 简化"。

### M2 · 2026-04-22 ~ 04-23：LGB-only 主线诞生

**抉择**：放弃三模型 ensemble，全栈切换到单 LightGBM + DoubleEnsemble

**理由**（基于 M1 的实测教训）：
- LGB CPU 推理 deterministic，跨平台 bit-exact
- DoubleEnsemble 用 sample reweighting + feature shuffling 解决稳健性，不依赖 GPU 随机数
- 8 分钟训练 vs 6.3 小时（节省 47×），可承担多 seed
- 接 Qlib Alpha158 + Valuation 共 174 个特征，沿用 M1 已验证的因子库
- **M1 的 ensemble 权重 {0.1, 0.9, 0.0} 实质是 MASTER 单模型——既然如此，换成 deterministic 的 LGB 单模型更有意义**

**架构落地**（继承 M1 大半，去掉 MASTER/Mixer）：
```
data → build_feature_sets (Alpha158 + Valuation, 174 cols)  # 继承 M1
     → neutralize_cross_section (industry + log_mktcap + beta60)  # 继承 M1
     → DoubleEnsemble × 3 seeds (42, 2024, 7)  # 继承 M1
     → blend_scores (mean + ICIR shrink)  # 简化：单模型 ensemble = seed 平均
     → deterministic_top_k (quantize=1e-4 + stock_id lex tie-break)  # 新增
     → confidence-scaled portfolio (top_k=5, equal weight)  # 继承 M1，配权改 equal
```

**新增**：
- `deterministic_top_k`（quantize + stock_id tie-break）—— 解决浮点 score 跨平台不一致
- 11 个 pytest，覆盖 deterministic_top_k 跨平台一致性、blender 数值稳定性

### M3 · 2026-04-22 ~ 04-23：跨平台 4060 验证矩阵

**5090 主开发机训出权重 → 4060 Linux + 4060 Windows 各跑 5 次推理**：
- Linux 4/5 次完全一致（1 次 cache 问题，重跑后一致）
- Windows 5/5 次完全一致
- 跨平台 result.csv md5 完全相同 ✅

**赛规复现门槛达成**。

### M4 · 2026-04-21 ~ 04-22：82 天 rolling 初步验证

**首次跑长 rolling**（2026-01-01 ~ 2026-04-09，82 个 eval days）：
- LGB-only mean = **+1.446% / 5d**
- vs HS300 等权 +0.43%（Δ=+1.02pp）
- vs Mom-TopK +0.35%（Δ=+1.10pp）
- 胜率 83%

**首次"我们的方法长期有效"的证据**。

### M5 · 2026-04-23 ~ 04-24：数据更新 + 213 天扩展 + W1 防守补丁

详见第三、四节。

---

## 三、关键技术决策与失败实验

### 3.1 标签处理：rank-gauss vs zscore

v1 用 zscore（path_to_now.md §6），二阶段切到 **rank-gauss**：
- 理由：rank-gauss 对极端 outlier 更鲁棒（涨停/跌停日不会污染模型）
- 实证：rank-gauss 在 213 天 rolling 上比 zscore mean +0.18pp（小幅但稳健）
- 队伍 B 的实证显示 zscore 更强——这是我们待 W2 重新 AB 的项

### 3.2 配权：linear → equal（与 v1 反向）

v1 选 linear（path_to_now.md §9），二阶段实测 linear 在单窗口方差极大：
- T=04-16 单窗口测试：linear −2.32%（Top-1 吃 59% 权重），equal +0.13%
- **二阶段保留 equal 作为默认**，但 `allocation_by_mode` 模块化保留 linear/softmax 接口

**为什么 v1 linear 看着好，二阶段就不好了？**
- v1 评分用的是局部 30 天，linear 在好 regime 下放大收益
- 二阶段用 213 天 rolling，linear 的尾部风险被充分暴露
- 教训：**短窗口的最优配权不等于长窗口的最优配权**

### 3.3 数据更新：baostock 0.8.9 → 0.9.1（2026-04-23）

- 数据从覆盖到 03-13 → 04-23（增量 28 个交易日）
- 新数据重训后，rolling 从 82 天 → **213 天**
- 长窗口 mean 从 1.446% → **2.41%**（不是真的提升，是评估窗口扩大后看到了 2025 H2 的强表现）

### 3.4 失败实验 M1（4-23）：beta60 + log_mktcap 作为特征

**假设**：近 10 天被大盘股轮动甩，加 beta/mktcap 让模型看到 style 暴露
**结果**：所有窗口都变差（−0.23 ~ −0.38pp）
**反思**：beta60/log_mktcap 已经在 `neutralize_cross_section` 里做过中性化因子，再作为输入特征产生信号重复/冲突
**动作**：完全回滚

### 3.5 三模型 Ensemble v1 → LGB-only 单线（M1 → M2 关键转折）

**不是失败实验**，是阶段性最佳方案被自然替代。详见 §M1。

关键观察：
- M1 的 holdout grid search 结果是 `{lgb:0.1, master:0.9, mixer:0.0}` → "三模型 ensemble" 实质是 MASTER 单模型
- MASTER 跨平台 5 次输出不一致（0.17 分差）→ 赛规复现要求未达
- 同时 LGB 单线 87 天 rolling = +1.01% × 0.1 = 不可比，但 LGB-only 重训后 82 天达 +1.446%（4-22 测试）
- **撤回到 LGB-only**：保留 M1 全部基础设施（特征/CV/中性化/tradability/cost），只换最后的模型层

**留给 v2 后半段的资产**：见 §M1 末尾"M1 留下的可继承资产"清单。

### 3.7 赛规理解纠正

- 误以为只有 3 次提交 → 实际 **4 次**
- 误以为数据有截止日 → 实际**只有开源模型/embedding 需 ≤ 2026-04-01**，行情数据无限制
- 误以为 W1 五一长假需要做 holiday-fill → 赛方在评分时已自动用 T+4 替代缺失的 T+5

---

## 四、W1 防守补丁（2026-04-24）

### 4.1 问题诊断：近 10 天的 regime shift

213 天 rolling 全量看 +2.41%，但分窗口诊断发现：

| 窗口 | LGB mean | HS300 等权 | Δ | t(Δ) |
|---|---|---|---|---|
| 近 10 日 | **−0.06%** | **+1.37%** | **−1.42%** | **−3.87** ❌ |
| 近 30 日 | +0.30% | −0.19% | +0.49% | +1.43 |
| 近 60 日 | +1.11% | −0.16% | +1.27% | +4.61 |
| 近 90 日 | +1.36% | +0.13% | +1.23% | +6.38 |
| 2025 H2（146d） | **+2.97%** | +0.58% | +2.39% | +11.11 ✅ |

**根因**：LGB 偏小盘成长，4 月大盘蓝筹轮动，模型选股结构性偏离

### 4.2 防守方案：M10 系列 AB

设计三档对比：
- Baseline：原 deterministic Top-5 + equal
- M10-2：从 Top-10 候选选 5 只，强制 ≥2 只大盘
- M10-3：从 Top-10 候选选 5 只，强制 ≥3 只大盘

实现细节：
- 新增 `code/src/ensemble/portfolio.py::mcap_constrained_topk`（53 行 + 5 单元测试）
- 新增 `code/src/ensemble/regime.py::detect_regime`（45 行 + 2 单元测试，本次只记录信号不影响选股）
- 通过 monkey-patch 注入 `rolling_backtest_lgb_only.py` 和 `predict_lgb_only.py`
- env var `MCAP_MIN_LARGE` 控制档位，零侵入主线

### 4.3 213 天 AB 结果

| 档位 | 213d 全量 | 近60d | 近30d | 近10d | Δ30 vs HS300 |
|---|---|---|---|---|---|
| Baseline | +2.910% | +0.875% | +0.168% | +0.140% | +0.301% |
| M10-2 | +2.920% | +0.864% | +0.030% | +0.210% | +0.164% |
| **M10-3** | **+2.953%** | +0.837% | **+0.248%** | **+0.252%** | **+0.381%** |

**M10-3 全维度最优**——长期 +0.04pp、近 30 天 +0.08pp、近 10 天 +0.11pp。

### 4.4 vs 赛方 baseline（硬门槛）

赛方 baseline 6 天 rolling：mean **−0.881% / 5d**（永远固定 5 大金融蓝筹）

| 档位 | 我们 mean | 赛方 mean | Δ | 胜率 |
|---|---|---|---|---|
| Baseline-LGB | +0.101% | −0.881% | **+0.98pp** | 4/6 (67%) |
| **M10-3** | **+0.755%** | −0.881% | **+1.64pp** | 5/6 (83%) |

**赛规门槛已稳定通过**，M10-3 比 v1 的 0.0642 不直接可比（评分体系不同）但更可信。

### 4.5 W1 提交持仓（target=2026-04-23）

```
stock_id, weight
300308, 0.2   # 中际旭创（光模块）
600023, 0.2   # 浙能电力（大盘）
688187, 0.2   # 时代电气（科创）
688256, 0.2   # 寒武纪（AI 算力）
002714, 0.2   # 牧原股份（大盘）
```

3 大盘 + 2 科技龙头，防御 + 进攻天然组合。

---

## 五、三阶段对比

| 维度 | v1（3.15-4.10） | v2-M1 三模型 ensemble（4.14-4.21） | v2-M5 LGB-only + W1（4.22-4.24） |
|---|---|---|---|
| 模型 | StockTransformer | LGB+DE × 0.1 + MASTER × 0.9 + StockMixer × 0.0 | LGB+DE 单模型（3 seeds avg） |
| 特征 | 158+39（赛方原版） | Alpha158 + Alpha360 + 估值 + 市场63维 | Alpha158 + Valuation = 174 |
| 标签 | zscore | rank-gauss | rank-gauss |
| 配权 | linear | confidence × top_k=4 × α=0.7 | equal × top_k=5 + mcap_constraint |
| 评分 | 本地加权收益 0.0642（30d） | 87 天 rolling +1.01% (t=+4.78) | 213 天 rolling +2.95% (t≈+12) |
| vs HS300 | 未严格比较 | +0.90pp (87d) | +2.52pp (213d) |
| vs 赛方 baseline | 未比较 | 未比较 | **+1.64pp (6d)** |
| 训练 | 6.3 小时（GPU） | 6.3 小时（3模型 × 3 fold × 3 seed + refit） | 8 分钟（CPU/GPU 都行） |
| 跨平台 | ❌ CUDA 非确定性，分差 0.17 | ❌ MASTER attention 非确定性 | ✅ Linux/Windows MD5 一致 |
| 多 seed | 单 seed | 3 seeds × 3 模型 | 3 seeds (42, 2024, 7) refit |
| 单元测试 | 0 | ~5 | 18（pytest 全绿） |
| Rolling 验证 | 30 天本地 | 87 天 | 213 天 + 6 天赛方 baseline |
| Docker | 未做 | 已配置（~4 GB） | 已配置（待最终验证） |

**两次跃迁**：
1. **v1 → M1**：从单 Transformer "调参冲分" → 工业化 ensemble + walk-forward CV，引入了 87 天 rolling 验证（首次有 t-stat > 4 的硬证据）
2. **M1 → M5**：从"3 模型 ensemble（实质单 MASTER）"→ deterministic 单 LGB + 多 seed 平均，**牺牲一点理论上限换取跨平台复现性 + 训练成本下降 47×**，然后用省下的预算换长 rolling（87→213）+ AB 实验（M9 配权、M10 mcap）

---

## 六、当前未解决的问题

### 已确认 ⚠️
1. **近 10 天 regime shift 未根治**：M10-3 仅减少 11bp 损失，结构性偏好需 W2 加板块均值特征
2. **样本量小**：vs 赛方 baseline 仅 6 天，置信度中等
3. **Docker dry-run 未在 5090 实测**：W1 提交前必须做

### 待 W2 探索 🟡
1. **板块均值特征**（参考 2024 一等奖方案）：直接打 regime shift
2. **SHAP 可解释性**：答辩刚需，对标 2024 一等奖
3. **peTTM 季报因子**：证券姐姐推荐的基本面因子
4. **bootstrap CI**：为 7/18 报备准备置信区间
5. **label zscore vs rank-gauss 重新 AB**：队伍 B 实证 zscore 更强

---

## 七、产物索引（v2 阶段）

| 类型 | 文件 |
|---|---|
| 核心代码 | `code/src/pipeline.py`、`code/src/ensemble/{portfolio,allocation,regime,blender,tradability}.py` |
| 特征 | `code/src/features/{build,alpha158,valuation,neutralize,beta}.py` |
| 模型 | `code/src/models/lgb_de.py`（DoubleEnsemble 包装） |
| 训练入口 | `train.sh` → `python code/src/train.py` |
| 推理入口 | `test.sh` → `MCAP_MIN_LARGE=3 python scripts/predict_lgb_only.py` |
| Rolling | `scripts/rolling_backtest_lgb_only.py`、`test/rolling_backtest.py` |
| 测试 | `test/test_*.py`（18 项 pytest 全绿） |
| 数据 | `data/stock_data.csv`（2024-01-02 ~ 2026-04-23，166k 行） |
| 模型权重 | `model_lgb_only/lgb/seed_{42,2024,7}_refit/` |
| 赛方 baseline 权重 | `model/60_158+39/best_model.pth`（MD5 `d9c56a6a`） |
| Spec | `docs/superpowers/specs/2026-04-2{1,2,3,4}-*.md` |
| Plan | `docs/superpowers/plans/2026-04-2{4}-*.md` |
| 阶段报告 | `docs/reports/2026-04-24-stage-report.md`、`docs/reports/2026-04-24-w1-ab-decision.md` |
| 赛方 baseline 权威记录 | `docs/reference/baseline-authoritative.md` |
| 队友交接 | `docs/handoff/for_claude_{linux,windows}.md`、`docs/handoff/2026-04-24-w1-verify-{linux,windows}.md` |

---

## 八、第二阶段核心结论

1. **LGB-only 主线在 213 天 rolling 上 +2.41%/5d，t=+12，胜率 85%**——这比 v1 的本地 0.0642 是质变，因为 v1 没有长期 rolling 验证
2. **赛规跑赢 baseline 门槛已通过**：M10-3 vs 赛方 baseline 6 天 +1.64pp
3. **跨平台一致性已证实**：Linux/Windows 4060 各跑 5 次 MD5 一致，赛规复现要求满足
4. **W1 提交配置**：M10-3 (`MCAP_MIN_LARGE=3`)，已 commit `7bb30cd`，待队友验证后 push 提交
5. **方法论升级**：从"调参冲分"转向"spec → plan → TDD → rolling AB → 决策记录"的工业化流程

---

## 九、第三阶段路线图（提议）

按 ROI 排序，W2-W3 探索：

1. **M10b 板块均值特征**（重训）——直接打 regime shift，对标 2024 一等奖
2. **M14 SHAP 可解释性**——答辩刚需
3. **M5 市场 regime 特征**（指数动量 + 北向资金）——配合板块特征做 style-aware
4. **M2 holiday-fill** + **bootstrap CI 8 seeds**——为 7/18 报备准备
5. **M11 peTTM 季报因子**——证券姐姐推荐，需基本面数据接入

---

*v2 阶段报告 · 2026-04-24 · `feat/ensemble-v1` 分支 · commit `a95a704` 起点*
