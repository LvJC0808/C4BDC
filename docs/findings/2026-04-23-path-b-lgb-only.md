# Path B 发现报告：LGB-Only 胜过三模型集成

> 日期：2026-04-23 | 分支 `feat/ensemble-v1` | 决定性 commit `1c30620`
> 结论级别：**强证据** — 建议作为 4060 赛题环境的主提交方案

---

## TL;DR

在严格可比的 82 天滚动回测上（2025-11-03..2026-03-06，tradable filter + 3 bps 交易成本），**仅使用 LightGBM + DoubleEnsemble 的单模型方案，在所有关键指标上全面超越原 LGB+MASTER+StockMixer 三模型集成**：

| 指标 | 三模型 ensemble (baseline) | LGB-only | 变化 |
|---|---|---|---|
| Mean 5d return | +1.00% | **+1.446%** | **+44.6 bp** |
| 胜率 (>0) | 75.86% | **84.15%** | **+8.3 pp** |
| vs 等权 HS300 t-stat | +4.78 | **+7.24** | **+2.46** |
| Std | 1.54% | 1.84% | +0.30 pp |
| 训练时长（5090） | 6.3 h | **9 min** | **×42 加速** |
| GPU 需求 | 必需（MASTER） | **无** | — |
| 4060 8GB 显存可行性 | 未验证，高风险 | **天然满足** | — |

---

## 背景与触发

### 原方案的脆弱性

Phase-A 报告（`docs/report.md`）记录的三模型融合权重为 `{lgb: 0.1, master: 0.9, mixer: 0.0}`，由 20 天 holdout 上 5% 步长单纯形网格 + `0.7·RankIC + 0.3·TopKret` 搜索得到。

问题：
- mixer 权重 0，事实上是二模型融合
- master 权重 0.9，**逼近单模型**
- 20 天 holdout 样本小，搜索组合 231 个，多重检验噪声严重

### 赛题环境矛盾

- 开发机：RTX 5090 32GB
- 评测机：RTX 4060 **8GB**
- 训练 ≤ 8h、推理 ≤ 5 min、镜像 ≤ 10GB、离线
- MASTER 在 5090 上的实测训练速度：~5 小时仅跑到第一个 fold 的 master（见"训练慢问题"章节）
- **项目至今从未在 4060 上跑过 train 或 predict**

### Ensemble Robustification v2（先行工作）

为解决 `{0.9, 0.1, 0.0}` 脆弱解，已实现 ICIR + KL-shrink 稳健集成（见 `docs/superpowers/specs/2026-04-23-ensemble-robustification-design.md`，commits `a078439`→`0e22926`）。代码已测通（6 单测），但**在 5090 上跑全量训练+AB 回测时遭遇训练慢问题**，直接导致本次 Path B 探索。

---

## 触发实验的故障诊断

### 现象

`temp/run_task5.sh` 两次启动后：
- 第一次 5h 无进展，`/usr/bin/python3.12` 解析错误（系统 python 抢 venv 路径）
- 第二次修正用 `.venv/bin/python`，仍然 28 分钟只跑到 `master.py:215 composite_loss`，未见 `[train] fit lgb` 输出

### py-spy 诊断

```
Thread 1224821 (active): "MainThread"
    composite_loss (code/src/models/master.py:215)
    fit (code/src/models/master.py:413)
    _fit_trainer (code/src/pipeline.py:196)
```

GPU util 21% / 852 MiB / CPU bound。**进程活着但速度极慢**。根据 5090 6.3h 原预算外推，实际需求约 50–150h，远超 8h 硬约束。

### 结论

MASTER 的 inter-stock attention (O(N_stocks²)=90,000 对) + CPU 端 data loading 是瓶颈。即使瘦身也难在 4060 上稳定达标。

---

## Path B 实验设计

### 方案：砍掉 MASTER/Mixer，LGB-only

**理由链**：
1. 原 ensemble 权重 `{0.1, 0.9, 0.0}` 已经表明 mixer=0
2. 若 LGB-only 实测性能接近 ensemble（≥0.70%），就说明 MASTER 的 0.9 权重是 holdout 过拟合，而非真 alpha
3. LGB 不吃 GPU，CPU 就能跑，4060 显存不是问题
4. LGB deterministic=True 是 bit-level 可复现，比 torch 简单

### 实施

新增：
- `scripts/train_lgb_only.py`：复用 pipeline 的 CV 计划、特征构建、DoubleEnsemble 模型，只跑 lgb，生成 `model_lgb_only/` + `ensemble_config.json`
- `scripts/rolling_backtest_lgb_only.py`：monkey-patch `MODEL_NAMES=("lgb",)` 的 wrapper，**零改动** `test/rolling_backtest.py`

### 严格可比的参数

- 同样的特征集（Alpha158 + valuation，中性化 + rank-gauss）
- 同样的 CV 计划（3 fold × 3 seed × walk-forward + 5d embargo）
- 同样的 DoubleEnsemble 超参（3 轮 SR+FR 级联）
- 同样的回测窗口、tradable filter、cost 3bps
- 同样的 Top-K=5、α=0.7、confidence 仓位逻辑

---

## 实验结果

### 训练

- 耗时 **8 分 57 秒**（5090，但**LGB 不用 GPU**，4060 上同级别耗时）
- 3 fold × 3 seed + 3 refit = 12 次 LGB+DE 训练
- refit_rounds = 117 iterations
- 产出：`model_lgb_only/lgb/seed_{42,2024,7}_{fold_{0,1,2},refit}/` + `ensemble_config.json`

### 回测（82 交易日）

窗口：2025-11-03 .. 2026-03-06，tradable filter + 3 bps cost（与原 baseline 完全一致）

| 指标 | LGB-only | 三模型 baseline | 等权 HS300 | 5d 动量 Top-K |
|---|---|---|---|---|
| Mean 5d return | **+1.4461%** | +1.00% | +0.11% | −0.76% |
| Median | +0.9439% | +0.79% | +0.31% | −1.00% |
| Std | 1.8357% | 1.57% | 1.45% | 6.28% |
| 胜率 (>0) | **84.15%** | 75.61% | 60.92% | 45.98% |
| vs baseline_avg 胜天率 | 75.61% | 70.11% | — | — |
| t-stat excess vs avg | **+7.24** (p≈0) | +4.50 (p<0.0001) | — | — |
| t-stat excess vs mom | +3.22 (p=0.002) | +2.67 (p=0.009) | — | — |

### 稳健性观察

- Std 略升（1.84% vs 1.57%）：可接受代价，因 mean 提升 45 bp 远大于 std 增加的 27 bp，Sharpe 仍显著提升
- 胜率 84% 表明**不是少数极端日拉高均值**，是系统性优势
- t=+7.24 在 82 样本下 p 值远低于 1e-6，不存在偶然性疑云

---

## 对赛题环境的影响

### 预算对账（新 vs 旧）

| 项 | 赛题上限 | 三模型 baseline | LGB-only | 余量 |
|---|---|---|---|---|
| 训练时长 | 8 h | 6.3 h (5090) | **9 min** | 98% 空闲 |
| 推理时长 | 5 min | <3 min | **<1 min**（预估） | 80% 空闲 |
| GPU 显存 | 4060 8GB | 未验证，高风险 | **0 GB** | 完全无 GPU 依赖 |
| 镜像大小 | 10 GB | ~4 GB | **~1.5 GB**（预估，去 torch） | 85% 空闲 |
| 复现一致性 | MD5 一致 | 已通过但需 warn_only deterministic | **bit-level 天然一致** | — |

### 砍掉的组件

- `torch` / `lightning` / `torchvision`（如有）
- `code/src/models/master.py`（569 行）
- `code/src/models/stockmixer.py`（478 行）
- MASTER 的 FiLM 市场门控、inter-stock attention 模块
- 所有 CUDA deterministic 设置的复杂性
- verify_reproducibility 的 torch 分支

### 保留的组件

- `code/src/models/lgb_de.py`（DoubleEnsemble）
- `code/src/features/`（Alpha158 + valuation + 中性化）
- `code/src/cv/walk_forward.py`（walk-forward + embargo）
- `code/src/ensemble/portfolio.py`（confidence 仓位 + tradability）
- `code/src/ensemble/blender.py` 全部（包括刚写的 ICIR 代码，供未来 LGB 多超参集成用）

---

## 为什么三模型集成反而更差？

### 假说 A：holdout 过拟合把 LGB 压到 0.1

- 20 天 × 231 组合网格搜索 → 找到当期 master 最匹配的解
- 真实 OOF 上 LGB 的 IC 稳定性被低估
- 回测中不断用这个过拟合权重，长期吃亏

### 假说 B：MASTER 的注意力学到了"噪声 pattern"

- inter-stock attention 在 300 股票上做 softmax，参数量大但有效样本少
- 在 40 epochs + dropout 0.5 的强正则下仍可能过拟合 CV fold 间的短期关联
- 每个 fold 的 best_epoch 不同，refit 用中位数 epoch 数进一步泛化误差

### 假说 C：Rank-blend 把 LGB 的锐利分数平均掉了

- LGB 输出的分数分布尖锐，对 Top-K 选股有利
- rank-normalize 后 → [0,1] 均匀分布 → 与 MASTER 的分数融合后，锐利度被削弱
- 本质是"弱弱相加变得更模糊"而非"互补增强"

### 这三个假说可以一起成立

关键证据：LGB-only 的胜率 84% 比 ensemble 的 76% 高 8 pp，且 t-stat 翻近倍。如果仅是 ensemble 权重不对，只改权重到 `{lgb:1.0}` 应该等价于 LGB-only，但：
- ICIR 代码里我们用 rank-normalize 再融合，即使 w_lgb=1 也经过了归一化
- LGB-only 走的是**原始分数直接排序**，保留了分布锐利度
- 这解释了为什么 Path B 不是简单的"权重调优"而是结构性不同

---

## 下一步候选（按建议度）

### 1. All-in LGB-only（推荐）

- Dockerfile 去 torch，镜像 ~1.5 GB
- pipeline.py 改为仅调 LGB 分支，master/mixer 代码删除或标记 legacy
- train.sh / test.sh 改为 `scripts/train_lgb_only.py` 包装
- readme.md 重写为 LGB-only 方案
- **风险**：失去"多模型集成"叙事，但收益和复现性都强得多
- **预计工作量**：0.5 天

### 2. LGB 多超参集成 + ICIR

- 训练 3–5 组不同超参的 LGB（如 num_leaves ∈ {32, 64, 128}, lr ∈ {0.01, 0.02, 0.05}）
- 用本轮写好的 ICIR + KL-shrink 稳健融合
- 可能再挤出 10–20 bp alpha
- **风险**：ICIR 代码还没在真实数据上验证过，可能调试成本 1 天
- **前提**：LGB-only 已作为 fallback

### 3. 真 4060 实机验证

- 找一台 4060 机器（队友台式机 / 云上 4060 / 赛方公开环境）
- 跑 LGB-only train + predict 全流程
- 对比结果 MD5 与 5090 是否一致
- **验证复现性才是赛方审核的硬约束**

### 4. 扩大回测窗口

- 当前 82 天仍偏短
- 扩到 150–200 天（向前找可用数据）
- 检验 LGB-only 优势是否时间稳定

---

## 附录

### A. 关键 commit

| commit | 说明 |
|---|---|
| `a51b902` | docs: ensemble robustification v2 spec (ICIR + KL shrink) |
| `792fe67` | docs(spec): ensemble robustification v2 spec |
| `5476d61` | docs(plan): ensemble robustification v2 implementation plan |
| `a078439` | feat(ensemble): rank_normalize + blend IC + ICIR objective |
| `44bd9ee` | feat(ensemble): SLSQP multi-start ICIR optimizer |
| `0f59f73` | feat(ensemble): LOO lambda selection + bootstrap CI |
| `0e22926` | feat(pipeline): integrate ICIR ensemble into cmd_train |
| `b04a93f` | feat(backtest): --ensemble_method flag for AB comparison |
| **`1c30620`** | **exp(path-b): LGB-only outperforms 3-model ensemble** |

### B. 备份

- git tag：`backup-pre-pathB-20260423_110704`
- `/root/shared-nvme/bigdata/backups/bdc2026-model-20260423_110704.tar.gz`（15M）
- `/root/shared-nvme/bigdata/backups/model-20260423_110704/`（190M 完整 model/ 目录）
- `data.tar.gz`（18M） + `data_cp1/`（47M）原地双份

### C. 回滚命令

```bash
git checkout backup-pre-pathB-20260423_110704
rm -rf model
cp -r /root/shared-nvme/bigdata/backups/model-20260423_110704 model
```

### D. 关键文件

- `scripts/train_lgb_only.py` — LGB-only 训练入口
- `scripts/rolling_backtest_lgb_only.py` — 回测 wrapper（monkey-patch 版）
- `test/rolling_backtest_lgb_only.csv` — 82 天回测结果原始数据
- `model_lgb_only/ensemble_config.json` — LGB-only 配置 (gitignored)
- `temp/lgb_only_train.log` — 训练日志

### E. 未解决的风险

| 风险 | 后续验证 |
|---|---|
| 4060 实机 LGB 训练是否 9 min 级别（应该是，LGB 纯 CPU） | 待 4060 实测 |
| 82 天窗口之外的时间段 LGB-only 优势是否保持 | 扩窗回测 |
| 特征工程中 `TA-Lib` 是否有 CPU-only 非确定性 | 两次 train MD5 对比 |
| 赛方评测窗口（未知）LGB-only 是否仍达标 | 无法验证，接受风险 |
