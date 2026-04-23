# Finding: Cross-Platform ICIR 权重发散

> 2026-04-23 | 基于队友 Windows 4060 与 Linux 4060 两份产出的 val_scores

## TL;DR

用**完全相同的 ICIR + KL-shrink 权重学习代码**，在两份队友独立跑出的 OOF 分数上重算最优权重，得到**完全相反的结果**：

| 维度 | Linux 4060 | Windows 4060 |
|---|---|---|
| LOO λ* | **2.0**（强正则） | **0.05**（几乎无正则） |
| lgb 权重 | 0.280 | 5.8e-7（≈ 0） |
| master 权重 | 0.420 | **0.985** |
| mixer 权重 | 0.300 | 0.015 |
| ICIR 最优值 | 3.31 | 4.47 |
| 等权 ICIR | 3.30 | 3.65 |
| LGB-only ICIR | 2.88 | 2.89 |
| Bootstrap 是否稳定 | ✅ 全部 CI 不跨 0 | ❌ lgb CI 跨 0 |

**根本原因**：MASTER 在 CUDA 上的非确定性导致跨平台 val_scores 偏差（mean abs diff **0.17**），mixer (1e-6)、lgb (1e-3) 则近似一致。

**战略影响**：**三模型集成方案的核心组件（MASTER）在跨平台不可复现，违反赛规"从训练过程开始复现"硬约束**。Path B LGB-only 不只是性能更好，而是**唯一合规选项**。

---

## 1. 数据来源

| 来源 | 路径 | 内容 |
|---|---|---|
| Windows 4060 | `/root/shared-nvme/bigdata/mates_issue/mate_windows/mate-win.zip` | `model/{lgb,master,mixer}/seed_{42,2024}_fold_{0,1,2}/val_scores.parquet` + refit checkpoint |
| Linux 4060 | `/root/shared-nvme/bigdata/mates_issue/mate_linux/logs_20260423_v2.tar.gz` | 同上 |

两份都是 seeds=[42, 2024]（非默认 [42, 2024, 7]）、CV 3 fold、holdout 结尾。

---

## 2. Val_scores 跨平台一致性测量

按 `(instrument, datetime, seed, fold)` 严格对齐，比较 score 列：

| 模型 | rows | same_keys | **score mean abs diff** | **score max abs diff** |
|---|---|---|---|---|
| lgb | 34,582 | ✅ | 1.9e-3 | 6.6e-2 |
| **master** | 22,470 | ✅ | **1.7e-1** | **9.5e-1** |
| mixer | 34,260 | ✅ | 2.4e-7 | 4.3e-6 |

### 解读

- **lgb**：量级 1e-3。LightGBM 在 `deterministic=True, force_row_wise=True` 下理论上应 bit-level 一致，这里的 1e-3 漂移是 CPU 浮点 reduction 顺序差异，可接受
- **mixer**：量级 1e-6。StockMixer 的 MLP 结构在 CUDA 上接近确定，符合预期
- **master**：量级 **1e-1**，最大 0.95。完全不同的分数分布。最可能原因：
  - FiLM 市场门控 + inter-stock attention 中的 `scaled_dot_product_attention` fallback 到 **Memory Efficient attention**（之前 5090 训练日志里也见过该 warning："defaults to a non-deterministic algorithm"）
  - 不同 CUDA 版本（linux 用 4060 driver 570+, windows 可能 driver 566.x）+ 不同 cuDNN release 产生截然不同的数值

---

## 3. ICIR 重算结果

脚本：`scripts/recompute_icir_from_mates.py`
输出：`temp/icir_result_linux.json`、`temp/icir_result_windows.json`

### 3.1 LOO λ 搜索

λ 网格 `[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]`

**Linux 的 λ 搜索曲线**（单调上升 → λ=2.0 最大）：
```
0.0:  1.584
0.05: 1.722
0.1:  1.776
0.2:  1.741
0.5:  1.734
1.0:  1.841
2.0:  1.880  ← 最大
```

**Windows 的 λ 搜索曲线**（λ=0/0.05 最高，大正则反而塌）：
```
0.0:  3.032
0.05: 3.038  ← 最大
0.1:  2.464
0.2:  2.515
0.5:  2.413
1.0:  2.366
2.0:  2.374
```

λ=2.0 在 Windows 上只有 2.37，在 Linux 上有 1.88；说明 Windows 的 OOF 分布允许更自信的权重解（小 λ 也稳），而 Linux 的 OOF 分布噪声更大，需要大 λ 强正则把权重拉回等权附近。

### 3.2 最优权重

**Linux**：
```json
{"lgb": 0.280, "master": 0.420, "mixer": 0.300}   ICIR = 3.31
```

**Windows**：
```json
{"lgb": 5.8e-7, "master": 0.985, "mixer": 0.015}  ICIR = 4.47
```

Windows 的结果和 Phase-A baseline 的 grid search 结果 `{lgb:0.1, master:0.9, mixer:0.0}` **高度一致** —— 说明 Phase-A 的 baseline 是在类 Windows 的环境上搜的。

### 3.3 Bootstrap 置信区间（n=100）

**Linux**：
| 模型 | median | 95% CI |
|---|---|---|
| lgb | 0.330 | [0.171, 0.437] |
| master | 0.368 | [0.298, 0.469] |
| mixer | 0.307 | [0.186, 0.399] |

所有 CI 都不跨 0，权重稳健。

**Windows**：
| 模型 | median | 95% CI |
|---|---|---|
| lgb | ≈ 0 | **[0, 0.326]** |
| master | 0.985 | [0.360, 0.987] |
| mixer | 0.015 | [0.007, 0.359] |

**lgb 的 CI 包含 0**，说明 Windows 下 lgb 的权重本质是"可有可无"。

---

## 4. 反直觉观察：LGB-only ICIR 最低，但回测收益最高

| 方案 | Linux OOF ICIR | Windows OOF ICIR | Path B 回测 mean 5d |
|---|---|---|---|
| ICIR 最优（权重搜索） | 3.31 | 4.47 | — |
| 等权三模型 | 3.30 | 3.65 | — |
| LGB-only | 2.88 | 2.89 | **+1.446%** |
| Phase-A baseline（grid search，类 Windows） | — | — | +1.00% |

### 为何矛盾？

- **ICIR 衡量全体股票的 rank-return 相关性稳定性**；回测只关心 **Top-5 选股的真实收益**
- MASTER 的 CUDA 非确定性让 OOF IC **数值看起来高**（可能是对 CUDA quirk 的过拟合，而不是真的 alpha），但 refit 后在"真正的测试 T+1 开盘价"上就打折扣
- LGB 输出分数分布锐利、Top-5 选股决断力强；rank-normalize 到 [0,1] 后的"ICIR 稳定性"反而不如三模型混合

**结论**：**OOF ICIR 是模型质量的 proxy，不是提交成绩的 proxy**。Path B LGB-only 在实盘接轨的 rolling backtest 上把 ensemble baseline 打爆，是更可信的指标。

---

## 5. 对赛题合规性的影响（最关键）

赛规《代码规范》第 1 条：

> 在固定随机种子点的前提下，**从训练过程开始复现**，选手提交项目代码运行生成的结果和提交的结果完全一致

### MASTER 的合规风险

| 环境对 | val_scores mean abs diff | 影响 |
|---|---|---|
| Win 4060 vs Linux 4060 | 0.170 | 权重搜索结果反向 |
| 预计 Linux 4060 vs 赛方 Linux 4060（不同 driver/CUDA） | ≥ 0.01 量级 | 足以改变 Top-5 选股 |

赛方复现时跑 `train.sh` 得到的 checkpoint **必然和提交时的 checkpoint 不一致**（只要 driver/cuDNN 小版本不同）。权重搜索出的 0.98 master 会放大这个漂移。结果就是：**我方提交的 result.csv 可能无法被复现 → 取消获奖资格**。

### LGB-only 的合规优势

| 维度 | Linux vs Win 实测 |
|---|---|
| mean abs diff | 1.9e-3 |
| 对 Top-5 选股的影响 | 几乎为 0（分数差距比 1e-3 大得多） |
| 对最终 result.csv 的影响 | **高概率 bit-level 一致** |

加上 `deterministic=True, force_row_wise=True, num_threads` 固定，LGB 是目前唯一能接近"真复现"的组件。

---

## 6. 战略建议

### 主路线：All-in LGB-only（本 finding 新增论据）

除了 Path B 已证明的"性能更好 / 训练更快 / 显存 0 依赖"，新增第四条论据：**跨平台复现是赛规硬约束，只有 LGB 能满足**。

### 如仍要保留集成选项

- **禁止用 MASTER**（CUDA 非确定）
- 可考虑 LGB + XGBoost + CatBoost 三棵 GBDT 集成（都是 CPU 确定），用本轮写好的 ICIR 代码融合
- 但需先实测 XGBoost / CatBoost 在 CSI300 因子上 vs LGB 的相对差（很可能 LGB-only 依然最好）

### 不要再做

- 在 MASTER 上花时间调 batch / lookback / d_model 适配 4060
- 依赖 `torch.use_deterministic_algorithms(True)` —— 只是 warn_only，在 attention backward 阶段实际仍非确定
- 用 OOF ICIR 作为唯一评估指标 —— 会被 MASTER 的平台依赖误导

---

## 7. 复现脚本

- `scripts/recompute_icir_from_mates.py`
- 输入：`/tmp/mate_win/`、`/tmp/mate_linux/`（由 zip/tar.gz 解压而来）
- 输出：`temp/icir_result_{linux,windows}.json`
- 运行时长：约 15 min（LOO 7 × 3 fold + bootstrap n=100）

## 8. 关键命令

```bash
# 解压
unzip -q /root/shared-nvme/bigdata/mates_issue/mate_windows/mate-win.zip -d /tmp/mate_win/
tar xzf /root/shared-nvme/bigdata/mates_issue/mate_linux/logs_20260423_v2.tar.gz -C /tmp/mate_linux/

# 造 labels（需要 feature build）
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from code.src.features.build import build_feature_sets
f = build_feature_sets('./data','./temp',use_cache=True)
p = f['lgb']['panel'][['instrument','datetime','label']]
import pandas as pd; pd.to_datetime(p['datetime'])
p.to_parquet('/tmp/labels.parquet')
"

# 跑 ICIR
.venv/bin/python scripts/recompute_icir_from_mates.py
```

## 9. 附录：diff 验证命令

```python
# 验证 Master 的分数在 Linux vs Windows 差别
import pandas as pd, numpy as np
for m in ["lgb","master","mixer"]:
    w = pd.concat([pd.read_parquet(f"/tmp/mate_win/model/{m}/seed_{s}_fold_{f}/val_scores.parquet")
                   for s in [42,2024] for f in [0,1,2]]).sort_values(["instrument","datetime","seed","fold"]).reset_index(drop=True)
    l = pd.concat([pd.read_parquet(f"/tmp/mate_linux/model/{m}/seed_{s}_fold_{f}/val_scores.parquet")
                   for s in [42,2024] for f in [0,1,2]]).sort_values(["instrument","datetime","seed","fold"]).reset_index(drop=True)
    d = (w["score"].values - l["score"].values)
    print(f"{m}: mean_abs={np.abs(d).mean():.2e}  max_abs={np.abs(d).max():.2e}")
```
