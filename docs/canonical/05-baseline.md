# 赛方 Baseline · 权威记录 (SSOT)

> SSOT (Single Source of Truth) · 判断"跑赢 baseline"前必读。
> 取代旧文档 `docs/reference/baseline-authoritative.md`（已 archive）。
> 最后更新：2026-04-24 · Phase 4

---

## 1. 赛方 baseline 定义

**本体**：`/root/shared-nvme/C4BD/THU-BDC2026-main/`（赛方下发原版，**勿改**）

**组件**：
- `code/src/model.py` — `StockTransformer`
- `code/src/utils.py` — 特征工程（158+39 列）
- `code/src/predict.py` — 推理入口
- `model/60_158+39/best_model.pth` — 训练好的权重

**权重 MD5**：`d9c56a6a332c0fef08157ca7661a2acb`

---

## 2. 我们本仓库内 baseline 权重

**路径**：`model/60_158+39/best_model.pth`
**MD5**：`d9c56a6a332c0fef08157ca7661a2acb`（与赛方完全一致）
**来源**：赛方下发代码的预训练权重，未做任何修改

同一模型文件 + 同一输入 → 同一 baseline 输出。**我们对 baseline 的评估可信度高**。

---

## 3. 我们如何跑赛方 baseline

**脚本**：`scripts/rolling_backtest_baseline.py`

- `from model import StockTransformer`（加载赛方 model.py）
- 加载 `model/60_158+39/best_model.pth`
- 自己 reimplement 推理逻辑（与赛方 predict.py 等价，但嵌入我们的 rolling 框架）

**赛方代码差异**（不影响 baseline 推理结果）：

| 文件 | 一致性 | 备注 |
|---|---|---|
| `code/src/model.py` | ✅ 一致 | 赛方原版 |
| `code/src/utils.py` | ✅ 一致 | 赛方原版 |
| `code/src/predict.py` | ⚠️ 不同 | 我们不用它 |
| `code/src/train.py` | ⚠️ 不同 | 我们没重训 baseline 权重 |
| `code/src/config.py` | ⚠️ 不同 | 仅影响训练，推理不影响 |
| `model/60_158+39/best_model.pth` | ✅ MD5 一致 | 同权重 |

---

## 4. 赛方 baseline 真实表现

### 4.1 6 天 rolling（2026-04-01 ~ 04-09）

来源：`test/rolling_backtest_baseline_apr.csv`

| 日期 | Top-5 | 5d 收益 |
|---|---|---|
| 04-01 | 600919, 601658, 601939, 601328, 601169 | −1.04% |
| 04-02 | 600919, 601658, 601169, 601939, 601328 | −2.43% |
| 04-03 | 600919, 601658, 601169, 601939, 601816 | −1.43% |
| 04-07 | 600919, 601658, 601169, 601916, 601816 | −0.72% |
| 04-08 | 600919, 601658, 601169, 601916, 601816 | −0.51% |
| 04-09 | 600919, 601169, 601658, 601916, 601816 | +0.84% |

**统计**：
- mean = **−0.881% / 5d**
- t-stat ≈ −1.99
- 胜率 = 17% (1/6)

### 4.2 行为特征

1. **永远固定 5-8 只金融/基建蓝筹**：
   - 600919 江苏银行
   - 601658 邮储银行
   - 601169 北京银行
   - 601939 建设银行
   - 601328 交通银行
   - 601916 浙商银行
   - 601816 京沪高铁
   - 601398 工商银行（偶现）

2. **相邻日期换手 ≈ 0**（5 只里 4 只重叠）

3. **本质**：固定单一风格 anchor（大盘金融），不是动态选股

---

## 5. 跑赢 baseline 的硬门槛

**赛规第 3 条**：必须跑赢 baseline 才能参与排名。

### 5.1 我们 vs 赛方 baseline（6 天重叠）

| 档位 | 我方 mean | 赛方 mean | 超额 | 胜率 |
|---|---|---|---|---|
| Baseline-LGB (MCAP=0) | +0.101% | −0.881% | **+0.98 pp** | 4/6 (67%) |
| **M10-3 (W1 提交)** | **+0.755%** | **−0.881%** | **+1.64 pp** | **5/6 (83%)** |

**赛规硬门槛已通过**。详见 `07-decisions.md` DR-008。

---

## 6. 重要注意事项 ⚠️

### 6.1 Baseline 不是 HS300 等权

HS300 等权是我们 213 天 rolling 的参照基准之一，但**不是赛规门槛**。

- HS300 等权 2026-04 近 10 天 +1.37%
- 赛方 baseline 同期 −0.88%

**"跑赢"的唯一正确口径 = vs 赛方 baseline**，不是 vs HS300。

### 6.2 样本小

6 天 rolling 是我们目前跑过的赛方 baseline 样本。样本小但可靠性高，因为 baseline 是 deterministic 的固定金融蓝筹策略——跨日变异极小，趋势已充分反映。

### 6.3 赛方 baseline 不会变

赛方权重已锁定（MD5 `d9c56a6a...`），我们的 baseline rolling 结果对 W1-W4 四次提交都有效。

### 6.4 如何新增 baseline rolling 天数

```bash
python scripts/rolling_backtest_baseline.py \
  --eval_start <YYYY-MM-DD> --eval_end <YYYY-MM-DD> \
  --model_dir ./model/60_158+39 \
  --out test/rolling_backtest_baseline_<tag>.csv
```

每跑一天需 GPU 推理 + 人工，成本较高，不建议大规模跑。

---

## 7. 相关产物索引

| 文件 | 内容 |
|---|---|
| `test/rolling_backtest_baseline_apr.csv` | 4 月 6 天赛方 baseline rolling 结果 |
| `scripts/rolling_backtest_baseline.py` | 赛方 baseline rolling wrapper |
| `scripts/score_self_compare.py` | 单窗口 baseline vs LGB 对比 |
| `model/60_158+39/best_model.pth` | 我们本仓库内赛方 baseline 权重 |
| 赛方代码权威副本 | `/root/shared-nvme/C4BD/THU-BDC2026-main/`（勿改） |
