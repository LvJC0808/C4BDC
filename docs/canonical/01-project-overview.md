# 项目总览 · Project Overview (SSOT)

> 本文是项目现状的唯一真理源。查看完整方法论 → `02-methodology.md`；查看数据结果 → `03-results.md`。
> 最后更新：2026-04-24 · Phase 4

---

## 1. 项目定位

**THU-BDC2026 · 沪深 300 五日组合预测系统**

- **任务**：输入沪深 300 成分股过去 60 日 OHLCV + 衍生数据，输出最多 5 只股票组合（权重和 ≤ 1）
- **结算**：T+1 开盘买入，T+5 开盘卖出
- **评分**：必须跑赢官方 StockTransformer baseline 才进入排名
- **分支**：`feat/ensemble-v1` · 最新 commit `c5bb8aa`

---

## 2. 当前提交配置（W1 · 2026-04-25）

| 项 | 值 |
|---|---|
| 主线 | LightGBM + DoubleEnsemble × 3 seeds 平均 |
| Risk Containment | M10-3 (`MCAP_MIN_LARGE=3`) |
| 推理入口 | `bash test.sh` → `scripts/predict_lgb_only.py` |
| 训练入口 | `bash train.sh` → `scripts/train_lgb_only.py` |
| result.csv MD5 | `f13034946c0aaea5cb1e3f2d0d6ad692` |
| Top-5 持仓 | `300308, 600023, 688187, 688256, 002714`（各 0.2） |

详见决策档 `07-decisions.md` DR-005。

---

## 3. 核心成绩

| 指标 | 数值 | 参照 |
|---|---|---|
| 213 天 rolling mean | **+2.95% / 5d** | LGB-only + M10-3 |
| t-stat vs HS300 等权 | **+12.05** | p < 0.0001 |
| 胜率（port_ret > 0） | **87%** | 213 天样本 |
| Δ vs HS300 等权 | **+2.58 pp** | 213 天 |
| Δ vs 赛方 baseline | **+1.64 pp** | 6 天样本 |
| Δ vs 5 日动量 Top-K | **+6.49** (t-stat) | 213 天 |

完整数据 → `03-results.md`。

---

## 4. 阶段轨迹（Phase 1 → Phase 4）

项目在 40 天内经历 **4 个阶段、4 次方法论跃迁**：

| Phase | 时间 | 主线 | 关键数字 | 主要产出 |
|---|---|---|---|---|
| **Phase 1 · Baseline Alignment** | 03/15 – 04/10 | StockTransformer 调参 | 本地 0.0642 | 基础设施、数据抓取、baseline 对齐 |
| **Phase 2 · Ensemble Foundation** | 04/14 – 04/21 | LGB + MASTER + Mixer 三模型 | 87d rolling +1.01%, t=+4.78 | 严格 CV、rank-blend、Phase-A 稳健化 |
| **Phase 3 · Deterministic Core** | 04/22 – 04/23 | 单 LGB + DoubleEnsemble | 213d +2.41%, t=+12.05 | 跨平台复现、长 rolling、bootstrap CI |
| **Phase 4 · Risk Containment** | 04/24 | + mcap_constrained_topk | +2.95% / Δ30=+0.38pp | M10-3 提交配置、赛规硬合规 |

**四次跃迁的共同模式**：每次切换都由**新证据**迫使撤回「当时最高分方案」——不是因为它们差，而是因为发现了更严重的风险。

详见方法论 `02-methodology.md`。

---

## 5. 赛规合规状态

| 赛规条款 | 状态 | 证据 |
|---|---|---|
| §1 固定种子训练复现一致 | ✅ | 73 文件双跑 MD5 identical |
| §1 推理复现一致 | ✅ | 本机双跑 + Linux/Windows 4060 × 5 次 |
| §2 训练 ≤ 8h | ✅ | 实测 5.5 min（余量 86×） |
| §2 推理 ≤ 5 min | ✅ | 实测 < 3 min |
| §3 Docker ≤ 10 GB | ⏳ | 预估 ~4 GB，待 build 实测 |
| §4 开源模型 4-1 前报备 | ✅ | 无使用 |
| §5 复现不联网 | ✅ | 完全离线 |
| **跑赢 baseline** | ✅ | +1.64 pp（6 天样本） |

详见复现性证据 `04-reproducibility.md`、赛方 baseline `05-baseline.md`。

---

## 6. 主要代码结构

```
code/src/
├── pipeline.py              # train/predict CLI
├── config.py                # 超参数与路径
├── features/                # Alpha158 + 估值 + 中性化（174 列）
├── cv/walk_forward.py       # 3 段 walk-forward + 5d embargo
├── models/lgb_de.py         # DoubleEnsemble LGB 包装
└── ensemble/
    ├── portfolio.py         # deterministic_top_k + mcap_constrained_topk
    ├── allocation.py        # equal/linear/softmax 配权
    ├── blender.py           # rank-blend + rolling IC
    ├── tradability.py       # 涨跌停过滤
    └── regime.py            # 大盘/小盘累计收益差（Phase 4 新增）

scripts/
├── train_lgb_only.py        # W1 训练入口
├── predict_lgb_only.py      # W1 推理入口
├── rolling_backtest_lgb_only.py  # 213 天 rolling
├── rolling_backtest_baseline.py  # 赛方 baseline rolling
├── bootstrap_ci.py          # 10000 重采样 CI
└── run_w1_ab.sh             # 三档 AB 一键脚本

model_lgb_only/lgb/seed_{42,2024,7}_refit/  # W1 权重
```

详见 `02-methodology.md` §架构层次。

---

## 7. 文档导航

| 你想查 | 打开 |
|---|---|
| 现在系统什么样 | 本文 |
| 每个组件为什么这样设计 | `02-methodology.md` |
| 所有实验数据和 rolling 结果 | `03-results.md` |
| 跨平台 MD5 证据 | `04-reproducibility.md` |
| 赛方 baseline 真实表现 | `05-baseline.md` |
| 后续 W2-W4 做什么 | `06-roadmap.md` |
| 为什么不做 X / 为什么撤回 Y | `07-decisions.md` |
| W1 提交的具体状态 | `../submission/W1-submission-log.md` |
| 队友跨机验证指引 | `../submission/W1-verification-checklist.md` |

**旧文档位置**：所有 2026-04-24 之前的报告、spec、plan、findings 已归档到 `../archive/`（只读，不主动维护）。
