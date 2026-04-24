# 2026-04-24 优化路线图

> T=04-16 单窗口揭示 LGB 在近期 regime 下相对 baseline 退化（+0.13% vs +0.44%）。
> 82 天 rolling 均值仍 +1.446% vs −0.686%，但需排查近期退化原因并寻找增量 alpha。

## 诊断

| 口径 | LGB-only | Baseline | LGB 胜？ |
|---|---|---|---|
| 82 天 rolling (2025-11~2026-03) | **+1.446%** | −0.686% | ✅ +213 bp |
| T=03-03 单窗口 | +3.485% | −0.944% | ✅ +443 bp |
| T=03-06 单窗口 | +1.761% | +1.325% | ✅ +44 bp |
| **T=04-16 单窗口 (最新)** | **+0.13%** | **+0.44%** | ❌ **−31 bp** |

单窗口落后的假设：
1. **regime shift**：训练 val 在更早的 2024-2025，学到的因子暴露（可能偏动量/成长）在 04 月防御/价值风格下失效
2. **label 口径偏差**：没做 holiday-forward-fill，近期跨假期样本污染
3. **特征过期**：Alpha158 + 39 是纯量价，没有基本面，近期可能是财报期驱动
4. **seed 抖动**：仅 3 个 seed，单窗口不稳

---

## 优化清单（ROI 排序）

### 🔴 P0 立刻

**M1. 加基本面特征**（当前"免费"的 alpha）
- `stock_data.csv` 已含 `peTTM, pbMRQ, psTTM, pcfNcfTTM` 但**未被用于建模**
- 加入原值 + 5d/20d/60d 变化率 + 截面 rank
- 预期 +5~20 bp
- 工时 0.5 天

**M2. 扩 rolling 到 400 天 + 近期分组**
- 现在 82 天仅覆盖 2025-11~2026-03，不知道是否覆盖多 regime
- 扩到 2024-06 ~ 2026-04-16（≈ 400 交易日）
- 分组报告：全窗口均值 / 近 60 天 / 2024 / 2025 / 2026Q1
- 诊断 regime shift 是否真实
- 工时 1 小时（数据已就绪）

**M3. Label holiday-forward-fill**
- `build_label()` 从 `shift(-k)` 改为严格按交易日序号 + 跨假期 forward fill
- 赛规正确性硬需求（队伍 A Wolf 已指出）
- 工时 2-3 小时

### 🟠 P1 本周

**M4. 配权 AB**
- 在 rolling 上对比 `equal / linear_floor5 / linear_floor25 / softmax_t03`
- 确定是否 linear 配权（单窗口不稳定）在 rolling 多窗口上仍有均值优势
- 工时 1.5 小时

**M5. 多 seed + bootstrap CI**
- SEEDS 扩到 8 个 `[42, 2024, 7, 1337, 999, 31415, 2718, 1024]`
- 82 天 rolling 均值报告 95% CI
- 叙事强度 × 2
- 工时 2 小时

**M6. 行业中性化 Top-5 约束**
- Top-5 里单一行业 ≤ 2 只
- 分散系统性风格风险
- 预期 +5~15 bp（降低方差 > 降低均值）
- 工时 2-3 小时

### 🟡 P2 下周

**M7. label zscore vs rank-gauss AB**（队伍 B 实证 zscore 更强）
- 工时 1 天

**M8. Portfolio-aware eval metric**
- LGB 训练时 early-stop 基于 top-5 组合收益（而非 Spearman IC）
- 工时 0.5 天

**M9. 配权幂衰减 / max_weight 约束**
- `weight ∝ rank^(-0.5)` 或 `max_weight=0.35`
- 工时 1 小时

### 🟢 P3 如有余力

**M10. 加入市场 regime 特征**
- 指数 20d/60d 动量、波动率、北向资金
- 工时 1-2 天

**M11. 端到端组合优化 loss**
- 工时 1 周+

---

## 排除的方向

| 方案 | 排除理由 |
|---|---|
| RNN / GRU | 引入 torch 依赖，破坏工程约束 |
| 5 分钟 K 线 | 与日频 label 不对齐，数据量爆炸 |
| 重新训 MASTER | 已证实跨平台不可复现 |
| linear 满仓（照抄队伍 B） | T=04-16 实测 −2.32%，winner-take-all 风险大 |
| 追求 Windows bit-level | 评测在 Linux Docker，无意义 |

---

## 当前执行

1. **M1** 加基本面特征，重训 LGB
2. **M2** 扩 rolling 到 400 天，分组诊断
3. 两件事完成后根据数据决定 M3-M6 的顺序
