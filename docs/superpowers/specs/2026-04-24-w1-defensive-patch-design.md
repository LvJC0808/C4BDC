# W1 提交前防守补丁 · 设计文档

- **日期**：2026-04-24
- **分支**：`feat/ensemble-v1`
- **目标窗口**：近 30 天跑赢 HS300（W1 提交 2026-04-25）
- **硬约束**：不重训、不改 label、不动 DoubleEnsemble；仅推理层改动

---

## 一、背景与问题

213 天 rolling 显示 LGB-only 主线 **+2.41% / 5d，t=+12，胜率 85%**，但**近 10 天 -0.06% vs HS300 +1.37%（t=-3.87）**。根因：LGB 选出小盘成长股，市场近期轮动到大盘蓝筹。

W1 提交若延续当前小盘偏离，极可能在近期风格下**单窗口跑输 baseline**（4-16 单窗口已观察到 LGB 输 baseline 31bp）。

## 二、设计目标

1. **近 30 天窗口从 +0.30% 提升至 > HS300 的 +0.X%**（主要 KPI）
2. **213 天整体跌幅控制在 ≤ 0.3 pp**（不破坏长期 alpha）
3. **零重训、当日可实现验证**（工时 ≤ 3 小时）

## 三、方案总览

两个推理层组件，均不触碰模型训练：

- **A2 · 市值硬约束 Top-K 选股**（主力）：Top-10 候选中强制 ≥ 2 只大盘股
- **A3 · Regime-aware 配权**（辅助）：始终 `equal`（保守），regime 信号仅记录供诊断

---

## 四、组件设计

### 4.1 `mcap_constrained_topk`

**位置**：`code/src/ensemble/portfolio.py`（新增函数）

**签名**：
```python
def mcap_constrained_topk(
    scores: pd.DataFrame,     # [stock_id, score_q]，已去重去 NaN
    mktcap: pd.DataFrame,     # [stock_id, log_mktcap]，同日快照
    k: int = 5,
    candidate_k: int = 10,
    min_large_cap: int = 2,
    large_cap_quantile: float = 0.5,  # HS300 log_mktcap 中位数作为大盘下限
) -> pd.DataFrame
```

**逻辑**：
1. 用现有 `deterministic_top_k(scores, k=candidate_k)` 取 Top-10 候选
2. 从当日 mktcap 快照计算 HS300 的 `log_mktcap` 分位阈值 `M = quantile(large_cap_quantile)`
3. 标记 Top-10 中 `is_large = log_mktcap >= M`
4. 若 Top-10 中大盘数 ≥ `min_large_cap`：
   - 贪心选出 k=5：先按 score 降序选前 5；若大盘数 < min_large_cap，则把最后一个小盘替换为 Top-10 剩余中 score 最高的大盘，直到满足约束
5. 若 Top-10 中大盘数 < `min_large_cap`（罕见 regime）：**降级**为原始 Top-5（记录日志 `FALLBACK`）
6. 返回 `[stock_id, score_q]`，长度 = k

**不变式**：
- 输出长度恒 = k（除非候选池本身 < k）
- 输出保持 score 降序（便于后续 softmax/linear 配权，虽然当前只用 equal）
- 当 `min_large_cap=0` 时行为与原 `deterministic_top_k(k=5)` 等价（测试用）

### 4.2 `detect_regime`

**位置**：`code/src/ensemble/regime.py`（新增文件）

**签名**：
```python
def detect_regime(
    panel: pd.DataFrame,       # 含 datetime, instrument, close, log_mktcap
    as_of: pd.Timestamp,
    lookback_days: int = 10,
    large_quantile: float = 0.67,  # top1/3 为"大盘"
    small_quantile: float = 0.33,  # bottom1/3 为"小盘"
    threshold: float = 0.01,    # 大盘-小盘累计 > 1% 视为 rotation
) -> dict
```

**逻辑**：
1. 在 `as_of` 当日，按 `log_mktcap` 划 HS300 为大盘组（Q ≥ 0.67）/ 小盘组（Q ≤ 0.33）
2. 计算 `as_of - lookback_days` 到 `as_of` 的累计收益均值：`r_large`、`r_small`
3. `diff = r_large - r_small`
4. 返回 `{"regime": "large_cap_rotation" if diff > threshold else "neutral", "r_large": ..., "r_small": ..., "diff": ...}`

**用途（本次）**：
- **仅记录到日志和 CSV**，不改变配权（A3 简化为始终 equal）
- 为后续 M5 regime-aware 实验留接口

### 4.3 集成点

**`scripts/predict_lgb_only.py`**：
- 在 `deterministic_top_k` 之后调用 `mcap_constrained_topk`
- 需要从 panel 取当日 `log_mktcap` 快照（已有 `neutralize_cross_section` 前特征）
- regime 日志打印到 stderr

**`scripts/rolling_backtest_lgb_only.py`**：
- 通过 monkey-patch `rb.build_portfolio` 或更上游注入约束
- **或更简单**：在 `test/rolling_backtest.py` 的选股阶段（早于 `build_portfolio`）hook

**优选方案**：在 `rolling_backtest_lgb_only.py` 中 monkey-patch `rb.deterministic_top_k`，包装为"先取 Top-10 → 市值约束 → 取 Top-5"，对下游透明。

## 五、数据流

```
LGB blended scores (300 stocks)
  ↓
deterministic_top_k(k=10) → Top-10 候选
  ↓
mcap_constrained_topk(候选, log_mktcap, k=5, min_large_cap=2)
  ↓
[副作用] detect_regime(panel, as_of) → 日志
  ↓
build_portfolio → 原 confidence scaling
  ↓
allocate_by_mode(Top-5, "equal")
  ↓
result.csv / rolling CSV
```

## 六、验证方案

### 6.1 Rolling AB（必做）

在 213 天 rolling 上跑三档：

| 档位 | 选股 | 配权 | 产物 CSV |
|---|---|---|---|
| Baseline | deterministic Top-5 | equal | `rolling_lgb_alloc_equal.csv`（已存在） |
| **M10-2** | mcap_constrained(≥2, Top-10) | equal | `rolling_lgb_mcap2.csv` |
| M10-3 | mcap_constrained(≥3, Top-10) | equal | `rolling_lgb_mcap3.csv` |

### 6.2 决策规则

| 条件 | 动作 |
|---|---|
| M10-2 近 30 天 > HS300 **且** 213 天跌幅 ≤ 0.3 pp | **提交 M10-2** |
| M10-3 比 M10-2 更好（近期更强且整体损失更少） | 提交 M10-3 |
| 均不满足但 M10-2 近 30 天改善 ≥ +0.3pp 且整体损失 ≤ 0.5pp | 提交 M10-2（接受小幅长期代价换防守） |
| 全部不满足 | **提交原版 equal**（保底 +2.41%） |

### 6.3 时间盒

- 实现 + 单元测试：≤ 2 h
- Rolling 验证（3 档）：≤ 1 h
- 决策与提交：≤ 0.5 h
- **Deadline：2026-04-25 03:00 前完成**，否则直接提交原版

## 七、单元测试

`test/test_mcap_constrained.py`：
1. `min_large_cap=0` 等价于原 Top-5
2. Top-10 全大盘时输出 Top-5（无需替换）
3. Top-10 仅 1 只大盘时触发 FALLBACK（记录 + 返回原 Top-5）
4. Top-10 刚好 2 只大盘（排名靠后）时，替换最后 2 只小盘
5. 输出长度恒为 k，score 降序

## 八、风险与回退

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| M10 近期未改善 | 中 | 无增益 | 提交原版 |
| M10 长期受损 > 0.5pp | 中 | 长期叙事削弱 | 决策规则卡点 |
| log_mktcap 快照缺失 | 低 | 选股报错 | FALLBACK 到原 Top-5 |
| rolling 时间超预算 | 低 | 提交延期 | 03:00 硬 deadline + 原版保底 |

## 九、YAGNI 砍掉项

- peTTM 季报因子（需重训）
- K 线形态特征（需重训）
- SHAP 可解释性（答辩用，非 W1 提交必需）
- 板块均值特征（需重训）
- softmax/linear 配权 AB（近期波动大时 equal 更稳）
- Regime-aware 动态配权（简化为记录信号，不影响选股）

## 十、后续工作（W2+）

- M10 若有效，用 `min_large_cap` 作为超参数在 rolling 上做粗粒度扫描（1/2/3）
- 若 regime 信号在 rolling 上与 LGB 失效期高度相关，启用 A3 原方案（regime-aware 配权）
- 补 SHAP 分析作为答辩素材（非紧急）
- 板块均值特征 + 重训（W2 M5/M10 合并）
