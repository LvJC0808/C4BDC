# 方法论与架构 · Methodology (SSOT)

> 从数据到 result.csv 的完整方法论。
> 最后更新：2026-04-24 · Phase 4

---

## 1. 端到端架构

```
┌────────────────────────────────────────────────────────────────────┐
│ Data Layer (baostock)                                              │
│   stock_data.csv + industry_map + hs300_history + csi300 + basic   │
│   2024-01-02 ~ 2026-04-23 · HS300 成分股 · 日频                      │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Feature Layer (174 cols)                                           │
│   Alpha158 (Qlib + TA-Lib)                                         │
│   + Valuation (peTTM/pbMRQ/psTTM/pcfNcfTTM × raw/z20/z60/ind_rank) │
│   + Cross-sectional neutralize                                     │
│       → industry mean-shift                                        │
│       → OLS residual on [log_mktcap, beta60]                       │
│       → 3σ winsorize                                               │
│       → rank-gauss                                                 │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Model Layer                                                        │
│   LightGBM + DoubleEnsemble                                        │
│   × 3 seeds [42, 2024, 7] · 3-fold walk-forward CV                 │
│   · 5-day embargo · 20-day holdout                                 │
│   · refit_epochs = CV median × 1.05                                │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Blending Layer                                                     │
│   3-seed score mean (simple average)                               │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│ Portfolio Layer (W1 强化)                                          │
│   1. deterministic_top_k(k=10, quantize=1e-4, id tie-break)        │
│   2. mcap_constrained_topk(Top-10 → ≥3 大盘 → k=5)  ← Phase 4     │
│   3. equal weight 0.2                                              │
│   4. LF line terminator → result.csv                               │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Layer · 数据层

### 2.1 数据源
- **baostock**（免费公开、4-1 前可用，赛规合规）
- 所有原始数据通过 `code/src/data_prep/fetch_all.py` 一键抓取

### 2.2 6 份 CSV 分工
| 文件 | 内容 | 用途 |
|---|---|---|
| `stock_data.csv` | HS300 日线 OHLCV + 换手率 + 涨跌幅 + 估值 | 主数据 |
| `industry_map.csv` | 申万一级行业映射 | 中性化 |
| `stock_basic.csv` | 上市日期 / 状态 | 数据清洗 |
| `hs300_history.csv` | HS300 历史成分股（半年频快照） | 动态池过滤 |
| `csi300_index.csv` | HS300 指数日线（sh.000300） | 市场特征 / Beta 计算 |
| `trade_calendar.csv` | 交易日历 | 日期对齐 |

### 2.3 兜底（Phase 4 加强）
赛方挂载 `/app/data` 时可能只给 `train.csv + test.csv`（赛规模板）。
→ **`init.sh` 自愈**：从 train+test 合并 stock_data.csv；辅助 5 文件从 `/app/data_bundled/` 补齐。

---

## 3. Feature Layer · 特征层（174 列）

### 3.1 Alpha158（Qlib 标准因子）

- 9 K 线形态 (KMID/KLEN/KUP/KLOW/KSFT/...)
- 4 价比 (OPEN0/HIGH0/LOW0/VWAP0)
- 5 窗口（5/10/20/30/60 日）× 29 技术因子
  - 移动平均 MA / 收益率 ROC / 标准差 STD
  - Beta / RSQR / RESI（回归统计量）
  - MAX / MIN / QTLU / QTLD / RANK / RSV
  - IMAX / IMIN / IMXD（极值位置）
  - CORR / CORD（量价相关）
  - CNTP / CNTN / CNTD / SUMP / SUMN / SUMD（计数/和）

实现：`code/src/features/alpha158.py`（TA-Lib 向量化）

### 3.2 Valuation 因子

4 个基础估值：`peTTM / pbMRQ / psTTM / pcfNcfTTM`
对每个做 4 种变换：`raw / z-score 20d / z-score 60d / industry rank`
共 **16 个估值因子**。

实现：`code/src/features/valuation.py`

### 3.3 Cross-sectional Neutralize

每日横截面独立执行（消除风格暴露）：

1. **行业内减均值**（申万一级）
2. **OLS 残差**：对 `[log_mktcap, beta60]` 回归，取残差
3. **3σ winsorize**
4. **Rank-gauss**：`rank / N → Φ⁻¹`（标准正态逆变换）

标签（5 日开盘-开盘收益率）也同步做 rank-gauss。

实现：`code/src/features/neutralize.py`

**为什么 rank-gauss 而非 zscore**：对极端 outlier 更鲁棒（涨停/跌停日不污染模型）。详见 `07-decisions.md` DR-002。

---

## 4. CV Layer · 交叉验证

### 4.1 Walk-forward 切分

```
段 1: train[D_start..D_end-150]   embargo 5  val[D_end-144..D_end-125]
段 2: train[D_start..D_end-100]   embargo 5  val[D_end-94..D_end-75]
段 3: train[D_start..D_end-50]    embargo 5  val[D_end-44..D_end-25]
holdout:                                     [D_end-20..D_end]
```

### 4.2 关键参数

| 参数 | 值 | 理由 |
|---|---|---|
| `n_folds` | 3 | 平衡训练样本与时效性 |
| `embargo` | 5 天 | 标签跨 5 日，必须隔开防泄漏 |
| `val_days` | 20 | 每段验证窗口 |
| `holdout_days` | 20 | 最终策略选择 |
| `seeds` | [42, 2024, 7] | 3 seed 平均降方差 |
| `refit_epochs` | CV median × 1.05 | 经验公式 |

实现：`code/src/cv/walk_forward.py`

---

## 5. Model Layer · 模型层

### 5.1 LightGBM + DoubleEnsemble

**LGB 参数**（`code/src/config.py`）：
```python
LGB_PARAMS = {
  "objective": "regression_l1",  # MAE
  "num_leaves": 64,
  "learning_rate": 0.02,
  "feature_fraction": 0.7,
  "bagging_fraction": 0.7,
  "min_data_in_leaf": 200,
  "deterministic": True,           # 赛规第 5 条
  "force_row_wise": True,
  "num_threads": 4,
}
```

**DoubleEnsemble**（Han et al. 2020）：
- 2 轮级联 booster
- **SR (Shrinkage Reweighting)**：残差 L 升序 rank → `w_i = σ(−α·(r_i − 0.5))`
- **FR (Feature Reweighting)**：逐特征 shuffle 置换测 MAE 增量 → softmax 成特征权重（floor 0.1）
- 最终 score = 3 个 booster 的均值

实现：`code/src/models/lgb_de.py`

### 5.2 为什么 LGB-only（非三模型 ensemble）

Phase 2 实测表明三模型 ensemble 权重收敛到 `{lgb:0.1, master:0.9, mixer:0.0}` → 伪集成。MASTER 跨平台非确定性违反赛规。详见 `07-decisions.md` DR-001。

---

## 6. Blending Layer · 融合层

**简化方案**：3 seed score **简单平均**。

放弃了 Phase 2 的复杂融合（rank-normalize + holdout simplex grid + ICIR shrink + rolling IC），原因：
- 已切到单 LGB 模型 → 无多模型可融合
- 多 seed 简单平均 = 降方差的标准做法

实现：`scripts/predict_lgb_only.py`

---

## 7. Portfolio Layer · 组合层（Phase 4 核心强化）

### 7.1 Deterministic Top-K

**解决的问题**：LightGBM 跨平台浮点尾差会改变 Top-5 选股。

**机制**：
1. 对 score 量化到 `1e-4` 精度
2. 量化后相同分数，按 `stock_id` 字典序 tie-break
3. 保证同数据 → bit-identical result.csv

实现：`code/src/ensemble/portfolio.py::deterministic_top_k`

### 7.2 Market-cap Constrained Top-K（M10-3，W1 提交核心）

**解决的问题**：LGB 选出的 Top-5 在 2026-04 大盘轮动期集中在小盘成长（4/5 小盘），近 10 天 -1.42% vs HS300。

**算法**：
```
1. deterministic_top_k(k=10) → 取 Top-10 候选
2. 计算当日 HS300 log_mktcap 中位数 M 作为大盘门槛
3. 贪心从 Top-10 中挑 5 只：
   - 先按 score 降序选 5 只
   - 若大盘数 < 3 → 替换最低 score 的小盘为最高 score 的大盘
4. Fallback：若 Top-10 中大盘 < 3 → 降级回原 Top-5（保护）
```

**参数**：`MCAP_MIN_LARGE=3, MCAP_CAND_K=10, MCAP_LARGE_Q=0.5`

**详见**：`07-decisions.md` DR-005；AB 对比详见 `03-results.md` §2。

实现：`code/src/ensemble/portfolio.py::mcap_constrained_topk`

### 7.3 Regime 诊断（W1 未启用）

`code/src/ensemble/regime.py::detect_regime` 计算近 10 日大盘 vs 小盘累计收益差 → 判定是否 `large_cap_rotation`。

**W1 只记录信号到日志，不影响选股**。W2+ 可能启用 regime-aware 动态配权。

### 7.4 配权

Top-5 等权 0.2。详见 `07-decisions.md` DR-003。

### 7.5 输出

`result.csv`，LF 行尾符（`lineterminator="\n"`）。

---

## 8. 损失函数

**LGB**：`regression_l1`（MAE）
- 对收益率厚尾分布更鲁棒
- 叠加 DoubleEnsemble 样本权重

详见 `code/src/config.py` 和 `code/src/models/lgb_de.py`。

---

## 9. 数据扩增

**本方案不做显式数据扩增**（不引入合成样本）。

替代手段（在特征层和标签层）：
- **Cross-sectional neutralize**（§3.3）
- **Rank-gauss transform**（§3.3）
- **3σ winsorize**（§3.3）

---

## 10. 模型集成

- **多 seed 集成**：3 个 seeds × 3 folds + 3 refit → 平均
- **多模型集成**：不做（Phase 2 → Phase 3 跃迁理由见 DR-001）

---

## 11. 工程保证

### 11.1 Deterministic
- `LGB_PARAMS`: `deterministic=True, force_row_wise=True, num_threads=固定`
- `PYTHONHASHSEED=0`（Dockerfile）
- `set_global_seed(seed)` 在每次 fit / predict 前重置 numpy/random/lightgbm

### 11.2 赛规硬合规
| 赛规 | 实现 |
|---|---|
| §1 固定种子复现 | 双跑 73 文件 MD5 bit-identical |
| §1 结果一致 | Linux × 3 节点 MD5 完全一致 |
| §2 训练 ≤ 8h | 实测 5.5 min |
| §2 推理 ≤ 5 min | 实测 < 3 min |
| §3 Docker ≤ 10 GB | 镜像 1.64 GB |
| §4 开源模型 4-1 前 | 无使用预训练模型 |
| §5 不联网 | 无网络调用 |

详见 `04-reproducibility.md`。

---

## 12. 关键代码入口

| 用途 | 入口 |
|---|---|
| 推理（W1 提交） | `bash test.sh` → `scripts/predict_lgb_only.py` |
| 训练 | `bash train.sh` → `scripts/train_lgb_only.py` |
| 初始化 + data 自愈 | `bash init.sh` |
| Rolling 回测 | `scripts/rolling_backtest_lgb_only.py` |
| 赛方 baseline 回测 | `scripts/rolling_backtest_baseline.py` |
| Bootstrap CI | `scripts/bootstrap_ci.py` |
| 三档 AB 一键 | `scripts/run_w1_ab.sh` |
