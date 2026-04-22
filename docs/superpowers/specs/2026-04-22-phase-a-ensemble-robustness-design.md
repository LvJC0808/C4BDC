# Phase-A 改进设计（集成稳健化 + 交易摩擦 + 动态权重）

日期：2026-04-22
分支：`feat/ensemble-v1`（基于 `snapshot/5090-v1` tag 之后）

## 0. 背景

87 天滚动回测（2025-11-03..2026-03-13）结果：组合 mean 5 日收益 +1.01%，t=+4.78, p<0.0001。当前已合规提交，但诊断发现三处**显著可优化点**：

1. **Ensemble 权重过拟合 holdout**：grid search 在最后 10 天 holdout 上得到 `{lgb:0.1, master:0.9, mixer:0.0}`，mixer 权重被压成 0 不符合模型间互补直觉，master 占比过高有 overfit 风险。
2. **未建模交易摩擦**：涨停（±10% / ±20% for ChiNext/STAR）次日无法按开盘买入、手续费 3 bp/side 全被忽略。
3. **权重静态**：模型能力随市场 regime 漂移，但当前 ensemble 固定权重无法适应。

## 1. 目标

在**不重训模型**的前提下，通过改进 ensemble / portfolio 层降低过拟合、贴近真实成交约束：

| 目标 | 指标 |
|---|---|
| 权重稳健化 | 滚动回测 mean 下降 ≤ 15%，但**方差**下降 ≥ 10%，且 **p 值仍 < 0.01** |
| 交易摩擦建模 | `result.csv` 选股不含 T 日涨停，回测扣费后 mean 下降 ~10 bp/周 在预期内 |
| 动态权重 | 比固定最优权重在滚动回测上 **胜率 ≥ 当前 70.11%** |

## 2. 改进项详细设计

### 2.1 Ensemble 权重稳健化（Robust Blending）

**现状**：`grid_search_weights` 在 holdout 单点搜索，返回单一最优 combo。

**新设计**：
1. 候选权重集：
   - `w_holdout` = 当前 grid search 结果
   - `w_cv_ic` = CV 各 fold val 集上每模型的 RankIC 均值做 softmax（β=1.0）
   - `w_uniform` = `{1/n_models}`
2. **正则化挑选**：计算每个候选 `w` 在 CV 全部 9 个 val 集（3 fold × 3 seed）上的 **平均 Top-K 实现收益**；选收益-方差比（类 Sharpe）最高者。
3. 若最优候选与 `w_uniform` 的 KL 散度 > 0.5 nat，fallback 到 `0.5·w_best + 0.5·w_uniform` 做 shrinkage。

**实现**：新增 `code/src/ensemble/robust_blend.py`
- `cv_ic_weights(cv_val_scores, labels) -> dict[str, float]`
- `evaluate_weights_on_cv(weights, model_scores_by_fold, labels, top_k) -> float` 返回 Sharpe-like
- `robust_grid_search(model_scores_by_fold, labels, top_k_list, shrinkage_threshold=0.5)` 封装三步流程

**集成点**：`pipeline.py cmd_train` 在写 `ensemble_config.json` 前调用新函数（或保留旧函数做 A/B）。

### 2.2 交易摩擦建模（Tradability Filter + Fees）

**现状**：`build_portfolio` 按 `final_score` 降序直接取 Top-K。

**新设计**：
1. **涨停过滤**：接受 `tradable_fn(stock_id, date) -> bool` 回调，若 T 日涨幅 ≥ 9.9%（主板）或 ≥ 19.9%（创业板/科创板 `30xxxx/68xxxx`）则剔除该股。
2. **跌停过滤**（对称，防卖不出）：T 日跌幅 ≤ −9.9% 也剔除。
3. **Top-K 候选扩展**：若过滤后候选不足 K，从 Top-(K+5) 里补，最少保证 K-2 只票；仓位相应按实际数量等权。
4. **手续费扣除**：回测脚本 `rolling_backtest.py` 里对每只票收益 ×(1 − 2·cost)，`cost=3e-4` 默认。

**实现**：
- `code/src/ensemble/tradability.py` 新增：
  - `is_tradable(stock_df, stock_id, target_date, limit_pct=0.099, special_limit_pct=0.199) -> bool`
  - `filter_tradable_topk(scores_df, stock_df, target_date, top_k, expand=5) -> list[str]`
- 修改 `code/src/ensemble/portfolio.py` 的 `build_portfolio` 接受可选 `tradable_ids` 参数；为 None 时行为不变（向后兼容）。
- 修改 `code/src/pipeline.py cmd_predict` 构建 `tradable_ids`。
- 修改 `test/rolling_backtest.py` 加 `--cost_bps` 参数（默认 3），对 portfolio 和两个 baseline 同样扣费。

### 2.3 动态时变权重（Time-Adaptive Blending）

**现状**：`blend_scores(model_scores, weights)` 的 `weights` 是标量 dict。

**新设计**：扩展 `blend_scores` 支持按日期动态权重。
- 新函数 `blend_scores_dynamic(model_scores, weight_fn, ...)`，其中 `weight_fn(date) -> dict[str, float]`。
- 提供工具函数 `rolling_ic_weights(model_scores, labels, window=20, beta=1.0) -> pd.DataFrame[date, model, weight]`：对每个交易日 t，计算每模型过去 `window` 天的 RankIC，softmax(β·IC) 得权重。缺失 IC fallback 到 uniform。
- **推理侧**：`cmd_predict` 用最近 20 天历史分数+标签算一次当天权重；避免每天重算 CV。
- **回测侧**：`rolling_backtest.py` 加 `--dynamic_weights` flag，开启后每个 T 用 `rolling_ic_weights(t)`。

**实现**：
- `code/src/ensemble/blender.py`：加 `blend_scores_dynamic` 和 `rolling_ic_weights`。
- `code/src/pipeline.py cmd_predict`：条件分支（读 ensemble_config 里 `"blend_mode": "static" | "dynamic"`）。
- ensemble_config.json 扩展可选字段 `blend_mode`、`rolling_window`、`beta`。

## 3. 非目标（明确排除）

- **不重训任何模型**：本次只改 ensemble / portfolio 层，复用 `model/5090/` 现成权重。
- **不改特征 / CV / 模型结构**：放到 Phase B。
- **不改 Docker / 交付结构**：readme.md 更新即可。

## 4. 测试策略

每个子项都要有：

1. **单元测试**（合成数据，<5s）：在模块自身 `__main__` 里跑，`./.venv/bin/python -m code.src.ensemble.<mod>` 打印 `OK`。
2. **集成测试**：`test/rolling_backtest.py` 能用新逻辑跑 87 天，不崩溃。
3. **对比回测**：生成 4 份 backtest CSV：
   - `rolling_backtest_full.csv`（baseline，已存在）
   - `rolling_backtest_robust.csv`（仅 2.1）
   - `rolling_backtest_tradable.csv`（2.1 + 2.2）
   - `rolling_backtest_dynamic.csv`（2.1 + 2.2 + 2.3）
4. **报告更新**：在 `docs/report.md` 加一节 "Phase-A 优化" 展示 4 组对比。

## 5. 验收标准

- [ ] 三个新模块各自 `__main__` 测试打印 `OK`
- [ ] `rolling_backtest_robust.csv` 生成，方差降低 ≥ 10% 或 p<0.01 保持
- [ ] `rolling_backtest_tradable.csv` 生成，涨停股不出现在 `tickers` 列中
- [ ] `rolling_backtest_dynamic.csv` 生成，3 模型动态权重均值不等于 holdout 固定权重
- [ ] `ensemble_config.json` 新字段向后兼容（老 config 仍能 predict）
- [ ] `docs/report.md` 新增 Phase-A 章节，4 列对比表格

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Robust blend 均值下降过多 | `shrinkage_threshold` 和 `cv_ic_weights` 的 β 是可调超参，网格搜最优 |
| 涨停过滤后候选不足 | `expand=5` 且 `K-2` 的下限保证 |
| 动态权重回测过拟合 | `rolling_ic_weights` 只用历史 20 天，严格 t-1 可见性 |
| ensemble_config 破坏旧 predict | 所有新字段可选 + 显式 default |

## 7. 交付物清单

- `code/src/ensemble/robust_blend.py`
- `code/src/ensemble/tradability.py`
- `code/src/ensemble/blender.py`（追加 `blend_scores_dynamic` + `rolling_ic_weights`）
- `code/src/ensemble/portfolio.py`（修改 `build_portfolio` 签名）
- `code/src/pipeline.py`（集成三个改进，兼容旧配置）
- `test/rolling_backtest.py`（加 `--cost_bps / --dynamic_weights / --tradable_filter` 参数）
- `docs/report.md`（更新 Phase-A 章节）
