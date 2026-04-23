# Ensemble Robustification v2 设计文档

> 2026-04-23 | 分支 `feat/ensemble-v1` | 对应 plan `2026-04-23-ensemble-robustification-plan.md`

## 1. 背景与问题

当前集成方案在 20 天 holdout 上用 5% 步长单纯形网格搜索，目标 `0.7·RankIC + 0.3·TopKret`，搜出 `{lgb:0.1, master:0.9, mixer:0.0}` —— 近似单模型。问题：

- 样本只有 20 天，搜索空间 231 组（步长 0.05）→ **多重检验严重**。
- `0.3·TopKret` 被 20 天内极端收益日主导，把权重推向 master。
- 一旦 master 在 4060 上训练结果漂移（混合精度/非确定算子），整个成绩塌陷。
- Phase-A 的 rolling-IC softmax 动态权重反而掉 15 bp，说明动态估计在小样本下噪声更大。

## 2. 目标

把权重学习改成**对样本噪声鲁棒、可解释、可复现**的方法，把 max 权重从 0.9 压到 ≤ 0.65，且回测收益下降不超过 5 bp。只改 `ensemble/`，不动基模型/特征/portfolio。

## 3. 方案概览

```
旧: holdout 20d × grid(step=0.05) × (0.7·IC + 0.3·TopKret) → {0.1, 0.9, 0.0}
新: (OOF ∪ holdout) × SLSQP × [ICIR − λ·KL(w ‖ 1/3)] + bootstrap CI
    │
    ├── 样本: CV OOF (~60 天) + holdout (20 天) ≈ 80 天
    ├── 目标: ICIR(w) = mean_t IC_blend(t;w) / std_t IC_blend(t;w) · √N
    ├── 正则: λ·KL(w ‖ 1/3)，λ 用 leave-one-fold-out 选
    └── 诊断: bootstrap 1000 次 → 每个 w_m 的 95% CI
```

## 4. 数据流

训练阶段，在 `cmd_train` 的 CV 循环结束（已产出 `val_scores.parquet`）后新增：

1. **聚合 OOF**：合并 3 model × 3 fold × 3 seed × val_scores，按 `(instrument, datetime)` 合并，对 seed 求均值。
2. **追加 holdout refit 预测**：已存在于 `holdout_preds`。
3. **每日截面 rank-normalize** 每个模型分数至 [0,1]。
4. **blend 后重算 IC**：`IC_blend(t;w) = spearman(Σ w_m·rank(S_m)(t), label(t))`。注意必须是 blend 后求 IC，不是 `Σ w·IC_m`（后者为近似且会丢非线性）。

## 5. 权重优化

### 5.1 目标函数

```
maximize  J(w) = ICIR(w) − λ·KL(w ‖ 1/3)
subject to  w_m ≥ 0,  Σw_m = 1

ICIR(w) = mean_t IC_blend(t;w) / (std_t IC_blend(t;w) + ε) · √N
KL(w ‖ 1/3) = Σ w_m · log(w_m / (1/3)) · 1[w_m > 0]
ε = 1e-6
```

### 5.2 求解

`scipy.optimize.minimize(method="SLSQP", ftol=1e-8, maxiter=200)`，最小化 `-J(w)`。
多起点：等权 `[1/3,1/3,1/3]` + 三个单模型顶点 `e_m` + 10 个 `Dirichlet(1,1,1)` 采样（rng=config.SEED），取目标值最优。

### 5.3 λ 选择（leave-one-fold-out）

- λ 候选：`[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]`
- 对每个 λ：对 3 个 CV fold 各留出一个，用其他 2 fold 的 OOF + holdout 求 `w*(λ)`，在留出 fold 上算 ICIR；3 折平均。
- 选平均 ICIR 最大的 λ。λ=0 → 无正则；λ=∞ → 等权。

### 5.4 Bootstrap 诊断

对 OOF+holdout 的日期做 1000 次有放回抽样（`numpy.random.default_rng(config.SEED)`），每次求 `w*(λ*)`，输出每个模型权重的 95% CI 和中位数。若任一模型 CI 跨 0，日志 warning。

## 6. 代码改动

### 6.1 新增：`code/src/ensemble/blender.py`

```python
def stack_oof_scores(val_scores_by_model, holdout_preds_by_model) -> dict[str, pd.DataFrame]:
    """合并 OOF + holdout，按 seed 求均值，返回每模型 [instrument, datetime, score]。"""

def rank_normalize_daily(df) -> pd.DataFrame:
    """每日截面把 score rank → [0,1]。"""

def compute_blend_ic_series(ranked_scores, labels, weights) -> np.ndarray:
    """按 weights 融合 → 每日 RankIC 时间序列。"""

def icir_objective(weights, ranked_scores, labels, lam) -> float:
    """返回 -ICIR + λ·KL，供 SLSQP 使用。"""

def optimize_weights_icir(ranked_scores, labels, lam, seed=42) -> dict:
    """多起点 SLSQP，返回 {weights, icir, neg_objective}。"""

def select_lambda_loo(ranked_scores_by_fold, labels, lam_grid, seed=42) -> dict:
    """返回 {lambda, mean_icir, per_fold_icir}。"""

def bootstrap_weights(ranked_scores, labels, lam, n=1000, seed=42) -> dict:
    """返回 {median, ci_low, ci_high, samples}。"""
```

保留旧 `grid_search_weights`，加 `# deprecated, kept for legacy` 注释。

### 6.2 改 `code/src/pipeline.py`

在 `cmd_train` 的"holdout ensemble grid search"段落前注入 ICIR 流程：

- 收集每 fold 的 OOF（需要 fold id），加 holdout refit 预测。
- rank-normalize。
- `select_lambda_loo` → `optimize_weights_icir` → `bootstrap_weights`。
- 写入 `ensemble_config.json` 新字段：`method`, `weights`, `lambda`, `icir`, `ci95`, `legacy_weights`（保留 grid search 结果做 AB）。

`cmd_predict` 保持 `ensemble_config["weights"]` 读取接口不变，自然切换。

### 6.3 改 `code/src/config.py`

```python
ENSEMBLE_METHOD = os.environ.get("ENSEMBLE_METHOD", "icir_shrink")  # "icir_shrink" | "legacy"
ICIR_LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
ICIR_BOOTSTRAP_N = 1000
ICIR_MULTI_START_DIRICHLET = 10
```

### 6.4 改 `test/rolling_backtest.py`

加 `--ensemble_method {legacy,icir_shrink}` flag，默认读 `ensemble_config.json.method`。

### 6.5 不改

基模型、特征、portfolio 构造、tradability、verify_reproducibility。

## 7. 测试

- `tests/test_ensemble_blender.py` 新增：
  - `test_rank_normalize_daily_uniform_in_01`
  - `test_icir_objective_etqual_weight_baseline`（等权时 KL=0）
  - `test_optimize_weights_icir_synthetic`（人造数据 master 信号强 → w_master > w_lgb/mixer 但 < 0.7）
  - `test_select_lambda_loo_returns_valid_lambda`
  - `test_bootstrap_weights_ci_covers_median`
- 不跑真实训练，用 synthetic 数据保证 CI 友好。

## 8. 验收标准

| 指标 | 当前 | 新版目标 |
|---|---|---|
| 回测 mean 5d（82 天 tradable+cost） | +1.00% | ≥ +0.95% |
| max 单模型权重 | 0.9 | ≤ 0.65 |
| λ* | — | 非 0 非 ∞ |
| 所有模型 bootstrap CI 跨 0 | 未知 | 至多 1 个 |
| 训练增量时间 | — | ≤ 5 min |
| 两次 train 可复现 MD5 | ✅ | 保持 ✅ |

未达 "回测 ≥ 0.95%" 则不发布，保留 legacy 权重走旧路径。

## 9. 风险与回退

| 风险 | 缓解 |
|---|---|
| 4060 基模型漂移 → 权重不再最优 | 权重基于 rank-normalized 分数，对分布平移鲁棒；legacy 权重在 config 保留可切换 |
| OOF 样本仍少 | λ·KL 正则 + bootstrap CI 诊断 |
| SLSQP 局部解 | 14 起点（1 等权 + 3 顶点 + 10 Dirichlet） |
| mixer 权重近 0 | bootstrap CI 若稳定≈0，下一 phase 可直接删 mixer 省 1/3 训练时间 |
| IC std 过小导致 ICIR 爆 | 加 `ε=1e-6` 到分母 |

## 10. 非目标（YAGNI）

- 不做 stacking / meta-learner（二级模型小样本必过拟合）
- 不做时变权重（Phase-A 已证明小样本下反而差）
- 不改基模型 / 特征 / portfolio
- 不加新模型
