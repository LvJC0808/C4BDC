# Ensemble Robustification v2 — 设计文档

- 分支建议：`feat/ensemble-robust-v2`
- 日期：2026-04-23
- 上下文：THU-BDC2026 沪深 300 Top-K 组合预测
- 相关前文：
  - `docs/superpowers/specs/2026-04-21-bdc2026-ensemble-design.md`（Ensemble v1）
  - `docs/superpowers/specs/2026-04-22-phase-a-ensemble-robustness-design.md`（Phase-A）
  - `docs/report.md` §3.1、§7、§11

---

## 0. 目标与约束

**目标**：把当前 `{lgb: 0.1, master: 0.9, mixer: 0.0}` 这种近似“单模型伪装成集成”的脆弱解，替换为对样本噪声鲁棒、可解释、可复现的权重，使成绩不因 MASTER 在 4060 上的漂移而塌陷。

**硬约束（不变）**：
- 训练 ≤ 8 h，推理 ≤ 5 min，镜像 ≤ 10 GB
- 离线运行，固定随机种子两次训练/推理 MD5 一致
- 目标硬件 i7-13650H / 16 GB RAM / RTX 4060 8 GB

**范围**：**仅修改 ensemble / 权重学习模块**；不动特征工程、不动三个基模型、不动 portfolio 构造与 tradability 过滤。风险面最小，收益面明确。

**非目标**：
- 不新增基模型
- 不改动态权重逻辑（Phase-A 的 `rolling_ic_weights` 保留但不启用）
- 不改 Top-K 内部权重分配（后续 phase 处理）

---

## 1. 问题分析

当前 `grid_search_weights`：
- 仅用 **20 天 holdout** 做样本
- 目标函数 `0.7·RankIC + 0.3·TopK_return`
- 0.05 步长单纯形网格，总计 231 组合
- 输出极端解 `{0.1, 0.9, 0.0}`

**三大缺陷**：
1. **样本不足**：20 天下估计 3 模型权重相当于每参数约 7 个独立观测
2. **TopK 项是尾部估计**：20 天 TopK 收益对单日极值高度敏感，把权重拉向在极端日表现好的模型
3. **无正则**：极端解无代价，自然塌缩到最强单模型

Phase-A 的 rolling-IC softmax 动态权重在同窗口反而 -15 bp，说明**样本量才是第一矛盾**，不是动态性。

---

## 2. 架构变化

```
旧：holdout_20d × grid_search(step=0.05) → {0.1, 0.9, 0.0}
     └── 目标: 0.7·RankIC + 0.3·TopK
     └── 样本: 20 days
     └── 无正则

新：OOF(CV 9×) ∪ holdout_20d → shrink(ICIR 最优解, 1/3, λ*)
     └── 目标: ICIR(w) = mean_t IC_blend(t;w) / std_t IC_blend(t;w) · √N
     └── 样本: ~70–80 days（OOF ≈ 50–60 + holdout 20）
     └── 软正则: argmax_w [ICIR(w) − λ·KL(w ‖ 1/3)]
     └── λ 用 leave-one-fold-out 选
     └── bootstrap 95% CI 做稳定性诊断
```

---

## 3. 数据流

在现有 `cmd_train` 的 CV 循环结束后（`val_scores.parquet` 已经产出），新增以下步骤：

1. **聚合 OOF 分数**
   - 读取 `model/{name}/seed_*_fold_*/val_scores.parquet`
   - 按 `(instrument, datetime)` 在 seed 维度求均值，得到每模型每 OOF 日的截面分数 `S_lgb, S_master, S_mixer`
   - OOF 天数约 `3 folds × 20 val_days ≈ 60`（若 fold 间有 gap 实际略少）

2. **追加 holdout**
   - 把 refit 阶段已经产出的 `holdout_preds[name]`（seed 平均后）追加进统一 DataFrame，`source ∈ {oof, holdout}`
   - 总样本 ≈ 70–80 天

3. **每日截面 rank-normalize**
   - 对每个模型分数在每个交易日内做 `rank / N → [0, 1]`
   - 标签同样做 rank（保持与现有 rank-gauss 一致）

4. **组合 IC 序列**
   - 对候选权重 `w`，每日计算：
     ```
     blend(t) = Σ_m w_m · rank(S_m)(t)
     IC_blend(t; w) = spearman(blend(t), label(t))
     ```
   - 得到时间序列 `IC_blend ∈ R^N`
   - **注意**：不用线性近似 `Σ w_m·IC_m`，保留 rank 融合的非线性

5. **过滤**
   - 丢弃 `len(截面) < 50` 的日期（港股停市/HS300 调仓缺口）
   - 丢弃 `|IC_blend(t)|` 为 NaN 的日期

---

## 4. 权重优化（核心）

### 4.1 目标函数

```
J(w) = ICIR(w) − λ · KL(w ‖ 1/3)

ICIR(w)    = mean_t IC_blend(t; w) / std_t IC_blend(t; w) · √N
KL(w‖1/3)  = Σ_m w_m · log(w_m / (1/3))

约束:
  w_m ≥ 0
  Σ w_m = 1
```

- `KL(w ‖ 1/3)` 在 `w = 1/3` 时为 0，在单点解（如 `(1,0,0)`）时为 `log 3 ≈ 1.099`
- `λ = 0` 退化为无正则 ICIR 最优；`λ → ∞` 退化为等权 1/3
- ICIR 的 `std` 如小于 1e-6 则置为 1e-6 避免除零

### 4.2 求解器

- `scipy.optimize.minimize(method="SLSQP")`，`maxiter=200`，`ftol=1e-8`
- 约束：`eq: sum(w) - 1`，`bounds: [(0, 1)] × 3`
- 多起点避免局部解：
  - 等权 `(1/3, 1/3, 1/3)`
  - 单模型 `(1,0,0), (0,1,0), (0,0,1)`
  - Dirichlet(α=1) 采样 10 个点，种子 `config.SEED`
- 共 14 起点，取 `J(w)` 最大者

### 4.3 λ 选择（防过拟合关键）

- λ 候选：`{0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0}`
- 在 **OOF** 上做 **leave-one-fold-out**（holdout 不参与 λ 选择，留作终评）：
  - 留出 fold_k 的 OOF 日期
  - 用其他两折求 `w*(λ)`
  - 在 fold_k 上评估 `ICIR(w*(λ))`
  - 三折 ICIR 平均 → `ICIR_cv(λ)`
- 选 `λ* = argmax_λ ICIR_cv(λ)`
- 用 `OOF ∪ holdout` 全部样本在 `λ*` 下重新求 `w*`，写入配置

### 4.4 Bootstrap 稳定性诊断

- 对 OOF∪holdout 日期做 1000 次有放回抽样，种子固定
- 每次抽样重算 `w*(λ*)`
- 输出每模型权重的 95% 置信区间
- **报警规则**：若某模型 95% CI 下界 < 0.05，说明该模型近乎可删除；仅记录日志，不自动剔除
- Bootstrap 仅做诊断，不影响最终 `w*`

---

## 5. 代码改动

### 5.1 新增/修改文件（全部限定在 ensemble 目录）

- **`code/src/ensemble/blender.py`**（修改）
  - 新增 `aggregate_oof_scores(model_dir, seeds, model_names) -> dict[name -> DataFrame]`
  - 新增 `optimize_weights_icir(per_model_scores, labels, lam, w0=None, multi_start=True, rng=None) -> dict`
    - 返回：`{"weights": {...}, "icir": float, "j_value": float, "converged": bool}`
  - 新增 `select_lambda_loo(oof_per_model, oof_labels, fold_ids, lam_grid, rng) -> dict`
    - 返回：`{"lambda": float, "icir_cv": float, "per_lambda": [...]}`
  - 新增 `bootstrap_weights(per_model_scores, labels, lam, n=1000, rng=None) -> dict`
    - 返回：`{"mean": {...}, "ci_low": {...}, "ci_high": {...}}`
  - 保留 `grid_search_weights`（加 `# Legacy` 注释，不删除）
  - 保留 `rolling_ic_weights` / `blend_scores_dynamic`（Phase-A 动态权重，默认不启用）

- **`code/src/pipeline.py`**（修改 `cmd_train`）
  - CV 结束后：
    1. 调 `aggregate_oof_scores` 得到 OOF 截面分数
    2. 调 `select_lambda_loo` 选 λ*
    3. 用 OOF∪holdout 调 `optimize_weights_icir` 求 w*
    4. 调 `bootstrap_weights` 做诊断
    5. **同时**跑一次 legacy `grid_search_weights`（只做日志对比，不用）
  - `ensemble_config.json` 字段扩展：
    ```json
    {
      "weights": {"lgb": ..., "master": ..., "mixer": ...},
      "top_k": ..., "alpha": ...,
      "method": "icir_shrink_v2",
      "lambda": 0.1,
      "icir_mean": 0.83,
      "icir_cv_by_lambda": {...},
      "bootstrap_ci95": {"lgb": [lo, hi], ...},
      "legacy_weights": {"lgb": 0.1, "master": 0.9, "mixer": 0.0},
      "blend_mode": "static",
      "refit_epochs": {...}, "feature_version": "v1.0",
      "seeds": [...], "cv_folds": 3
    }
    ```

- **`code/src/config.py`**（修改）
  - 新增：
    ```python
    ENSEMBLE_METHOD = "icir_shrink"  # "icir_shrink" | "legacy_grid"
    LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    BOOTSTRAP_N = 1000
    ICIR_MIN_DAYS_PER_SECTION = 50
    ```

- **`test/rolling_backtest.py`**（修改）
  - 新增 `--ensemble_method {legacy,icir_shrink}`，便于 AB 对比
  - 默认 `icir_shrink`

- **`code/src/pipeline.py` 的 `cmd_predict`**：若 `ensemble_config.json` 的 `method == "icir_shrink_v2"`，直接读 `weights` 字段（和旧逻辑一致），无需改动；仅在日志输出 `method` 字段便于排查

### 5.2 不改动

- `code/src/features/*`、`code/src/models/*`、`code/src/cv/*`、`code/src/data_prep/*`
- `ensemble/portfolio.py`、`ensemble/tradability.py`
- `Dockerfile`、`train.sh`、`test.sh`、`init.sh`

### 5.3 复现性要点

- 所有随机源使用 `np.random.default_rng(config.SEED)`
- SLSQP 本身 deterministic
- Bootstrap 和 Dirichlet 多起点使用独立 `rng` 派生
- 新旧逻辑切换通过 `config.ENSEMBLE_METHOD`（而非环境变量），保证 `pyproject.toml` + `config.py` 固定即结果固定
- `verify_reproducibility.py` 不变（仍对比两次 predict 的 MD5）

---

## 6. 验收标准

| 指标 | 当前 baseline | 新版目标 |
|---|---|---|
| 回测 mean 5 日收益（82 天 tradable+cost 窗口） | +1.00% | **≥ +0.95%**（允许轻微下降以换稳健性） |
| 权重集中度 `max w_m` | 0.90 | **≤ 0.65** |
| Bootstrap 95% CI 最窄模型下界 | 未知 | **≥ 0.05**（若 < 0.05 则报告中建议下阶段删除该模型） |
| LOO 选出的 λ | — | **非 0 非 ∞**（证明正则有效）；若为 0/∞ 则 spec 判定“数据说话”但需在 report 中解释 |
| 训练时间增量 | — | **≤ +5 min**（SLSQP 低维，bootstrap 1000×14 起点 ≈ 数十秒） |
| 复现一致性 | MD5 已一致 | **保持**（两次训练权重完全一致，两次推理 MD5 完全一致） |
| `grid_search` legacy 日志 | — | 与 new 权重并列打印便于对比 |

**回退门**：如果新版在 Phase-A 82 天回测上 mean 5 日收益比 baseline 低 > 5 bp，**不发布**，保留 legacy 权重，把 spec 结论写成“ICIR 在当前样本量下不占优，下阶段待扩窗再评”。

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| ICIR 优化在 5090 OOF 上好，但 4060 基模型漂移致权重不再最优 | 权重基于 rank-normalize 分数，对 score 尺度变化鲁棒；`legacy_weights` 永远保留在 config，一键回退 |
| OOF 样本仍偏少（<100 天），ICIR 方差大 | 正是 λ·KL 的角色；bootstrap CI 强制可视化 |
| SLSQP 卡局部解 | 14 个多起点（等权 + 3 单点 + 10 Dirichlet）取 `J(w)` 最大 |
| OOF 日期和 holdout 分布不同 | λ 仅用 OOF 做 LOO 选，holdout 仅用于最终求 `w*` 时加样本；若 LOO 最优 λ 在 holdout 上显著劣化，日志报警（判据：holdout ICIR < OOF ICIR × 0.5）|
| mixer 权重 CI 全部贴 0 | 诊断报警；下阶段考虑彻底删除 mixer 释放 1/3 训练预算 |
| 与 Phase-A dynamic weights 冲突 | `blend_mode="static"` 为默认；dynamic 路径保留但 README/report 说明不启用 |

---

## 8. 验证流程（plan 阶段会展开）

1. 单测：
   - `optimize_weights_icir` 在合成数据（三路相关信号）上能恢复已知最优
   - `select_lambda_loo` 在强信号数据返回 λ=0，在纯噪声数据返回大 λ
   - `bootstrap_weights` 的 CI 含有真值 ≥ 95%
2. 集成测试：跑一次完整 `pipeline.py train`，检查 `ensemble_config.json` 字段齐全
3. 回测：`test/rolling_backtest.py --ensemble_method icir_shrink` vs `--ensemble_method legacy`
4. 复现：两次 train，diff `ensemble_config.json`；两次 predict，diff `result.csv` MD5
5. 4060 预算演练：本地记录训练/推理耗时 Δ，确认在 +5 min 内

---

## 9. 交付物

- 代码 commit（按 plan 拆分）
- `ensemble_config.json` 新格式 + legacy 字段
- `docs/report.md` 追加 §12 节：集成稳健化实验
- `test/rolling_backtest_icir.csv` 对照结果

---

## 10. 开放问题（plan 阶段解决）

- LOO 的 fold 划分是否按 CV fold 原样使用，还是按日期等分？→ 倾向按 CV fold 原样，保持与训练一致
- λ_grid 是否需要 log-spaced 扩展到 5.0 / 10.0？→ 若 LOO 选到 2.0，则 plan 阶段扩展
- Bootstrap 是否对交易日做 block bootstrap（防 IC 自相关）？→ 默认 i.i.d. 抽样，block 版作为可选；若 OOF IC 自相关 ρ(1) > 0.3 再切 block
