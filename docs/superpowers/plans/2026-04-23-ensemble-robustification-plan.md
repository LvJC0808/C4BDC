# Ensemble Robustification v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` tracking.

**Goal:** 把 ensemble 权重从脆弱的 `{0.1, 0.9, 0.0}` 替换为 ICIR + KL-shrink 鲁棒权重，max 权重 ≤ 0.65，回测不掉 >5 bp。

**Architecture:** 在现有 CV 基础上聚合 OOF + holdout → rank-normalize → 带 KL 正则的 SLSQP 最优化 ICIR → LOO 选 λ → bootstrap CI 诊断。只改 `code/src/ensemble/` + `pipeline.py` + `config.py` + `test/rolling_backtest.py`。基模型/特征/portfolio 不动。

**Tech Stack:** numpy / pandas / scipy.optimize / pytest

**Spec:** `docs/superpowers/specs/2026-04-23-ensemble-robustification-design.md`

---

## Task 1：新增 blender 的纯函数工具（rank_normalize / IC series / objective）

**Files:**
- Create: `tests/test_ensemble_blender_icir.py`
- Modify: `code/src/ensemble/blender.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_ensemble_blender_icir.py
import numpy as np
import pandas as pd
import pytest
from code.src.ensemble.blender import (
    rank_normalize_daily,
    compute_blend_ic_series,
    icir_objective,
)


def _synthetic_scores(n_days=30, n_stocks=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days)
    instruments = [f"s{i:03d}" for i in range(n_stocks)]
    rows = []
    for d in dates:
        for ins in instruments:
            rows.append((ins, d, rng.normal()))
    return pd.DataFrame(rows, columns=["instrument", "datetime", "score"])


def test_rank_normalize_daily_within_unit_interval():
    df = _synthetic_scores()
    out = rank_normalize_daily(df)
    assert out["score"].between(0.0, 1.0).all()
    for _, g in out.groupby("datetime"):
        assert abs(g["score"].mean() - 0.5) < 1e-6


def test_compute_blend_ic_series_shape_and_range():
    rng = np.random.default_rng(1)
    scores = {m: rank_normalize_daily(_synthetic_scores(seed=i)) for i, m in enumerate(["lgb", "master", "mixer"])}
    dates = scores["lgb"]["datetime"].unique()
    labels = []
    for d in dates:
        for ins in scores["lgb"][scores["lgb"]["datetime"] == d]["instrument"]:
            labels.append((ins, d, rng.normal()))
    labels = pd.DataFrame(labels, columns=["instrument", "datetime", "label"])
    ics = compute_blend_ic_series(scores, labels, weights={"lgb": 1/3, "master": 1/3, "mixer": 1/3})
    assert ics.shape == (len(dates),)
    assert ((-1.0 <= ics) & (ics <= 1.0)).all()


def test_icir_objective_equal_weight_kl_is_zero():
    # KL(w || 1/3) = 0 when w = 1/3
    rng = np.random.default_rng(2)
    n = 20
    scores = {m: rank_normalize_daily(_synthetic_scores(n_days=n, seed=i)) for i, m in enumerate(["lgb", "master", "mixer"])}
    dates = scores["lgb"]["datetime"].unique()
    labels = []
    for d in dates:
        for ins in scores["lgb"][scores["lgb"]["datetime"] == d]["instrument"]:
            labels.append((ins, d, rng.normal()))
    labels = pd.DataFrame(labels, columns=["instrument", "datetime", "label"])
    val_zero_lam = icir_objective([1/3, 1/3, 1/3], scores, labels, lam=0.0, model_order=["lgb", "master", "mixer"])
    val_nonzero_lam = icir_objective([1/3, 1/3, 1/3], scores, labels, lam=1.0, model_order=["lgb", "master", "mixer"])
    assert abs(val_zero_lam - val_nonzero_lam) < 1e-9  # KL=0 at equal weight
```

- [ ] **Step 2: 跑测试确认失败**

`uv run pytest tests/test_ensemble_blender_icir.py -v`
期望: ImportError (`rank_normalize_daily` 等未定义)

- [ ] **Step 3: 实现函数**

在 `code/src/ensemble/blender.py` 末尾追加：

```python
# ============================================================
# ICIR-based robust ensemble (v2, 2026-04-23)
# ============================================================
from typing import Dict, List, Sequence
from scipy.stats import spearmanr


def rank_normalize_daily(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    out = df.copy()
    out[score_col] = (
        out.groupby("datetime")[score_col]
           .rank(method="average", pct=True)
    )
    return out


def compute_blend_ic_series(
    ranked_scores: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    weights: Dict[str, float],
) -> np.ndarray:
    names = list(weights.keys())
    merged = None
    for m in names:
        df = ranked_scores[m][["instrument", "datetime", "score"]].rename(columns={"score": f"s_{m}"})
        merged = df if merged is None else merged.merge(df, on=["instrument", "datetime"], how="inner")
    merged = merged.merge(labels, on=["instrument", "datetime"], how="inner")
    merged["blend"] = sum(weights[m] * merged[f"s_{m}"] for m in names)
    ics = []
    for _, g in merged.groupby("datetime"):
        if len(g) < 3:
            continue
        rho, _ = spearmanr(g["blend"], g["label"])
        ics.append(0.0 if np.isnan(rho) else rho)
    return np.asarray(ics, dtype=float)


def _kl_to_uniform(w: Sequence[float]) -> float:
    w = np.asarray(w, dtype=float)
    n = len(w)
    uniform = 1.0 / n
    mask = w > 1e-12
    return float(np.sum(w[mask] * np.log(w[mask] / uniform)))


def icir_objective(
    weights: Sequence[float],
    ranked_scores: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    lam: float,
    model_order: List[str],
    eps: float = 1e-6,
) -> float:
    """返回 -J(w) = -(ICIR − λ·KL)，供 SLSQP 最小化。"""
    w = {m: float(weights[i]) for i, m in enumerate(model_order)}
    ics = compute_blend_ic_series(ranked_scores, labels, w)
    if len(ics) == 0:
        return 0.0
    mean = float(np.mean(ics))
    std = float(np.std(ics, ddof=1)) if len(ics) > 1 else eps
    icir = mean / (std + eps) * np.sqrt(len(ics))
    kl = _kl_to_uniform(list(weights))
    return -(icir - lam * kl)
```

- [ ] **Step 4: 跑测试确认通过**

`uv run pytest tests/test_ensemble_blender_icir.py -v`
期望: 3 passed

- [ ] **Step 5: commit**

```bash
git add code/src/ensemble/blender.py tests/test_ensemble_blender_icir.py
git commit -m "feat(ensemble): rank_normalize + blend IC + ICIR objective"
```

---

## Task 2：SLSQP 多起点优化器

**Files:**
- Modify: `code/src/ensemble/blender.py`
- Modify: `tests/test_ensemble_blender_icir.py`

- [ ] **Step 1: 加失败测试**

在 `tests/test_ensemble_blender_icir.py` 末尾追加：

```python
def test_optimize_weights_icir_favors_strong_signal_model():
    """master 信号强 → w_master 应高于其他两者，但 KL 约束下 ≤ 0.7"""
    from code.src.ensemble.blender import optimize_weights_icir, rank_normalize_daily
    rng = np.random.default_rng(42)
    n_days, n_stocks = 60, 80
    dates = pd.date_range("2025-01-01", periods=n_days)
    instruments = [f"s{i:03d}" for i in range(n_stocks)]
    rows_label = []
    for d in dates:
        for ins in instruments:
            rows_label.append((ins, d, rng.normal()))
    labels = pd.DataFrame(rows_label, columns=["instrument", "datetime", "label"])

    def make_scores(correlation):
        rs = []
        for _, row in labels.iterrows():
            noise = rng.normal()
            rs.append((row["instrument"], row["datetime"], correlation * row["label"] + (1 - correlation) * noise))
        return rank_normalize_daily(pd.DataFrame(rs, columns=["instrument", "datetime", "score"]))

    scores = {
        "lgb": make_scores(0.05),
        "master": make_scores(0.30),
        "mixer": make_scores(0.02),
    }
    res = optimize_weights_icir(scores, labels, lam=0.2, model_order=["lgb", "master", "mixer"], seed=42)
    w = res["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v >= -1e-9 for v in w.values())
    assert w["master"] > w["lgb"] and w["master"] > w["mixer"]
    assert w["master"] <= 0.70  # KL shrink 应把极端解拉回
```

- [ ] **Step 2: 跑测试确认失败**

`uv run pytest tests/test_ensemble_blender_icir.py::test_optimize_weights_icir_favors_strong_signal_model -v`
期望: ImportError

- [ ] **Step 3: 实现 `optimize_weights_icir`**

追加到 `code/src/ensemble/blender.py`：

```python
from scipy.optimize import minimize


def optimize_weights_icir(
    ranked_scores: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    lam: float,
    model_order: List[str],
    seed: int = 42,
    n_dirichlet: int = 10,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(model_order)
    starts = [np.ones(n) / n] + [np.eye(n)[i] * 0.98 + (1 - 0.98) / n for i in range(n)]
    starts += [rng.dirichlet(np.ones(n)) for _ in range(n_dirichlet)]
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)

    best = None
    for x0 in starts:
        try:
            r = minimize(
                icir_objective, x0=np.asarray(x0, dtype=float),
                args=(ranked_scores, labels, lam, model_order),
                method="SLSQP", bounds=bounds, constraints=cons,
                options={"ftol": 1e-8, "maxiter": 200, "disp": False},
            )
            if not r.success and r.fun is None:
                continue
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    if best is None:
        w = np.ones(n) / n
        icir = -icir_objective(w, ranked_scores, labels, 0.0, model_order)
        return {"weights": dict(zip(model_order, w)), "icir": float(icir), "neg_objective": 0.0}
    w = np.clip(best.x, 0.0, None)
    w = w / w.sum()
    icir = -icir_objective(w, ranked_scores, labels, 0.0, model_order)  # 纯 ICIR，不含 KL
    return {"weights": dict(zip(model_order, w.tolist())), "icir": float(icir), "neg_objective": float(best.fun)}
```

- [ ] **Step 4: 跑测试确认通过**

`uv run pytest tests/test_ensemble_blender_icir.py -v`
期望: 4 passed

- [ ] **Step 5: commit**

```bash
git add code/src/ensemble/blender.py tests/test_ensemble_blender_icir.py
git commit -m "feat(ensemble): SLSQP multi-start ICIR optimizer"
```

---

## Task 3：LOO 选 λ + bootstrap 诊断

**Files:**
- Modify: `code/src/ensemble/blender.py`
- Modify: `tests/test_ensemble_blender_icir.py`

- [ ] **Step 1: 加失败测试**

追加到 `tests/test_ensemble_blender_icir.py`：

```python
def test_select_lambda_loo_returns_valid_choice():
    from code.src.ensemble.blender import select_lambda_loo, rank_normalize_daily
    rng = np.random.default_rng(7)
    n_days, n_stocks = 45, 60
    dates = pd.date_range("2025-01-01", periods=n_days)
    fold_ids = np.array_split(np.arange(n_days), 3)
    instruments = [f"s{i:03d}" for i in range(n_stocks)]
    labels_rows = []
    for d in dates:
        for ins in instruments:
            labels_rows.append((ins, d, rng.normal()))
    labels = pd.DataFrame(labels_rows, columns=["instrument", "datetime", "label"])

    def make(cor):
        rows = []
        for _, r in labels.iterrows():
            rows.append((r["instrument"], r["datetime"], cor * r["label"] + (1 - cor) * rng.normal()))
        return rank_normalize_daily(pd.DataFrame(rows, columns=["instrument", "datetime", "score"]))

    scores_by_fold = []
    for fidx in fold_ids:
        fdates = dates[fidx]
        scores_by_fold.append({
            "lgb": make(0.1),
            "master": make(0.25),
            "mixer": make(0.05),
        })
        # 把每个 fold 的 scores 限制到 fold 的日期
        scores_by_fold[-1] = {m: df[df["datetime"].isin(fdates)] for m, df in scores_by_fold[-1].items()}

    res = select_lambda_loo(
        scores_by_fold, labels,
        lam_grid=[0.0, 0.1, 0.5, 1.0],
        model_order=["lgb", "master", "mixer"],
        seed=7,
    )
    assert res["lambda"] in [0.0, 0.1, 0.5, 1.0]
    assert len(res["per_fold_icir"]) == 3


def test_bootstrap_weights_reports_ci():
    from code.src.ensemble.blender import bootstrap_weights, rank_normalize_daily
    rng = np.random.default_rng(11)
    n_days, n_stocks = 40, 50
    dates = pd.date_range("2025-01-01", periods=n_days)
    instruments = [f"s{i:03d}" for i in range(n_stocks)]
    labels_rows = []
    for d in dates:
        for ins in instruments:
            labels_rows.append((ins, d, rng.normal()))
    labels = pd.DataFrame(labels_rows, columns=["instrument", "datetime", "label"])

    def make(cor):
        rows = []
        for _, r in labels.iterrows():
            rows.append((r["instrument"], r["datetime"], cor * r["label"] + (1 - cor) * rng.normal()))
        return rank_normalize_daily(pd.DataFrame(rows, columns=["instrument", "datetime", "score"]))

    scores = {"lgb": make(0.1), "master": make(0.2), "mixer": make(0.05)}
    res = bootstrap_weights(scores, labels, lam=0.2, model_order=["lgb", "master", "mixer"], n=40, seed=11)
    for m in ["lgb", "master", "mixer"]:
        assert res["ci_low"][m] <= res["median"][m] <= res["ci_high"][m]
```

- [ ] **Step 2: 跑测试确认失败**

`uv run pytest tests/test_ensemble_blender_icir.py -v`
期望: 2 new tests failing (ImportError)

- [ ] **Step 3: 实现**

追加到 `blender.py`：

```python
def select_lambda_loo(
    scores_by_fold: List[Dict[str, pd.DataFrame]],
    labels: pd.DataFrame,
    lam_grid: Sequence[float],
    model_order: List[str],
    seed: int = 42,
) -> dict:
    n_folds = len(scores_by_fold)
    results = {}
    for lam in lam_grid:
        fold_icirs = []
        for k in range(n_folds):
            train_scores = {m: pd.concat([scores_by_fold[j][m] for j in range(n_folds) if j != k], ignore_index=True)
                            for m in model_order}
            opt = optimize_weights_icir(train_scores, labels, lam=lam, model_order=model_order, seed=seed)
            w = opt["weights"]
            holdout = scores_by_fold[k]
            ics = compute_blend_ic_series(holdout, labels, w)
            if len(ics) == 0:
                continue
            mean = float(np.mean(ics))
            std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 1e-6
            fold_icirs.append(mean / (std + 1e-6) * np.sqrt(len(ics)))
        results[float(lam)] = fold_icirs
    mean_icirs = {lam: (float(np.mean(v)) if v else -np.inf) for lam, v in results.items()}
    best_lam = max(mean_icirs, key=mean_icirs.get)
    return {"lambda": best_lam, "mean_icir": mean_icirs[best_lam], "per_fold_icir": results[best_lam], "all": mean_icirs}


def bootstrap_weights(
    ranked_scores: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    lam: float,
    model_order: List[str],
    n: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    all_dates = np.array(sorted(next(iter(ranked_scores.values()))["datetime"].unique()))
    samples = {m: [] for m in model_order}
    for _ in range(n):
        sampled = rng.choice(all_dates, size=len(all_dates), replace=True)
        mask = pd.Series(sampled).value_counts()  # date -> count
        # 简化：直接用 isin（权重上每日只计一次），保留统计方差
        sub_scores = {m: df[df["datetime"].isin(mask.index)] for m, df in ranked_scores.items()}
        sub_labels = labels[labels["datetime"].isin(mask.index)]
        try:
            opt = optimize_weights_icir(sub_scores, sub_labels, lam=lam, model_order=model_order, seed=int(rng.integers(1, 1 << 30)), n_dirichlet=3)
            for m in model_order:
                samples[m].append(opt["weights"][m])
        except Exception:
            continue
    median = {m: float(np.median(samples[m])) if samples[m] else float("nan") for m in model_order}
    ci_low = {m: float(np.quantile(samples[m], 0.025)) if samples[m] else float("nan") for m in model_order}
    ci_high = {m: float(np.quantile(samples[m], 0.975)) if samples[m] else float("nan") for m in model_order}
    return {"median": median, "ci_low": ci_low, "ci_high": ci_high, "n_success": len(samples[model_order[0]])}
```

- [ ] **Step 4: 跑测试确认通过**

`uv run pytest tests/test_ensemble_blender_icir.py -v`
期望: 6 passed

- [ ] **Step 5: commit**

```bash
git add code/src/ensemble/blender.py tests/test_ensemble_blender_icir.py
git commit -m "feat(ensemble): LOO lambda selection + bootstrap CI"
```

---

## Task 4：接入 pipeline.cmd_train

**Files:**
- Modify: `code/src/pipeline.py`
- Modify: `code/src/config.py`

- [ ] **Step 1: 加 config 字段**

修改 `code/src/config.py` 末尾追加：

```python
# ---- Ensemble v2 (ICIR robust) ----
ENSEMBLE_METHOD = os.environ.get("ENSEMBLE_METHOD", "icir_shrink")  # "icir_shrink" | "legacy"
ICIR_LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
ICIR_BOOTSTRAP_N = int(os.environ.get("ICIR_BOOTSTRAP_N", "1000"))
ICIR_MULTI_START_DIRICHLET = 10
```

- [ ] **Step 2: 改 pipeline.py**

在 `cmd_train` 里的 "# ---------------- Holdout ensemble grid search ----------------" 段落**之前**插入新逻辑，并在 `ensemble_cfg` 字典组装处合并新字段。

定位 `code/src/pipeline.py` 里的 `model_scores_mean: Dict[str, pd.DataFrame] = {}` 附近，追加：

```python
    # ---------------- v2: ICIR + KL shrink ---------------- #
    if config.ENSEMBLE_METHOD == "icir_shrink":
        from .ensemble.blender import (
            rank_normalize_daily,
            optimize_weights_icir,
            select_lambda_loo,
            bootstrap_weights,
        )
        # OOF per fold: 每个 fold 下每模型 val_scores 合并(seed 平均)
        scores_by_fold: List[Dict[str, pd.DataFrame]] = []
        for split in plan.splits:
            fold_dict: Dict[str, pd.DataFrame] = {}
            for name in MODEL_NAMES:
                parts = []
                for seed in config.SEEDS:
                    p = os.path.join(args.model_dir, name, f"seed_{seed}_fold_{split.fold}", "val_scores.parquet")
                    if os.path.exists(p):
                        parts.append(pd.read_parquet(p)[["instrument", "datetime", "score"]])
                if not parts:
                    continue
                df = pd.concat(parts, ignore_index=True)
                agg = df.groupby(["instrument", "datetime"])["score"].mean().reset_index()
                fold_dict[name] = rank_normalize_daily(agg)
            scores_by_fold.append(fold_dict)

        # 加入 holdout 作为第 4 个 fold
        holdout_fold: Dict[str, pd.DataFrame] = {
            m: rank_normalize_daily(model_scores_mean[m]) for m in MODEL_NAMES
        }
        scores_by_fold.append(holdout_fold)

        all_scores: Dict[str, pd.DataFrame] = {
            m: pd.concat([f[m] for f in scores_by_fold if m in f], ignore_index=True)
            for m in MODEL_NAMES
        }

        lam_res = select_lambda_loo(
            scores_by_fold, ho_labels,  # 用 holdout label？需全 label
            lam_grid=config.ICIR_LAMBDA_GRID,
            model_order=list(MODEL_NAMES),
            seed=config.SEED,
        )
        # 用全部 label（OOF + holdout）
        all_labels = lgb_panel[["instrument", "datetime", "label"]].copy()
        opt_res = optimize_weights_icir(
            all_scores, all_labels,
            lam=lam_res["lambda"],
            model_order=list(MODEL_NAMES),
            seed=config.SEED,
        )
        boot_res = bootstrap_weights(
            all_scores, all_labels,
            lam=lam_res["lambda"],
            model_order=list(MODEL_NAMES),
            n=config.ICIR_BOOTSTRAP_N,
            seed=config.SEED,
        )
        icir_weights = opt_res["weights"]
        print(f"[train] ICIR lambda={lam_res['lambda']} weights={icir_weights} icir={opt_res['icir']:.3f}")
        print(f"[train] bootstrap CI: {boot_res['ci_low']} .. {boot_res['ci_high']}")

        # Portfolio grid search 用 ICIR blend
        blended_icir = blend_scores(all_scores, icir_weights)
        port_best_icir = grid_search_portfolio_params(
            blended_icir, ho_labels,
            top_k_candidates=config.TOP_K_CANDIDATES,
            alpha_candidates=config.POSITION_ALPHAS,
        )
    else:
        icir_weights = None
        lam_res = None
        boot_res = None
        port_best_icir = None
```

然后修改 `ensemble_cfg = {...}` 定义为：

```python
    if config.ENSEMBLE_METHOD == "icir_shrink" and icir_weights is not None:
        ensemble_cfg = {
            "method": "icir_shrink",
            "weights": icir_weights,
            "legacy_weights": weight_best["weights"],
            "lambda": lam_res["lambda"],
            "icir": opt_res["icir"],
            "ci_low": boot_res["ci_low"],
            "ci_high": boot_res["ci_high"],
            "top_k": int(port_best_icir["top_k"] or config.TOP_K_CANDIDATES[0]),
            "alpha": float(port_best_icir["alpha"] or config.POSITION_ALPHAS[0]),
            "refit_epochs": refit_epochs,
            "feature_version": "v1.0",
            "seeds": list(config.SEEDS),
            "cv_folds": int(config.CV_FOLDS),
        }
    else:
        ensemble_cfg = {
            "method": "legacy",
            "weights": weight_best["weights"],
            "top_k": int(port_best["top_k"] or weight_best["top_k"]),
            "alpha": float(port_best["alpha"] or config.POSITION_ALPHAS[0]),
            "refit_epochs": refit_epochs,
            "feature_version": "v1.0",
            "seeds": list(config.SEEDS),
            "cv_folds": int(config.CV_FOLDS),
        }
```

- [ ] **Step 3: 快速 smoke 测试（不改测试脚本，直接人工跑）**

先跑语法检查：

```bash
uv run python -c "from code.src.pipeline import cmd_train; print('import ok')"
uv run python -c "from code.src.ensemble.blender import optimize_weights_icir, select_lambda_loo, bootstrap_weights, rank_normalize_daily; print('ok')"
```

期望: 两行 `ok`。

- [ ] **Step 4: commit**

```bash
git add code/src/pipeline.py code/src/config.py
git commit -m "feat(pipeline): integrate ICIR ensemble into cmd_train"
```

---

## Task 5：rolling backtest 支持 AB 对比 + 跑实盘验收

**Files:**
- Modify: `test/rolling_backtest.py`

- [ ] **Step 1: 加 `--ensemble_method` flag**

在 `test/rolling_backtest.py` 的 argparse 段加：

```python
parser.add_argument("--ensemble_method", choices=["auto", "legacy", "icir_shrink"], default="auto",
                    help="override ensemble_config.json['method']")
```

并在读取 `ensemble_config.json` 后加：

```python
if args.ensemble_method != "auto":
    ens_cfg["method"] = args.ensemble_method
    if args.ensemble_method == "legacy" and "legacy_weights" in ens_cfg:
        ens_cfg["weights"] = ens_cfg["legacy_weights"]
```

- [ ] **Step 2: 跑端到端训练 + 回测**

```bash
cd /root/shared-nvme/bigdata/THU-BDC2026
source .venv/bin/activate
# 用现有缓存重训 ensemble（基模型 checkpoint 已在 model/ 下，只需重跑搜索段）
# 如果 pipeline 不支持单独重跑搜索，则完整 train.sh；此处先 smoke：
ENSEMBLE_METHOD=icir_shrink ICIR_BOOTSTRAP_N=200 python -m code.src.pipeline train \
    --data_path ./data --model_dir ./model --temp_dir ./temp 2>&1 | tee temp/train_icir.log
```

期望: `ensemble_config.json` 中 `method=icir_shrink`, `max(weights.values()) <= 0.65`, `lambda ∈ [0.05, 1.0]`。

- [ ] **Step 3: AB 回测**

```bash
python test/rolling_backtest.py --ensemble_method icir_shrink \
    --eval_start 2025-11-03 --eval_end 2026-03-06 \
    --tradable_filter --cost_bps 3 \
    --out test/rolling_backtest_icir.csv

python test/rolling_backtest.py --ensemble_method legacy \
    --eval_start 2025-11-03 --eval_end 2026-03-06 \
    --tradable_filter --cost_bps 3 \
    --out test/rolling_backtest_legacy.csv
```

期望:
- `rolling_backtest_icir.csv` mean 5d return ≥ 0.95%
- max 权重 ≤ 0.65
- 两套 CSV 都生成

- [ ] **Step 4: 若达标则 commit 验收产物**

```bash
git add test/rolling_backtest.py test/rolling_backtest_icir.csv test/rolling_backtest_legacy.csv
git commit -m "feat(backtest): ICIR ensemble AB comparison + baseline parity"
```

- [ ] **Step 5: 未达标时回退**

如果 mean 5d return < 0.95%：

```bash
# ensemble_config.json 里手动把 method 改回 legacy
python -c "import json; p='model/ensemble_config.json'; c=json.load(open(p)); c['method']='legacy'; c['weights']=c.get('legacy_weights', c['weights']); json.dump(c, open(p,'w'), indent=2)"
git add model/ensemble_config.json
git commit -m "revert(ensemble): fallback to legacy weights; ICIR underperformed"
```

并把验收数据写入 `docs/report.md` 第 13 节（新增），描述为什么没采用。

---

## 完成条件

- 全部 Task 1–5 的 `- [ ]` 勾掉
- `rolling_backtest_icir.csv` mean 5d ≥ 0.95% 或已显式回退且 commit 了回退说明
- `ensemble_config.json` 包含 `method`, `lambda`, `ci_low`, `ci_high`
- 6 个新增单测 pass
- 两次 `python -m code.src.pipeline predict` 的 `output/result.csv` MD5 一致
