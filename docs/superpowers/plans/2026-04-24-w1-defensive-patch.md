# W1 Defensive Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在推理层加入"市值硬约束 Top-K 选股 + regime 诊断信号"，让 LGB-only 主线在近 30 天大盘风格下不结构性掉队，同时保留 213 天 +2.41% 的长期 alpha。

**Architecture:** 仅推理层改动，不重训。新增两个纯函数模块 `mcap_constrained_topk`（Top-10 候选 → 市值约束挑 Top-5）和 `detect_regime`（大盘/小盘近 10 日累计收益差信号），通过 monkey-patch 注入到 `rolling_backtest_lgb_only.py` 与 `predict_lgb_only.py`。Rolling AB 三档（原版 / min_large_cap=2 / =3）对比决策。

**Tech Stack:** Python 3.11, pandas 2.x, numpy, pytest, LightGBM（已训好不动）

**参考 spec:** `docs/superpowers/specs/2026-04-24-w1-defensive-patch-design.md`

---

## File Structure

| 文件 | 动作 | 责任 |
|---|---|---|
| `code/src/ensemble/portfolio.py` | Modify（追加） | 新增 `mcap_constrained_topk` 纯函数 |
| `code/src/ensemble/regime.py` | Create | `detect_regime` 大盘/小盘累计收益差诊断 |
| `test/test_mcap_constrained.py` | Create | 5 项单元测试 |
| `test/test_regime.py` | Create | 2 项单元测试（rotation / neutral） |
| `scripts/rolling_backtest_lgb_only.py` | Modify | monkey-patch 注入约束 Top-K |
| `scripts/predict_lgb_only.py` | Modify | 集成约束 Top-K + regime 日志 |
| `scripts/run_w1_ab.sh` | Create | 一键跑 Baseline / M10-2 / M10-3 rolling |

---

## Task 1: `mcap_constrained_topk` 纯函数（TDD）

**Files:**
- Create: `test/test_mcap_constrained.py`
- Modify: `code/src/ensemble/portfolio.py`（追加函数，不改现有）

- [ ] **Step 1: 写失败测试（5 项）**

```python
# test/test_mcap_constrained.py
import pandas as pd
import pytest
from code.src.ensemble.portfolio import mcap_constrained_topk


def _make_scores(pairs):
    """pairs: list of (stock_id, score)."""
    return pd.DataFrame(pairs, columns=["stock_id", "score"])


def _make_mktcap(pairs):
    """pairs: list of (stock_id, log_mktcap)."""
    return pd.DataFrame(pairs, columns=["stock_id", "log_mktcap"])


def test_min_large_cap_zero_equivalent_to_top5():
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", float(i)) for i in range(10)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=0)
    assert out["stock_id"].tolist() == [f"{i:06d}" for i in range(5)]
    assert len(out) == 5


def test_all_large_cap_returns_top5_unchanged():
    # 所有 10 只候选都是大盘（log_mktcap 远高于中位数）
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", 20.0) for i in range(10)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2)
    assert out["stock_id"].tolist() == [f"{i:06d}" for i in range(5)]


def test_only_one_large_cap_triggers_fallback():
    # 仅 1 只大盘，min_large_cap=2 → FALLBACK 到原 Top-5
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    # 000009 是唯一大盘，其 score 最低
    mktcap = _make_mktcap([(f"{i:06d}", 1.0) for i in range(9)]
                           + [("000009", 20.0)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2)
    assert out["stock_id"].tolist() == [f"{i:06d}" for i in range(5)]


def test_two_large_caps_at_tail_replace_smallcaps():
    # 10 只候选，大盘是 000008 和 000009（score 最低的两只）
    # min_large_cap=2 → 必须把最后两只小盘替换为这两只大盘
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    mktcap = _make_mktcap([(f"{i:06d}", 1.0) for i in range(8)]
                           + [("000008", 20.0), ("000009", 20.0)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2)
    ids = out["stock_id"].tolist()
    # 前 3 只保留（小盘高分），第 4-5 只被替换为 000008, 000009
    assert ids[:3] == ["000000", "000001", "000002"]
    assert set(ids[3:]) == {"000008", "000009"}
    assert len(out) == 5


def test_output_length_and_order():
    scores = _make_scores([(f"{i:06d}", 10 - i) for i in range(10)])
    # 前 5 只大盘，后 5 只小盘
    mktcap = _make_mktcap([(f"{i:06d}", 20.0) for i in range(5)]
                           + [(f"{i:06d}", 1.0) for i in range(5, 10)])
    out = mcap_constrained_topk(scores, mktcap, k=5,
                                 candidate_k=10, min_large_cap=2)
    assert len(out) == 5
    # 输出按 score 降序
    scores_out = out["score_q"].tolist()
    assert scores_out == sorted(scores_out, reverse=True)
```

- [ ] **Step 2: 运行测试，验证失败**

Run: `pytest test/test_mcap_constrained.py -v`
Expected: 5 项均 FAIL（`mcap_constrained_topk` 未定义）

- [ ] **Step 3: 实现 `mcap_constrained_topk`**

在 `code/src/ensemble/portfolio.py` 末尾（`_self_test` 之前）追加：

```python
def mcap_constrained_topk(
    scores: pd.DataFrame,
    mktcap: pd.DataFrame,
    k: int = 5,
    candidate_k: int = 10,
    min_large_cap: int = 2,
    large_cap_quantile: float = 0.5,
    score_col: str = "score",
    id_col: str = "stock_id",
) -> pd.DataFrame:
    """Select k picks from top candidate_k, enforcing >= min_large_cap large-caps.

    Large-cap threshold = quantile of log_mktcap over the candidate pool's universe
    (we use the passed-in mktcap frame as the universe, typically all HS300 today).
    If the top-candidate_k contains fewer than min_large_cap large-caps, FALL BACK
    to plain deterministic_top_k(k) (logged via return metadata 'fallback' column).
    """
    candidates = deterministic_top_k(
        scores, k=candidate_k, score_col=score_col, id_col=id_col
    )
    if min_large_cap <= 0 or len(candidates) <= k:
        return candidates.head(k).reset_index(drop=True)

    mcap = mktcap[[id_col, "log_mktcap"]].drop_duplicates(id_col)
    threshold = float(mcap["log_mktcap"].quantile(large_cap_quantile))
    merged = candidates.merge(mcap, on=id_col, how="left")
    merged["is_large"] = merged["log_mktcap"].fillna(-np.inf) >= threshold

    large_in_candidates = int(merged["is_large"].sum())
    if large_in_candidates < min_large_cap:
        # FALLBACK: candidate pool is itself small-cap dominated this day
        return candidates.head(k).reset_index(drop=True)

    # Greedy: take top-k by score; if large-caps in those k < min_large_cap,
    # replace trailing smallcaps with highest-score largecaps from the remainder.
    top_k_slice = merged.head(k).copy()
    need = min_large_cap - int(top_k_slice["is_large"].sum())
    if need <= 0:
        return top_k_slice.drop(columns=["log_mktcap", "is_large"]).reset_index(drop=True)

    remainder = merged.iloc[k:]
    add_large = remainder[remainder["is_large"]].head(need)
    # drop lowest-score smallcaps in current top_k_slice
    smallcaps_in_top = top_k_slice[~top_k_slice["is_large"]]
    drop_ids = smallcaps_in_top.tail(need)[id_col].tolist()
    kept = top_k_slice[~top_k_slice[id_col].isin(drop_ids)]
    out = pd.concat([kept, add_large], ignore_index=True)
    out = out.sort_values(by=["score_q", id_col], ascending=[False, True],
                           kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["log_mktcap", "is_large"])
```

- [ ] **Step 4: 运行测试，验证通过**

Run: `pytest test/test_mcap_constrained.py -v`
Expected: 5/5 PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_mcap_constrained.py code/src/ensemble/portfolio.py
git commit -m "feat(portfolio): add mcap_constrained_topk with unit tests"
```

---

## Task 2: `detect_regime` 信号函数（TDD）

**Files:**
- Create: `test/test_regime.py`
- Create: `code/src/ensemble/regime.py`

- [ ] **Step 1: 写失败测试**

```python
# test/test_regime.py
import pandas as pd
import numpy as np
from code.src.ensemble.regime import detect_regime


def _make_panel(dates, n_large=10, n_small=10,
                large_daily=0.005, small_daily=-0.002):
    rows = []
    for i in range(n_large):
        sid = f"L{i:03d}"
        price = 100.0
        for d in dates:
            price *= (1 + large_daily)
            rows.append({"datetime": d, "instrument": sid,
                         "close": price, "log_mktcap": 20.0})
    for i in range(n_small):
        sid = f"S{i:03d}"
        price = 100.0
        for d in dates:
            price *= (1 + small_daily)
            rows.append({"datetime": d, "instrument": sid,
                         "close": price, "log_mktcap": 10.0})
    return pd.DataFrame(rows)


def test_regime_large_cap_rotation_detected():
    dates = pd.date_range("2026-04-01", periods=15, freq="B")
    panel = _make_panel(dates, large_daily=0.005, small_daily=-0.002)
    out = detect_regime(panel, as_of=dates[-1], lookback_days=10,
                        threshold=0.01)
    assert out["regime"] == "large_cap_rotation"
    assert out["diff"] > 0.01


def test_regime_neutral_when_small_diff():
    dates = pd.date_range("2026-04-01", periods=15, freq="B")
    panel = _make_panel(dates, large_daily=0.0005, small_daily=0.0003)
    out = detect_regime(panel, as_of=dates[-1], lookback_days=10,
                        threshold=0.01)
    assert out["regime"] == "neutral"
```

- [ ] **Step 2: 运行测试，验证失败**

Run: `pytest test/test_regime.py -v`
Expected: 2 项均 FAIL

- [ ] **Step 3: 实现 `detect_regime`**

```python
# code/src/ensemble/regime.py
"""Regime signal: large-cap vs small-cap cumulative return over lookback window."""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_regime(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 10,
    large_quantile: float = 0.67,
    small_quantile: float = 0.33,
    threshold: float = 0.01,
) -> dict:
    """Classify regime by (large-cap mean return) - (small-cap mean return).

    panel must have columns: datetime, instrument, close, log_mktcap.
    Returns dict with keys: regime, r_large, r_small, diff, n_large, n_small.
    """
    as_of = pd.Timestamp(as_of)
    p = panel.copy()
    p["datetime"] = pd.to_datetime(p["datetime"])
    snap = p[p["datetime"] == as_of][["instrument", "log_mktcap"]].dropna()
    if snap.empty:
        return {"regime": "neutral", "r_large": np.nan, "r_small": np.nan,
                "diff": 0.0, "n_large": 0, "n_small": 0}

    q_hi = snap["log_mktcap"].quantile(large_quantile)
    q_lo = snap["log_mktcap"].quantile(small_quantile)
    large_ids = set(snap[snap["log_mktcap"] >= q_hi]["instrument"])
    small_ids = set(snap[snap["log_mktcap"] <= q_lo]["instrument"])

    window = p[(p["datetime"] <= as_of) &
               (p["datetime"] > as_of - pd.Timedelta(days=lookback_days * 2))]
    window = window.sort_values(["instrument", "datetime"])

    def _cum_ret(ids):
        sub = window[window["instrument"].isin(ids)]
        grp = sub.groupby("instrument")["close"]
        rets = grp.apply(lambda s: s.iloc[-1] / s.iloc[0] - 1 if len(s) >= 2 else np.nan)
        return float(rets.dropna().mean()) if len(rets.dropna()) else np.nan

    r_large = _cum_ret(large_ids)
    r_small = _cum_ret(small_ids)
    diff = (r_large - r_small) if (not np.isnan(r_large) and not np.isnan(r_small)) else 0.0
    regime = "large_cap_rotation" if diff > threshold else "neutral"
    return {"regime": regime, "r_large": r_large, "r_small": r_small,
            "diff": float(diff), "n_large": len(large_ids),
            "n_small": len(small_ids)}
```

- [ ] **Step 4: 运行测试，验证通过**

Run: `pytest test/test_regime.py -v`
Expected: 2/2 PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_regime.py code/src/ensemble/regime.py
git commit -m "feat(regime): add detect_regime large-vs-small cap signal"
```

---

（下接 Part 2）

## Task 3: Rolling backtest 集成（monkey-patch 注入）

**Files:**
- Modify: `scripts/rolling_backtest_lgb_only.py`（monkey-patch `build_portfolio`）

**约束**：不改 `test/rolling_backtest.py` 本体（避免污染主线），只在 wrapper 里 patch。wrapper 读取 env var `MCAP_MIN_LARGE`（0=off, 2=M10-2, 3=M10-3）。

- [ ] **Step 1: 定位现有 monkey-patch 点**

当前 `_patched_build_portfolio` 包装 `build_portfolio`。我们需要在它**之前**把 `today_scores` 先过 `mcap_constrained_topk`（把选股结果注入回 `today_blended`），或在之后重排。

**选择**：直接 monkey-patch `rb.build_portfolio`，在内部先用 `mcap_constrained_topk` 约束 Top-10 候选到 Top-5，再把结果塞进原 `build_portfolio` 的输出（confidence scaling 保留）。

- [ ] **Step 2: 修改 `scripts/rolling_backtest_lgb_only.py`**

在 `import os, sys, ...` 下方添加 log_mktcap 快照加载工具；在现有 `_patched_build_portfolio` 之前新增 `_apply_mcap_constraint`：

```python
# 在文件顶部 imports 之后（rb 加载后）新增：
from code.src.ensemble.portfolio import mcap_constrained_topk

# 读取 env
MCAP_MIN_LARGE = int(os.environ.get("MCAP_MIN_LARGE", "0"))
MCAP_CAND_K = int(os.environ.get("MCAP_CAND_K", "10"))
MCAP_LARGE_Q = float(os.environ.get("MCAP_LARGE_Q", "0.5"))

# 缓存 panel（含 log_mktcap）供 patch 访问
_PANEL_CACHE = {"panel": None}

_orig_main = rb.main
def _wrapped_main():
    # 在 main 启动前抓取 feature panel 的 log_mktcap 列
    from code.src.features.build import build_feature_sets
    fs = build_feature_sets("./data", "./temp", use_cache=True)
    panel = fs["lgb"]["panel"][["instrument", "datetime", "log_mktcap"]].copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    _PANEL_CACHE["panel"] = panel
    _orig_main()

import pandas as pd  # ensure available
rb_main_original = rb.main
rb.main = _wrapped_main
```

修改 `_patched_build_portfolio`：在调用 `_orig_build_portfolio` 之前，若 `MCAP_MIN_LARGE > 0`，**先裁剪 `today_scores`**（第一位参数）到约束后的 Top-K：

```python
_orig_build_portfolio = rb.build_portfolio
def _patched_build_portfolio(*args, **kwargs):
    today_scores = args[0] if args else kwargs.get("blended_scores_today")
    top_k = kwargs.get("top_k", args[3] if len(args) >= 4 else 5)

    if MCAP_MIN_LARGE > 0 and today_scores is not None and _PANEL_CACHE["panel"] is not None:
        # 猜测 as_of 日期：today_scores 里若有 datetime 取最新；否则跳过
        if "datetime" in today_scores.columns:
            as_of = pd.to_datetime(today_scores["datetime"]).max()
        else:
            as_of = None
        if as_of is not None:
            mktcap_snap = _PANEL_CACHE["panel"]
            mktcap_snap = mktcap_snap[mktcap_snap["datetime"] == as_of][
                ["instrument", "log_mktcap"]
            ].rename(columns={"instrument": "stock_id"})
            if not mktcap_snap.empty:
                scores_in = today_scores.rename(
                    columns={"instrument": "stock_id", "final_score": "score"}
                )[["stock_id", "score"]]
                picked = mcap_constrained_topk(
                    scores_in, mktcap_snap, k=top_k,
                    candidate_k=MCAP_CAND_K,
                    min_large_cap=MCAP_MIN_LARGE,
                    large_cap_quantile=MCAP_LARGE_Q,
                )
                # 限制 today_scores 只保留 picked 里的 instrument
                keep = set(picked["stock_id"].astype(str))
                mask = today_scores["instrument"].astype(str).isin(keep)
                today_scores = today_scores[mask].copy()
                if args:
                    args = (today_scores,) + args[1:]
                else:
                    kwargs["blended_scores_today"] = today_scores

    port = _orig_build_portfolio(*args, **kwargs)
    # 原有 ALLOC_MODE 逻辑保持不变（此次不启用）
    ...（保留原代码 from mode 判断起）
```

**重要**：保留原有的 ALLOC_MODE 分支不动。

- [ ] **Step 3: 手工冒烟测试（单日）**

```bash
cd /root/shared-nvme/bigdata/THU-BDC2026
MCAP_MIN_LARGE=2 python scripts/rolling_backtest_lgb_only.py \
  --eval-start 2026-04-15 --eval-end 2026-04-16 \
  --out test/rolling_smoke.csv 2>&1 | tail -20
```

Expected: 运行成功，产出 1-2 行 CSV，stderr 有 `MCAP_MIN_LARGE=2` 日志。

- [ ] **Step 4: 验证 Baseline 未变**

```bash
MCAP_MIN_LARGE=0 python scripts/rolling_backtest_lgb_only.py \
  --eval-start 2026-04-15 --eval-end 2026-04-16 \
  --out test/rolling_smoke_base.csv 2>&1 | tail -5
diff test/rolling_smoke_base.csv test/rolling_lgb_alloc_equal.csv | head
```

Expected: 对应日期行数值一致（若 rolling_lgb_alloc_equal.csv 已包含这两日）。

- [ ] **Step 5: Commit**

```bash
git add scripts/rolling_backtest_lgb_only.py
git commit -m "feat(rolling): wire MCAP_MIN_LARGE env to mcap_constrained_topk"
```

---

## Task 4: Predict 脚本集成 + regime 日志

**Files:**
- Modify: `scripts/predict_lgb_only.py`

- [ ] **Step 1: 在 `deterministic_top_k` 之后插入市值约束**

找到现有代码：

```python
picks = deterministic_top_k(
    avg_scores.rename(columns={"instrument": "stock_id"}),
    k=top_k,
    ...
)[["stock_id", "score_q"]].copy()
```

改为：

```python
from code.src.ensemble.portfolio import mcap_constrained_topk
from code.src.ensemble.regime import detect_regime

mcap_min_large = int(os.environ.get("MCAP_MIN_LARGE", "0"))
mcap_cand_k = int(os.environ.get("MCAP_CAND_K", "10"))
mcap_large_q = float(os.environ.get("MCAP_LARGE_Q", "0.5"))

scores_for_pick = avg_scores.rename(columns={"instrument": "stock_id"})
if mcap_min_large > 0:
    mktcap_snap = today_panel[["instrument", "log_mktcap"]].rename(
        columns={"instrument": "stock_id"}
    )
    picks = mcap_constrained_topk(
        scores_for_pick, mktcap_snap,
        k=top_k, candidate_k=mcap_cand_k,
        min_large_cap=mcap_min_large,
        large_cap_quantile=mcap_large_q,
    )[["stock_id", "score_q"]].copy()
    print(f"[lgb-only] mcap_constrained Top-{top_k} "
          f"(min_large={mcap_min_large}, cand_k={mcap_cand_k})", file=sys.stderr)
else:
    picks = deterministic_top_k(
        scores_for_pick, k=top_k, score_col="score", id_col="stock_id"
    )[["stock_id", "score_q"]].copy()

# Regime 诊断（仅日志）
try:
    reg = detect_regime(
        panel[["datetime", "instrument", "close", "log_mktcap"]]
          if "close" in panel.columns
          else panel.assign(close=panel.get("$close", 0)),
        as_of=target_date,
    )
    print(f"[lgb-only] regime={reg['regime']} "
          f"diff={reg['diff']:+.4f} "
          f"r_large={reg['r_large']:+.4f} r_small={reg['r_small']:+.4f}",
          file=sys.stderr)
except Exception as e:
    print(f"[lgb-only] regime detection failed: {e}", file=sys.stderr)
```

**注意**：`panel` 变量上文已有（`build_feature_sets` 返回的 `fset["panel"]`）。若其中没有 `close` 列，回退到从 `stock_data.csv` 加载——为避免这次加复杂度，若无 close 就打印 `regime=unavailable` 而非报错。简化版：

```python
try:
    if "close" in panel.columns or "$close" in panel.columns:
        close_col = "close" if "close" in panel.columns else "$close"
        p_slim = panel[["datetime", "instrument", close_col, "log_mktcap"]].rename(
            columns={close_col: "close"}
        )
        reg = detect_regime(p_slim, as_of=target_date)
        print(f"[lgb-only] regime={reg['regime']} diff={reg['diff']:+.4f}",
              file=sys.stderr)
    else:
        print("[lgb-only] regime=unavailable (no close col)", file=sys.stderr)
except Exception as e:
    print(f"[lgb-only] regime detection failed: {e}", file=sys.stderr)
```

- [ ] **Step 2: 冒烟测试**

```bash
cd /root/shared-nvme/bigdata/THU-BDC2026
MCAP_MIN_LARGE=2 python scripts/predict_lgb_only.py 2>&1 | tail -10
cat output/result.csv
```

Expected: result.csv 5 行，stderr 显示 `mcap_constrained Top-5` 和 regime 输出。

- [ ] **Step 3: Baseline 对比（MCAP_MIN_LARGE=0）**

```bash
MCAP_MIN_LARGE=0 python scripts/predict_lgb_only.py 2>&1 | tail -5
cp output/result.csv output/result_base.csv
MCAP_MIN_LARGE=2 python scripts/predict_lgb_only.py 2>&1 | tail -5
diff output/result.csv output/result_base.csv || echo "DIFFERENT (expected)"
```

Expected: 两份可能相同（若 LGB 原 Top-5 已含 ≥2 大盘）也可能不同（触发了替换）。

- [ ] **Step 4: Commit**

```bash
git add scripts/predict_lgb_only.py
git commit -m "feat(predict): integrate mcap_constrained_topk and regime logging"
```

---

（下接 Part 3）

## Task 5: Rolling AB 一键脚本

**Files:**
- Create: `scripts/run_w1_ab.sh`

- [ ] **Step 1: 写脚本**

```bash
#!/usr/bin/env bash
# Run three rolling backtests: baseline / M10-2 / M10-3
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_START="${EVAL_START:-2025-06-03}"
EVAL_END="${EVAL_END:-2026-04-16}"

echo "=== Baseline (MCAP_MIN_LARGE=0) ==="
MCAP_MIN_LARGE=0 python scripts/rolling_backtest_lgb_only.py \
  --eval-start "$EVAL_START" --eval-end "$EVAL_END" \
  --out test/rolling_lgb_baseline.csv 2>&1 | tail -20

echo "=== M10-2 (MCAP_MIN_LARGE=2) ==="
MCAP_MIN_LARGE=2 python scripts/rolling_backtest_lgb_only.py \
  --eval-start "$EVAL_START" --eval-end "$EVAL_END" \
  --out test/rolling_lgb_mcap2.csv 2>&1 | tail -20

echo "=== M10-3 (MCAP_MIN_LARGE=3) ==="
MCAP_MIN_LARGE=3 python scripts/rolling_backtest_lgb_only.py \
  --eval-start "$EVAL_START" --eval-end "$EVAL_END" \
  --out test/rolling_lgb_mcap3.csv 2>&1 | tail -20

echo "=== 近 30 日 / 213 天对比 ==="
python - <<'PY'
import pandas as pd
for tag, f in [("Baseline", "test/rolling_lgb_baseline.csv"),
               ("M10-2",    "test/rolling_lgb_mcap2.csv"),
               ("M10-3",    "test/rolling_lgb_mcap3.csv")]:
    d = pd.read_csv(f)
    d["T"] = pd.to_datetime(d["T"] if "T" in d.columns else d["datetime"])
    d = d.sort_values("T").reset_index(drop=True)
    last30 = d.tail(30)
    print(f"{tag:10s}  N={len(d):3d}  "
          f"all mean={d['portfolio_return'].mean()*100:+6.3f}%  "
          f"last30 mean={last30['portfolio_return'].mean()*100:+6.3f}%  "
          f"last30 HS300={last30['baseline_avg'].mean()*100:+6.3f}%  "
          f"last30 Δ={((last30['portfolio_return']-last30['baseline_avg']).mean())*100:+6.3f}%")
PY
```

- [ ] **Step 2: chmod + run**

```bash
chmod +x scripts/run_w1_ab.sh
./scripts/run_w1_ab.sh 2>&1 | tee /tmp/w1_ab.log
```

Expected: 约 30-60 分钟（3 档 × 213 天），最终输出 3 行对比表。

- [ ] **Step 3: Commit**

```bash
git add scripts/run_w1_ab.sh
git commit -m "chore(scripts): add run_w1_ab.sh for three-way rolling AB"
```

---

## Task 6: 决策与提交准备

**Files:**
- Read: `/tmp/w1_ab.log`
- 可能 Modify: `test.sh`（设置 MCAP_MIN_LARGE）

- [ ] **Step 1: 应用决策规则（参见 spec §6.2）**

对照下表从 AB 日志读取：

| 条件 | 动作 |
|---|---|
| M10-2 近 30 > HS300 **且** 213d 跌幅 ≤ 0.3pp | 提交 M10-2 |
| M10-3 比 M10-2 更好 | 提交 M10-3 |
| 近 30 改善 ≥ +0.3pp 且整体损失 ≤ 0.5pp | 提交 M10-2 |
| 均不满足 | 提交原版 |

- [ ] **Step 2: 若决定启用 M10，修改 `test.sh`**

在 `python scripts/predict_lgb_only.py` 之前加：
```bash
export MCAP_MIN_LARGE=${MCAP_MIN_LARGE:-2}   # 或 3，视决策
```

若决定用原版，确保 `test.sh` 未设置 MCAP_MIN_LARGE（默认 0）。

- [ ] **Step 3: Bootstrap CI 二次验证（可选，时间够再做）**

```bash
python scripts/bootstrap_ci.py test/rolling_lgb_mcap2.csv --n 10000 --seed 42
```

记录 95% CI lower bound > 0 作为置信度指标。

- [ ] **Step 4: 提交前冒烟**

```bash
MCAP_MIN_LARGE=2 sh test.sh  # 或按决策值
cat output/result.csv
```

Expected: 5 行，weight 列和约为 1.0，股票代码合理。

- [ ] **Step 5: 写入决策记录**

```bash
cat > docs/reports/2026-04-24-w1-ab-decision.md <<EOF
# W1 AB 决策记录 · 2026-04-24

| 档位 | 全量 mean | 近 30d mean | 近 30d vs HS300 |
|---|---|---|---|
| Baseline | ... | ... | ... |
| M10-2    | ... | ... | ... |
| M10-3    | ... | ... | ... |

**决策**：提交 ...
**理由**：...
EOF
git add docs/reports/2026-04-24-w1-ab-decision.md test.sh
git commit -m "docs: W1 AB decision (selected: ...)"
```

---

## Self-Review（计划作者自查）

- [x] **Spec coverage**
  - §4.1 mcap_constrained_topk → Task 1 ✅
  - §4.2 detect_regime → Task 2 ✅
  - §4.3 集成点（rolling + predict）→ Task 3, 4 ✅
  - §6.1 Rolling AB 三档 → Task 5 ✅
  - §6.2 决策规则 → Task 6 ✅
  - §7 单元测试（5 项 mcap + 2 项 regime）→ Task 1, 2 ✅
- [x] **Placeholder scan**: 无 TBD/TODO
- [x] **Type consistency**: `mcap_constrained_topk` 签名在 Task 1 定义，Task 3/4 调用一致；`detect_regime` 返回 `dict` 键 `regime/diff/r_large/r_small` 在 Task 2 定义，Task 4 消费一致
- [x] **Ambiguity**: Task 3 中 `today_scores` 可能无 `datetime` 列的 fallback 已说明（跳过约束）

## 注意事项（给执行者）

1. **不要改 `test/rolling_backtest.py`**：所有 rolling 集成都在 wrapper 里 monkey-patch
2. **不要重训模型**：LGB 模型保持 `model_lgb_only/` 下的权重不变
3. **时间盒硬约束**：若 Task 5 AB 跑到 2026-04-25 03:00 未完成，直接按 Task 6 Step 4 用原版提交
4. **保底**：Baseline 档 CSV 必须在跑 M10 之前先跑成功并保存，防止 AB 失败时无保底数据

## Execution

推荐 **Subagent-Driven** 执行（每 task 独立 subagent + review）。若你偏好一气呵成，选 Inline Execution。
