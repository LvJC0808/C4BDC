# Phase-A Ensemble Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve ensemble layer with robust weight selection, tradability filtering, and dynamic time-varying weights — without retraining any model.

**Architecture:** Three independent modules added to `code/src/ensemble/`; pipeline.py and rolling_backtest.py updated to use them conditionally. All changes backward-compatible with existing `ensemble_config.json`.

**Tech Stack:** Python 3.12, pandas, numpy, scipy (spearmanr)

**Spec:** `docs/superpowers/specs/2026-04-22-phase-a-ensemble-robustness-design.md`

---

## Task 1: Tradability filter module

**Files:**
- Create: `code/src/ensemble/tradability.py`

- [ ] **Step 1: Write tradability.py**

```python
"""Tradability filter: exclude stocks hitting price limits on target date."""
from __future__ import annotations
import numpy as np
import pandas as pd


def is_limit_up(pct_chg: float, code: str, threshold_main: float = 9.9,
                threshold_special: float = 19.9) -> bool:
    """Return True if stock hit upper price limit."""
    # ChiNext (300xxx) and STAR (688xxx) have 20% limits
    if code.startswith("30") or code.startswith("68"):
        return pct_chg >= threshold_special
    return pct_chg >= threshold_main


def is_limit_down(pct_chg: float, code: str, threshold_main: float = -9.9,
                  threshold_special: float = -19.9) -> bool:
    if code.startswith("30") or code.startswith("68"):
        return pct_chg <= threshold_special
    return pct_chg <= threshold_main


def get_tradable_ids(stock_df: pd.DataFrame, target_date: pd.Timestamp,
                     exclude_limit_up: bool = True,
                     exclude_limit_down: bool = True) -> set:
    """Return set of stock codes tradable on target_date.

    A stock is untradable if on target_date it hit the price limit
    (涨停 or 跌停), because next-day open will gap and may not fill.

    Parameters
    ----------
    stock_df : DataFrame with columns 股票代码, 日期, 涨跌幅
    target_date : the T date (we check T's price change)
    """
    stock_df = stock_df.copy()
    stock_df["日期"] = pd.to_datetime(stock_df["日期"])
    stock_df["股票代码"] = stock_df["股票代码"].astype(str).str.zfill(6)
    day = stock_df[stock_df["日期"] == target_date]
    if day.empty:
        return set(stock_df["股票代码"].unique())

    excluded = set()
    for _, row in day.iterrows():
        code = row["股票代码"]
        pct = float(row.get("涨跌幅", 0))
        if exclude_limit_up and is_limit_up(pct, code):
            excluded.add(code)
        if exclude_limit_down and is_limit_down(pct, code):
            excluded.add(code)
    all_codes = set(day["股票代码"].unique())
    return all_codes - excluded


def filter_tradable_topk(scores_df: pd.DataFrame, tradable_ids: set,
                         top_k: int, expand: int = 5,
                         score_col: str = "final_score",
                         id_col: str = "instrument") -> list:
    """Pick top-K from scores, skipping untradable stocks.

    If after filtering fewer than max(top_k-2, 1) remain, relax to
    whatever is available.
    """
    ranked = scores_df.sort_values(score_col, ascending=False)
    tradable = [r for _, r in ranked.iterrows()
                if r[id_col] in tradable_ids]
    n = max(min(top_k, len(tradable)), max(top_k - 2, 1))
    return [r[id_col] for r in tradable[:n]]


def _self_test():
    import pandas as pd
    codes = ["000001", "300001", "688001", "600001"]
    df = pd.DataFrame({
        "股票代码": codes,
        "日期": pd.Timestamp("2026-03-20"),
        "涨跌幅": [5.0, 20.1, 19.95, -10.1],
    })
    t = get_tradable_ids(df, pd.Timestamp("2026-03-20"))
    assert "000001" in t  # normal, not limit
    assert "300001" not in t  # ChiNext 20%+ = limit up
    assert "688001" in t  # STAR 19.95 < 19.9? no, 19.95 >= 19.9 → excluded
    # Actually 19.95 >= 19.9 is True, so 688001 should be excluded
    assert "688001" not in t
    assert "600001" not in t  # main board -10.1 <= -9.9 → limit down
    print("OK")


if __name__ == "__main__":
    _self_test()
```

- [ ] **Step 2: Run self-test**

```bash
cd /root/shared-nvme/bigdata/THU-BDC2026
.venv/bin/python -m code.src.ensemble.tradability
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add code/src/ensemble/tradability.py
git commit -m "feat: tradability filter for price-limit stocks (Phase-A)"
```

---

## Task 2: Dynamic time-varying weights

**Files:**
- Modify: `code/src/ensemble/blender.py` (add two functions)

- [ ] **Step 1: Add `rolling_ic_weights` and `blend_scores_dynamic` to blender.py**

Append to `code/src/ensemble/blender.py` before the `_self_test` function:

```python
def rolling_ic_weights(
    model_scores: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    window: int = 20,
    beta: float = 1.0,
    date_col: str = "datetime",
    id_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """Compute per-date per-model weights from rolling Spearman IC.

    Returns DataFrame[date, model, weight] where weights sum to 1 per date.
    For dates with insufficient history, returns uniform weights.
    """
    from scipy.stats import spearmanr

    labels = labels.dropna(subset=["label"])
    all_dates = sorted(set().union(*(set(v[date_col].unique())
                                     for v in model_scores.values())))
    names = list(model_scores.keys())
    n = len(names)
    rows = []
    for i, d in enumerate(all_dates):
        # Look back `window` dates
        hist_dates = [dd for dd in all_dates[max(0, i - window):i]]
        if len(hist_dates) < 3:
            for nm in names:
                rows.append({"date": d, "model": nm, "weight": 1.0 / n})
            continue
        ics = {}
        for nm in names:
            df = model_scores[nm]
            sub = df[df[date_col].isin(hist_dates)]
            merged = sub.merge(labels, on=[id_col, date_col], how="inner")
            day_ics = []
            for dd, g in merged.groupby(date_col):
                if len(g) < 5:
                    continue
                ic, _ = spearmanr(g[score_col], g["label"])
                if np.isfinite(ic):
                    day_ics.append(ic)
            ics[nm] = np.mean(day_ics) if day_ics else 0.0
        # Softmax
        vals = np.array([ics[nm] for nm in names])
        exp_vals = np.exp(beta * (vals - vals.max()))
        w = exp_vals / exp_vals.sum()
        for j, nm in enumerate(names):
            rows.append({"date": d, "model": nm, "weight": float(w[j])})
    return pd.DataFrame(rows)


def blend_scores_dynamic(
    model_scores: Dict[str, pd.DataFrame],
    weight_df: pd.DataFrame,
    date_col: str = "datetime",
    id_col: str = "instrument",
    score_col: str = "score",
) -> pd.DataFrame:
    """Like blend_scores but with per-date weights from weight_df.

    weight_df has columns [date, model, weight].
    """
    # Build lookup: date -> {model: weight}
    wlookup = {}
    for _, r in weight_df.iterrows():
        d = r["date"]
        if d not in wlookup:
            wlookup[d] = {}
        wlookup[d][r["model"]] = r["weight"]

    names = list(model_scores.keys())
    n_models = len(names)

    # Rank-normalize each model
    ranked = {}
    for name, df in model_scores.items():
        sub = df[[id_col, date_col, score_col]].copy()
        sub["rank"] = rank_normalize_panel(sub, score_col, date_col=date_col)
        ranked[name] = sub.set_index([id_col, date_col])["rank"]

    # Build unified index
    all_idx = None
    for name in names:
        idx = ranked[name].reset_index()[[id_col, date_col]]
        all_idx = idx if all_idx is None else pd.concat([all_idx, idx]).drop_duplicates()
    all_idx = all_idx.reset_index(drop=True)

    final = np.zeros(len(all_idx))
    for name in names:
        r = ranked[name]
        vals = []
        for _, row in all_idx.iterrows():
            key = (row[id_col], row[date_col])
            v = r.get(key, 0.5)
            vals.append(v)
        arr = np.array(vals)
        # Per-row weight from lookup
        ws = np.array([
            wlookup.get(row[date_col], {}).get(name, 1.0 / n_models)
            for _, row in all_idx.iterrows()
        ])
        final += ws * arr

    out = all_idx.copy()
    out["final_score"] = final
    return out
```

- [ ] **Step 2: Run existing blender self-test (shouldn't break)**

```bash
.venv/bin/python -m code.src.ensemble.blender
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add code/src/ensemble/blender.py
git commit -m "feat: rolling IC dynamic weights + blend_scores_dynamic (Phase-A)"
```

---

## Task 3: Update portfolio.py for tradability

**Files:**
- Modify: `code/src/ensemble/portfolio.py`

- [ ] **Step 1: Add `tradable_ids` parameter to `build_portfolio`**

Change the signature and body of `build_portfolio`:

```python
def build_portfolio(blended_scores_today: pd.DataFrame, confidence: float,
                    alpha: float, top_k: int, min_position: float = 0.5,
                    tradable_ids: set = None) -> pd.DataFrame:
    total_position = min_position + alpha * (1 - min_position) * confidence
    if tradable_ids is not None:
        eligible = blended_scores_today[
            blended_scores_today["instrument"].isin(tradable_ids)
        ]
    else:
        eligible = blended_scores_today
    picks = eligible.nlargest(top_k, "final_score")
    actual_k = len(picks)
    if actual_k == 0:
        return pd.DataFrame(columns=["stock_id", "weight"])
    w = total_position / actual_k
    out = picks[["instrument"]].copy()
    out = out.rename(columns={"instrument": "stock_id"})
    out["weight"] = w
    return out.reset_index(drop=True)
```

- [ ] **Step 2: Run portfolio self-test**

```bash
.venv/bin/python -m code.src.ensemble.portfolio
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add code/src/ensemble/portfolio.py
git commit -m "feat: build_portfolio accepts tradable_ids filter (Phase-A)"
```

---

## Task 4: Integrate into pipeline.py predict

**Files:**
- Modify: `code/src/pipeline.py` (cmd_predict function, ~lines 361-441)

- [ ] **Step 1: Add tradability + dynamic weight support to cmd_predict**

At the top of `cmd_predict`, after loading `ens_cfg`:

```python
    blend_mode = ens_cfg.get("blend_mode", "static")
    rolling_window = int(ens_cfg.get("rolling_window", 20))
    rolling_beta = float(ens_cfg.get("rolling_beta", 1.0))
```

Import at top of file (add to existing imports):

```python
from .ensemble.tradability import get_tradable_ids
from .ensemble.blender import blend_scores_dynamic, rolling_ic_weights
```

Before building portfolio (around line 437), add tradability:

```python
    # Tradability filter
    stock_df_path = os.path.join(args.data_path, "stock_data.csv")
    tradable = None
    if os.path.exists(stock_df_path):
        raw_stock = pd.read_csv(stock_df_path)
        tradable = get_tradable_ids(raw_stock, target_date)
        print(f"[predict] tradable stocks: {len(tradable)} / {len(today_df)}")

    portfolio = build_portfolio(today_df, confidence, alpha=alpha, top_k=top_k,
                                min_position=config.MIN_POSITION,
                                tradable_ids=tradable)
```

For dynamic blending, replace the static `blend_scores` calls when `blend_mode == "dynamic"`:

```python
    if blend_mode == "dynamic":
        wdf = rolling_ic_weights(per_model_hist, labels,
                                 window=rolling_window, beta=rolling_beta)
        blended_today = blend_scores_dynamic(per_model_today, wdf)
        blended_hist = blend_scores_dynamic(per_model_hist, wdf)
    else:
        blended_today = blend_scores(per_model_today, weights)
        blended_hist = blend_scores(per_model_hist, weights)
```

- [ ] **Step 2: Verify import doesn't break**

```bash
.venv/bin/python -c "from code.src.pipeline import cmd_predict; print('import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add code/src/pipeline.py
git commit -m "feat: pipeline predict supports tradability + dynamic blend (Phase-A)"
```

---

## Task 5: Update rolling_backtest.py

**Files:**
- Modify: `test/rolling_backtest.py`

- [ ] **Step 1: Add CLI flags `--cost_bps`, `--dynamic_weights`, `--tradable_filter`**

Add to argparser:

```python
    ap.add_argument("--cost_bps", type=float, default=0.0,
                    help="One-way transaction cost in basis points (e.g. 3)")
    ap.add_argument("--dynamic_weights", action="store_true",
                    help="Use rolling-IC dynamic weights instead of static")
    ap.add_argument("--tradable_filter", action="store_true",
                    help="Exclude price-limit stocks from portfolio")
```

In the eval loop, apply cost:

```python
        cost = args.cost_bps * 1e-4
        port_ret = float((weights_arr * rets.reindex(tickers).to_numpy()).sum())
        if cost > 0:
            port_ret -= 2 * cost * sum(weights_arr)  # buy + sell
```

Apply tradability filter:

```python
        if args.tradable_filter:
            from code.src.ensemble.tradability import get_tradable_ids
            tradable = get_tradable_ids(stock_df, T)
        else:
            tradable = None
```

Then pass `tradable` into `build_portfolio(..., tradable_ids=tradable)`.

Apply dynamic weights: before the eval loop, compute `weight_df` if `--dynamic_weights`:

```python
    if args.dynamic_weights:
        from code.src.ensemble.blender import rolling_ic_weights, blend_scores_dynamic
        all_labels = lgb_panel[["instrument", "datetime", "label"]].copy()
        weight_df = rolling_ic_weights(per_model_scores, all_labels,
                                       window=20, beta=1.0)
        blended_all = blend_scores_dynamic(per_model_scores, weight_df)
    else:
        blended_all = blend_scores(per_model_scores, weights)
```

- [ ] **Step 2: Test with existing flags (backward compatible)**

```bash
.venv/bin/python test/rolling_backtest.py --eval_start 2026-02-01 --eval_end 2026-03-13 --feature_end 2026-03-20 --out /tmp/rb_compat.csv 2>&1 | tail -15
```

Expected: runs without error, same output as before.

- [ ] **Step 3: Commit**

```bash
git add test/rolling_backtest.py
git commit -m "feat: rolling backtest supports cost/tradability/dynamic flags (Phase-A)"
```

---

## Task 6: Run comparative backtests

**Files:** None created; uses existing scripts.

- [ ] **Step 1: Run baseline (existing)**

Already at `test/rolling_backtest_full.csv`.

- [ ] **Step 2: Run with tradability filter**

```bash
.venv/bin/python test/rolling_backtest.py \
    --eval_start 2025-11-01 --eval_end 2026-03-13 \
    --feature_end 2026-03-20 \
    --tradable_filter --cost_bps 3 \
    --out test/rolling_backtest_tradable.csv
```

- [ ] **Step 3: Run with dynamic weights + tradability + cost**

```bash
.venv/bin/python test/rolling_backtest.py \
    --eval_start 2025-11-01 --eval_end 2026-03-13 \
    --feature_end 2026-03-20 \
    --dynamic_weights --tradable_filter --cost_bps 3 \
    --out test/rolling_backtest_dynamic.csv
```

- [ ] **Step 4: Compare results**

```bash
.venv/bin/python -c "
import pandas as pd
from scipy import stats
tags = {'baseline': 'test/rolling_backtest_full.csv',
        'tradable+cost': 'test/rolling_backtest_tradable.csv',
        'dynamic+trad+cost': 'test/rolling_backtest_dynamic.csv'}
for tag, path in tags.items():
    try:
        df = pd.read_csv(path)
        p = df['portfolio_return']
        a = df['baseline_avg'].dropna()
        ex = df['excess_vs_avg'].dropna()
        t, pv = stats.ttest_1samp(ex, 0) if len(ex) > 5 else (0, 1)
        print(f'{tag:25s}  mean={p.mean():.4%}  std={p.std():.4%}  '
              f'win={( p>0).mean():.1%}  excess_p={pv:.4f}  N={len(p)}')
    except Exception as e:
        print(f'{tag}: {e}')
"
```

- [ ] **Step 5: Commit backtest results**

```bash
git add test/rolling_backtest_tradable.csv test/rolling_backtest_dynamic.csv
git commit -m "data: Phase-A comparative backtest results"
```

---

## Task 7: Update report

**Files:**
- Modify: `docs/report.md`

- [ ] **Step 1: Add Phase-A section to report.md**

Append a new section "## 11. Phase-A 优化" before the conclusion, containing:
- 4-column comparison table (baseline / tradable+cost / dynamic+trad+cost)
- t-test results for each variant
- Key observations (variance change, win rate change, mean change)

- [ ] **Step 2: Commit**

```bash
git add docs/report.md
git commit -m "docs: Phase-A comparative results in report"
```
