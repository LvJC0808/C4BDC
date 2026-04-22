"""Rolling backtest for the ensemble pipeline.

For each trading date T in an evaluation window:
  1. Load refit models from `model_dir`.
  2. Score all HS300 stocks for date T using the ensemble pipeline.
  3. Build the portfolio (top-K × confidence-adaptive weights).
  4. Compute realized 5-day open-to-open return:
       ret_i = (open_{T+5} - open_{T+1}) / open_{T+1}
       daily_score = Σ_i weight_i * ret_i
  5. Also compute two baselines:
       - hs300_equal: equal-weight top-K by past 5-day return (naive).
       - hs300_avg:   unweighted average 5-day return of all stocks in test pool.
Output: CSV with columns [date, n_picks, portfolio_return, baseline_topk, baseline_avg,
                          excess_vs_avg, tickers, weights]

Usage:
    python test/rolling_backtest.py \
        --data_path ./data --model_dir ./model --temp_dir ./temp \
        --eval_start 2026-01-05 --eval_end 2026-03-13 \
        --out test/rolling_backtest.csv

Note: models in `model_dir` were trained on data up to some training cutoff.
You must pick --eval_start strictly AFTER that cutoff + embargo(5) to avoid
leakage. For smoke-trained models, a safe eval window is everything up to
min(model's training end, T where T+5 <= data max date).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make the `code.src` package importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.src import config  # noqa: E402
from code.src.features.build import build_feature_sets  # noqa: E402
from code.src.pipeline import (  # noqa: E402
    _load_model, _predict_scores, set_global_seed,
)
from code.src.ensemble.blender import blend_scores  # noqa: E402
from code.src.ensemble.portfolio import (  # noqa: E402
    compute_confidence, build_portfolio,
)
import json


MODEL_NAMES = ("lgb", "master", "mixer")


def _load_ensemble_cfg(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "ensemble_config.json")) as f:
        return json.load(f)


def _trading_dates(stock_df: pd.DataFrame) -> List[pd.Timestamp]:
    return sorted(pd.to_datetime(stock_df["日期"]).unique())


def _realized_return(stock_df: pd.DataFrame,
                     target_date: pd.Timestamp,
                     tickers: List[str]) -> pd.Series:
    """Open_{T+5} / Open_{T+1} - 1 per ticker."""
    future = stock_df[stock_df["日期"] > target_date].copy()
    future = future.sort_values(["股票代码", "日期"])
    out = {}
    for tic in tickers:
        g = future[future["股票代码"] == tic].head(5)
        if len(g) < 5:
            out[tic] = np.nan
            continue
        p1 = float(g.iloc[0]["开盘"])
        p5 = float(g.iloc[-1]["开盘"])
        out[tic] = (p5 - p1) / (p1 + 1e-12)
    return pd.Series(out, name="ret")


def _rolling_ic(recent_blended: pd.DataFrame,
                recent_labels: pd.DataFrame,
                window: int = 20) -> float:
    """Mean per-date Spearman rank-IC of final_score vs label over last `window` dates."""
    m = recent_blended.merge(recent_labels, on=["instrument", "datetime"], how="inner")
    m = m.dropna(subset=["label"])
    if m.empty:
        return 0.0
    dates = sorted(m["datetime"].unique())[-window:]
    ics = []
    from scipy.stats import spearmanr
    for d in dates:
        g = m[m["datetime"] == d]
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["final_score"], g["label"])
        if np.isfinite(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=config.DATA_PATH)
    ap.add_argument("--model_dir", default=config.MODEL_DIR)
    ap.add_argument("--temp_dir", default=config.TEMP_DIR)
    ap.add_argument("--eval_start", required=True)
    ap.add_argument("--eval_end", required=True)
    ap.add_argument("--feature_start", default="2024-01-01")
    ap.add_argument("--feature_end", default=None,
                    help="If None, use eval_end + 10 trading days")
    ap.add_argument("--out", default="test/rolling_backtest.csv")
    ap.add_argument("--min_future_days", type=int, default=5)
    args = ap.parse_args()

    set_global_seed(config.SEED)
    ens = _load_ensemble_cfg(args.model_dir)
    weights = ens["weights"]
    top_k = int(ens["top_k"])
    alpha = float(ens["alpha"])
    min_pos = config.MIN_POSITION

    # --- Load raw stock_data for realized returns ---
    stock_df = pd.read_csv(os.path.join(args.data_path, "stock_data.csv"))
    stock_df["股票代码"] = stock_df["股票代码"].astype(str).str.zfill(6)
    stock_df["日期"] = pd.to_datetime(stock_df["日期"])

    # --- Build features covering eval window + 5-day lookahead ---
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    feat_end = (pd.Timestamp(args.feature_end)
                if args.feature_end else eval_end + pd.Timedelta(days=14))
    print(f"[rb] feature window: {args.feature_start} .. {feat_end.date()}")
    feature_sets = build_feature_sets(
        data_path=args.data_path, temp_dir=args.temp_dir,
        start_date=args.feature_start, end_date=str(feat_end.date()),
        use_cache=True,
    )

    # --- Load refit models (average across seeds) ---
    seeds = ens.get("seeds", config.SEEDS)
    loaded = {name: [] for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        for seed in seeds:
            d = os.path.join(args.model_dir, name, f"seed_{seed}_refit")
            if os.path.isdir(d):
                loaded[name].append(_load_model(name, d))
    for n in MODEL_NAMES:
        if not loaded[n]:
            raise RuntimeError(f"No refit models found for {n} under {args.model_dir}")
    print(f"[rb] loaded refit models: "
          f"{ {n: len(loaded[n]) for n in MODEL_NAMES} }")

    # --- Score each panel over the feature window once, aggregate by model ---
    per_model_scores = {}
    for name in MODEL_NAMES:
        panel = feature_sets[name]["panel"].copy()
        panel["datetime"] = pd.to_datetime(panel["datetime"])
        fcols = feature_sets[name]["feature_cols"]
        mkt = feature_sets[name].get("market")
        # Restrict panel dates to eval_start - 60 days .. feat_end (for rolling IC history)
        lb_start = eval_start - pd.Timedelta(days=60)
        sl = panel[(panel["datetime"] >= lb_start) & (panel["datetime"] <= feat_end)]
        preds = []
        for mdl in loaded[name]:
            p = _predict_scores(name, mdl, sl, fcols, mkt)
            preds.append(p)
        # Mean across seeds
        cat = pd.concat(preds, ignore_index=True)
        agg = cat.groupby(["instrument", "datetime"])["score"].mean().reset_index()
        per_model_scores[name] = agg
        print(f"[rb] {name}: {len(agg)} scored rows, dates "
              f"{agg['datetime'].min().date()}..{agg['datetime'].max().date()}")

    # Blend once over the full feature window (per-date rank-normalize inside)
    blended_all = blend_scores(per_model_scores, weights)
    blended_all["datetime"] = pd.to_datetime(blended_all["datetime"])

    # Labels for rolling IC
    lgb_panel = feature_sets["lgb"]["panel"].copy()
    lgb_panel["datetime"] = pd.to_datetime(lgb_panel["datetime"])
    all_labels = lgb_panel[["instrument", "datetime", "label"]].copy()

    # --- Loop over eval dates ---
    all_dates = _trading_dates(stock_df)
    eval_dates = [d for d in all_dates if eval_start <= d <= eval_end]
    # Need T+5 to exist in stock_df
    max_target = all_dates[-args.min_future_days - 1] if len(all_dates) > args.min_future_days else all_dates[-1]
    eval_dates = [d for d in eval_dates if d <= max_target]
    print(f"[rb] {len(eval_dates)} eval dates: "
          f"{eval_dates[0].date() if eval_dates else '-'}..{eval_dates[-1].date() if eval_dates else '-'}")

    rows = []
    for T in eval_dates:
        today_blended = blended_all[blended_all["datetime"] == T]
        if today_blended.empty:
            continue
        # Rolling IC on past 20 dates of labeled blended scores (strictly < T)
        hist = blended_all[blended_all["datetime"] < T]
        ic = _rolling_ic(hist, all_labels, window=20)

        # Model-level top-K sets (re-rank per model on date T)
        topk_sets = []
        for name in MODEL_NAMES:
            s = per_model_scores[name]
            st = s[s["datetime"] == T]
            if st.empty:
                continue
            topk_sets.append(set(st.nlargest(top_k, "score")["instrument"].tolist()))

        today_scores = today_blended.set_index("instrument")["final_score"]
        conf = compute_confidence(today_scores, topk_sets, ic, top_k=top_k)

        port = build_portfolio(
            today_blended[["instrument", "final_score"]].copy(),
            conf, alpha, top_k, min_position=min_pos,
        )
        tickers = port["stock_id"].astype(str).str.zfill(6).tolist()
        weights_arr = port["weight"].to_numpy()

        rets = _realized_return(stock_df, T, tickers)
        port_ret = float((weights_arr * rets.reindex(tickers).to_numpy()).sum())

        # Baselines on same universe (HS300 stocks in stock_data that have T+5 data)
        universe = stock_df[stock_df["日期"] == T]["股票代码"].unique().tolist()
        uni_rets = _realized_return(stock_df, T, universe).dropna()
        baseline_avg = float(uni_rets.mean()) if len(uni_rets) else np.nan
        # Momentum top-K: past 5-day return of each stock
        past = stock_df[(stock_df["日期"] > T - pd.Timedelta(days=14)) &
                        (stock_df["日期"] <= T)].copy()
        mom = (past.sort_values(["股票代码", "日期"])
                  .groupby("股票代码")["开盘"]
                  .apply(lambda s: (s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 else np.nan))
        mom_top = mom.dropna().nlargest(top_k).index.tolist()
        mom_rets = _realized_return(stock_df, T, mom_top)
        baseline_mom = float(mom_rets.mean()) if not mom_rets.isna().all() else np.nan

        rows.append({
            "date": T.date(),
            "n_picks": len(tickers),
            "confidence": round(conf, 4),
            "rolling_ic": round(ic, 4),
            "portfolio_return": round(port_ret, 6),
            "baseline_mom_topk": round(baseline_mom, 6) if not np.isnan(baseline_mom) else None,
            "baseline_avg": round(baseline_avg, 6) if not np.isnan(baseline_avg) else None,
            "excess_vs_avg": round(port_ret - baseline_avg, 6) if not np.isnan(baseline_avg) else None,
            "excess_vs_mom": round(port_ret - baseline_mom, 6) if not np.isnan(baseline_mom) else None,
            "tickers": ",".join(tickers),
            "weights": ",".join(f"{w:.4f}" for w in weights_arr),
        })

    out = pd.DataFrame(rows)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[rb] wrote {out_path}  ({len(out)} rows)")

    # --- Summary stats ---
    if not out.empty:
        from scipy import stats
        port = out["portfolio_return"].dropna()
        avg = out["baseline_avg"].dropna()
        mom = out["baseline_mom_topk"].dropna()
        print()
        print("=" * 60)
        print(f"{'Metric':<28}{'Our':>12}{'BaselineAvg':>14}{'Mom-TopK':>14}")
        print("-" * 60)
        print(f"{'Mean daily return':<28}{port.mean():>12.4%}"
              f"{avg.mean():>14.4%}{mom.mean():>14.4%}")
        print(f"{'Median daily return':<28}{port.median():>12.4%}"
              f"{avg.median():>14.4%}{mom.median():>14.4%}")
        print(f"{'Std':<28}{port.std():>12.4%}"
              f"{avg.std():>14.4%}{mom.std():>14.4%}")
        print(f"{'Win rate (>0)':<28}{(port>0).mean():>12.2%}"
              f"{(avg>0).mean():>14.2%}{(mom>0).mean():>14.2%}")
        print(f"{'Win vs baseline_avg':<28}{(out['excess_vs_avg']>0).mean():>12.2%}")
        print(f"{'Win vs mom_topk':<28}{(out['excess_vs_mom']>0).mean():>12.2%}")
        # paired t-test: Our vs baseline_avg
        ex = out["excess_vs_avg"].dropna()
        if len(ex) > 5:
            t, p = stats.ttest_1samp(ex, 0.0)
            print(f"{'t-test excess vs avg':<28}  t={t:+.3f}  p={p:.4f}  N={len(ex)}")
        ex2 = out["excess_vs_mom"].dropna()
        if len(ex2) > 5:
            t, p = stats.ttest_1samp(ex2, 0.0)
            print(f"{'t-test excess vs mom':<28}  t={t:+.3f}  p={p:.4f}  N={len(ex2)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
