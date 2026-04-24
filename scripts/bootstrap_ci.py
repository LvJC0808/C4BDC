"""Bootstrap CI for rolling backtest daily returns.

Usage:
    python scripts/bootstrap_ci.py test/rolling_lgb_alloc_equal.csv \
        --n 10000 --seed 42

Outputs bootstrap 95% CI for mean 5d return, vs HS300 excess mean, and win rate.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rolling_csv")
    ap.add_argument("--n", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05, help="CI level (0.05 → 95%)")
    args = ap.parse_args()

    d = pd.read_csv(args.rolling_csv)
    ret = d["portfolio_return"].dropna().values
    base = d["baseline_avg"].dropna().values
    excess = ret - base
    N = len(ret)
    print(f"loaded {args.rolling_csv}: N={N}")

    rng = np.random.default_rng(args.seed)
    idx_mat = rng.integers(0, N, size=(args.n, N))

    def boot_stat(arr, fn):
        samples = fn(arr[idx_mat])
        lo = np.quantile(samples, args.alpha / 2)
        hi = np.quantile(samples, 1 - args.alpha / 2)
        return arr.mean() if fn is np.mean else None, lo, hi, samples

    # mean 5d return
    samples = ret[idx_mat].mean(axis=1)
    lo_r, hi_r = np.quantile(samples, [args.alpha / 2, 1 - args.alpha / 2])

    # excess vs baseline_avg
    samples_e = excess[idx_mat].mean(axis=1)
    lo_e, hi_e = np.quantile(samples_e, [args.alpha / 2, 1 - args.alpha / 2])

    # win rate
    wins = (ret > 0).astype(float)
    samples_w = wins[idx_mat].mean(axis=1)
    lo_w, hi_w = np.quantile(samples_w, [args.alpha / 2, 1 - args.alpha / 2])

    ci = int(100 * (1 - args.alpha))
    print(f"\n--- Bootstrap {ci}% CI (n={args.n}, N={N}) ---")
    print(f"{'':25s}  {'point':>8s}  {'CI lower':>10s}  {'CI upper':>10s}")
    print(f"{'Mean 5d return':25s}  {ret.mean()*100:+7.3f}%  {lo_r*100:+9.3f}%  {hi_r*100:+9.3f}%")
    print(f"{'Excess vs HS300 avg':25s}  {excess.mean()*100:+7.3f}%  {lo_e*100:+9.3f}%  {hi_e*100:+9.3f}%")
    print(f"{'Win rate':25s}  {wins.mean()*100:6.2f}%  {lo_w*100:8.2f}%  {hi_w*100:8.2f}%")

    # p(mean > 0), p(excess > 0)
    p_pos = (samples > 0).mean()
    p_excess_pos = (samples_e > 0).mean()
    print(f"\nP(mean > 0)       = {p_pos*100:6.2f}%")
    print(f"P(excess > 0)     = {p_excess_pos*100:6.2f}%")


if __name__ == "__main__":
    main()
