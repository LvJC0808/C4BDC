import sys, json
sys.path.insert(0, "/root/shared-nvme/bigdata/THU-BDC2026")
import numpy as np
import pandas as pd
from code.src.ensemble.blender import (
    rank_normalize_daily, optimize_weights_icir,
    select_lambda_loo, bootstrap_weights, compute_blend_ic_series,
)

labels = pd.read_parquet("/tmp/labels.parquet")
labels["datetime"] = pd.to_datetime(labels["datetime"])
LAM_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
MO = ["lgb","master","mixer"]

def load(src, m):
    dfs=[]
    for s in [42,2024]:
        for f in [0,1,2]:
            d = pd.read_parquet(f"{src}/model/{m}/seed_{s}_fold_{f}/val_scores.parquet")
            dfs.append(d[["instrument","datetime","score","fold"]])
    d = pd.concat(dfs, ignore_index=True)
    d["datetime"] = pd.to_datetime(d["datetime"])
    return d.groupby(["instrument","datetime","fold"])["score"].mean().reset_index()

def build(src):
    bm = {m: load(src,m) for m in MO}
    folds = sorted(bm["lgb"]["fold"].unique())
    sbf=[]
    for f in folds:
        fd={}
        for m in MO:
            fd[m]=rank_normalize_daily(bm[m][bm[m]["fold"]==f][["instrument","datetime","score"]])
        sbf.append(fd)
    return sbf, {m: rank_normalize_daily(bm[m][["instrument","datetime","score"]]) for m in MO}

for src, tag in [("/tmp/mate_linux","linux"), ("/tmp/mate_win","windows")]:
    print(f"\n==== {tag.upper()} ====", flush=True)
    sbf, allr = build(src)
    print(f"per-fold rows: {[len(x['lgb']) for x in sbf]}", flush=True)

    # --- SYNCHRONIZE labels with scores date range ---
    date_min = min(allr[m]["datetime"].min() for m in MO)
    date_max = max(allr[m]["datetime"].max() for m in MO)
    lbl = labels[(labels["datetime"]>=date_min)&(labels["datetime"]<=date_max)].copy()
    print(f"labels in range: {len(lbl)} dates=[{lbl['datetime'].min()}..{lbl['datetime'].max()}]", flush=True)

    # LOO lambda
    lam = select_lambda_loo(sbf, lbl, LAM_GRID, MO, seed=42)
    print(f"lambda* = {lam['lambda']}", flush=True)
    print(f"all lambda ICIR: {lam['all']}", flush=True)

    # optimize on all data
    opt = optimize_weights_icir(allr, lbl, lam=lam["lambda"], model_order=MO, seed=42, n_dirichlet=5)
    print(f"ICIR weights: {opt['weights']}", flush=True)
    print(f"ICIR score:   {opt['icir']:.4f}", flush=True)

    # baselines
    eq_ic = compute_blend_ic_series(allr, lbl, {"lgb":1/3,"master":1/3,"mixer":1/3})
    lo_ic = compute_blend_ic_series(allr, lbl, {"lgb":1.0,"master":0.0,"mixer":0.0})
    def _icir(a): return float(a.mean()/(a.std()+1e-6)*np.sqrt(len(a))) if len(a)>1 else 0.0
    print(f"EQ   weight: mean_IC={eq_ic.mean():.4f} std={eq_ic.std():.4f} ICIR={_icir(eq_ic):.4f}", flush=True)
    print(f"LGB-only:   mean_IC={lo_ic.mean():.4f} std={lo_ic.std():.4f} ICIR={_icir(lo_ic):.4f}", flush=True)

    # bootstrap
    print("running bootstrap (n=100)...", flush=True)
    boot = bootstrap_weights(allr, lbl, lam=lam["lambda"], model_order=MO, n=100, seed=42)
    print(f"bootstrap n_success={boot['n_success']}", flush=True)
    for m in MO:
        print(f"  {m}: median={boot['median'][m]:.3f} CI=[{boot['ci_low'][m]:.3f}, {boot['ci_high'][m]:.3f}]", flush=True)

    out = {"source": tag, "lambda": lam["lambda"], "lambda_all": lam["all"],
           "icir_weights": opt["weights"], "icir_score": opt["icir"],
           "baseline": {"equal_weight_icir": _icir(eq_ic), "lgb_only_icir": _icir(lo_ic)},
           "bootstrap": {"median": boot["median"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]}}
    with open(f"/root/shared-nvme/bigdata/THU-BDC2026/temp/icir_result_{tag}.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote temp/icir_result_{tag}.json", flush=True)
