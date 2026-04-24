"""Score both LGB-only and baseline StockTransformer on score_self.py's fixed window.

score_self uses data/test.csv which is 2026-03-04..2026-03-10 (5 trading days).
So T = 2026-03-03. Produce each model's Top-5 at T=2026-03-03 and run score_self.
"""
from __future__ import annotations
import json, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code" / "src"))

TARGET_DATE = pd.Timestamp(os.environ.get("TARGET_DATE", "2026-03-03"))
OUTPUT = REPO / "output" / "result.csv"


def lgb_top5():
    from code.src import config
    from code.src.ensemble.portfolio import deterministic_top_k
    from code.src.features.build import build_feature_sets
    from code.src.pipeline import _load_model, _predict_scores

    MODEL_DIR = "./model_lgb_only"
    with open(os.path.join(MODEL_DIR, "ensemble_config.json")) as f:
        ens_cfg = json.load(f)
    seeds = [int(s) for s in ens_cfg["seeds"]]

    fsets = build_feature_sets("./data", "./temp", use_cache=True)
    fset = fsets["lgb"]
    panel = fset["panel"].copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    fcols = fset["feature_cols"]

    today = panel[panel["datetime"] == TARGET_DATE].copy()
    if today.empty:
        raise RuntimeError(f"no LGB rows for {TARGET_DATE.date()}")
    print(f"[lgb] target={TARGET_DATE.date()} n={len(today)}")

    score_frames = []
    for seed in seeds:
        mdir = os.path.join(MODEL_DIR, "lgb", f"seed_{seed}_refit")
        mdl = _load_model("lgb", mdir)
        scored = _predict_scores("lgb", mdl, today, fcols, market_df=None)
        scored = scored.rename(columns={"score": f"s_{seed}"})
        score_frames.append(scored[["instrument", f"s_{seed}"]])
    merged = score_frames[0]
    for f in score_frames[1:]:
        merged = merged.merge(f, on="instrument")
    s_cols = [c for c in merged.columns if c.startswith("s_")]
    merged["score"] = merged[s_cols].mean(axis=1)
    merged = merged.rename(columns={"instrument": "stock_id"})
    top5 = deterministic_top_k(merged[["stock_id", "score"]], k=5)
    top5["weight"] = 0.2
    return top5[["stock_id", "weight"]]


def baseline_top5():
    import joblib, torch, multiprocessing as mp
    from tqdm import tqdm
    from model import StockTransformer  # type: ignore
    from utils import engineer_features_158plus39  # type: ignore
    from scripts.rolling_backtest_baseline import FEATURE_COLS_158_39  # type: ignore

    with open("./model/60_158+39/config.json") as f:
        cfg = json.load(f)
    raw = pd.read_csv("./data/train.csv", dtype={"股票代码": str})
    raw["股票代码"] = raw["股票代码"].astype(str).str.zfill(6)
    raw["日期"] = pd.to_datetime(raw["日期"])
    # Truncate to T = target_date so no leakage
    raw = raw[raw["日期"] <= TARGET_DATE]

    stock_ids = sorted(raw["股票代码"].unique())
    stockid2idx = {sid: i for i, sid in enumerate(stock_ids)}

    groups = [g for _, g in raw.groupby("股票代码", sort=False)]
    with mp.Pool(processes=6) as pool:
        processed = pd.concat(list(tqdm(pool.imap(engineer_features_158plus39, groups),
                                         total=len(groups), desc="feat"))).reset_index(drop=True)
    processed["instrument"] = processed["股票代码"].map(stockid2idx).astype(np.int64)
    processed["日期"] = pd.to_datetime(processed["日期"])
    feats = FEATURE_COLS_158_39
    processed[feats] = processed[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load("./model/60_158+39/scaler.pkl")
    processed[feats] = scaler.transform(processed[feats])

    seq_len = cfg["sequence_length"]
    seqs, sids = [], []
    for sid in stock_ids:
        g = processed[processed["股票代码"] == sid].sort_values("日期").tail(seq_len)
        if len(g) == seq_len:
            seqs.append(g[feats].values.astype(np.float32))
            sids.append(sid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockTransformer(input_dim=len(feats), config=cfg, num_stocks=len(stock_ids)).to(device)
    model.load_state_dict(torch.load("./model/60_158+39/best_model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(seqs)).unsqueeze(0).to(device)
        scores = model(x).squeeze(0).detach().cpu().numpy()
    order = np.argsort(scores)[::-1]
    top5 = pd.DataFrame({"stock_id": [sids[i] for i in order[:5]], "weight": 0.2})
    return top5


def run_score_self(label):
    res = subprocess.run([".venv/bin/python", "test/score_self.py"],
                         capture_output=True, text=True, cwd=str(REPO))
    print(f"--- {label} ---")
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr)
    tmp = pd.read_csv(REPO / "temp" / "tmp.csv")
    return float(tmp["Final Score"].iloc[0])


def main():
    # LGB
    lgb5 = lgb_top5()
    print("[lgb] top5:", lgb5.to_dict("records"))
    lgb5.to_csv(OUTPUT, index=False)
    lgb_score = run_score_self("LGB-only @ 2026-03-03")

    # Baseline
    bl5 = baseline_top5()
    print("[baseline] top5:", bl5.to_dict("records"))
    bl5.to_csv(OUTPUT, index=False)
    bl_score = run_score_self("Baseline StockTransformer @ 2026-03-03")

    print()
    print("=" * 60)
    print(f"  LGB-only                   : {lgb_score*100:+.4f}%")
    print(f"  Baseline StockTransformer  : {bl_score*100:+.4f}%")
    print(f"  Delta (LGB - baseline)     : {(lgb_score-bl_score)*100:+.4f}%")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
