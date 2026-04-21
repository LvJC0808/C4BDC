# BDC2026 Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-model ensemble (LightGBM+DE, MASTER, StockMixer) with advanced feature engineering, walk-forward CV, and confidence-adaptive position sizing to beat the BDC2026 baseline.

**Architecture:** Modular pipeline with `features/` (Alpha158+360+neutralize), `models/` (lgb, master, mixer), `cv/` (walk-forward+embargo), `ensemble/` (rank blend + portfolio). Single `pipeline.py` entry point for train/predict.

**Tech Stack:** Python 3.10, PyTorch, LightGBM, TA-Lib, NumPy, Pandas, SciPy, scikit-learn, pyarrow, baostock

**Spec:** `docs/superpowers/specs/2026-04-21-bdc2026-ensemble-design.md`

---

## Phase 1: Project Scaffolding & Data Preparation

### Task 1: Directory structure + config

**Files:**
- Create: `code/src/config.py` (overwrite baseline)
- Create: `code/src/features/__init__.py`
- Create: `code/src/models/__init__.py`
- Create: `code/src/cv/__init__.py`
- Create: `code/src/ensemble/__init__.py`
- Create: `code/src/data_prep/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /root/shared-nvme/bigdata/THU-BDC2026
mkdir -p code/src/{features,models,cv,ensemble,data_prep}
touch code/src/{features,models,cv,ensemble,data_prep}/__init__.py
```

- [ ] **Step 2: Write unified config.py**

```python
# code/src/config.py
import os

SEED = 42
SEEDS = [42, 2024, 7]

DATA_PATH = os.environ.get("DATA_PATH", "./data")
MODEL_DIR = os.environ.get("MODEL_DIR", "./model")
TEMP_DIR = os.environ.get("TEMP_DIR", "./temp")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")

# Feature sets
SEQUENCE_LENGTH = 60
ALPHA158_DIM = 158
ALPHA360_FIELDS = ["open", "high", "low", "close", "volume", "amount"]
ALPHA360_LOOKBACK = 60

# Neutralization
NEUTRALIZE = True
NEUTRALIZE_FACTORS = ["industry", "log_mktcap", "beta60"]

# CV
CV_FOLDS = 3
CV_EMBARGO = 5
CV_VAL_DAYS = 20
CV_FOLD_GAP = 50   # gap between fold end points
HOLDOUT_DAYS = 20

# LightGBM
LGB_PARAMS = {
    "objective": "regression_l1",
    "num_leaves": 64,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "seed": SEED,
}
LGB_ROUNDS = 2000
LGB_EARLY_STOP = 50
LGB_DE_ROUNDS = 3  # DoubleEnsemble iterations

# MASTER
MASTER_CONFIG = {
    "lookback": 8,
    "d_model": 256,
    "n_head": 4,
    "dropout": 0.5,
    "beta": 5,
    "market_feat_dim": 63,
    "feat_dim": 158,
    "epochs": 40,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "patience": 10,
    "loss_mse_w": 0.5,
    "loss_ic_w": 0.5,
    "loss_topk_w": 0.1,
}

# StockMixer
MIXER_CONFIG = {
    "lookback": 16,
    "hidden": 128,
    "scale": 3,
    "market_scale": 3,
    "feat_dim": 6,
    "epochs": 40,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "patience": 10,
}

# Ensemble
ENSEMBLE_INIT_WEIGHTS = {"lgb": 0.4, "master": 0.4, "mixer": 0.2}
TOP_K_CANDIDATES = [3, 4, 5]
POSITION_ALPHAS = [0.3, 0.5, 0.7]
MIN_POSITION = 0.5

# Sample pool filters
MIN_LIST_DAYS = 250
MAX_SUSPEND_DAYS = 5
```

- [ ] **Step 3: Commit**

```bash
git add code/src/
git commit -m "feat: project scaffolding + unified config"
```

### Task 2: Data fetching (fetch_all.py)

**Files:**
- Create: `code/src/data_prep/fetch_all.py`

- [ ] **Step 1: Write fetch_all.py**

Uses baostock to download: stock_data (with valuation), industry_map, stock_basic,
hs300_history, csi300_index, trade_calendar. Saves to `data/` as CSV.

Key functions:
- `fetch_stock_data(start, end, out_dir)` — hs300 daily OHLCV + peTTM/pbMRQ/psTTM/pcfNcfTTM
- `fetch_industry_map(out_dir)` — sw_industry for all hs300 stocks
- `fetch_stock_basic(out_dir)` — ipoDate
- `fetch_hs300_history(start, end, out_dir)` — semi-annual constituent snapshots
- `fetch_csi300_index(start, end, out_dir)` — sh.000300 daily line
- `fetch_trade_calendar(start, end, out_dir)` — trading days

CLI: `python fetch_all.py --start 2018-01-01 --end 2026-04-18 --out ./data`

- [ ] **Step 2: Run fetch and verify outputs**

```bash
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
cd code/src && python data_prep/fetch_all.py --start 2018-01-01 --end 2026-04-18 --out ../../data
ls -lh ../../data/
```

Expected: 6 CSV files, stock_data.csv ~200 MB

- [ ] **Step 3: Commit**

```bash
git add code/src/data_prep/fetch_all.py
git commit -m "feat: baostock data fetcher (Z1 extended)"
```

## Phase 2: Feature Engineering

### Task 3: Alpha158

**Files:**
- Create: `code/src/features/alpha158.py`

Reuse baseline `utils.py` logic for 158 alpha factors (TA-Lib based).
Input: DataFrame with OHLCV columns per stock.
Output: DataFrame with 158 alpha columns + date + stock_id.

- [ ] **Step 1: Extract and refactor alpha158 from baseline utils.py**
- [ ] **Step 2: Test on sample data**
- [ ] **Step 3: Commit**

### Task 4: Alpha360

**Files:**
- Create: `code/src/features/alpha360.py`

Qlib-style: for each stock, take past 60 days × 6 fields (OHLCV + amount),
divide each by close_t to normalize. Flatten to 360-dim vector per (date, stock).

- [ ] **Step 1: Implement alpha360**
- [ ] **Step 2: Test**
- [ ] **Step 3: Commit**

### Task 5: Valuation + Market + Beta features

**Files:**
- Create: `code/src/features/valuation.py`

PE/PB/PS/PCF raw + 20d/60d z-score + industry rank.
Market 63-dim features from CSI300 index (for MASTER).
Beta-60d: rolling OLS of stock return vs index return.

- [ ] **Step 1: Implement**
- [ ] **Step 2: Test**
- [ ] **Step 3: Commit**

### Task 6: Neutralization + rank-gauss

**Files:**
- Create: `code/src/features/neutralize.py`

Per cross-section (each trading day):
1. Industry demeaning
2. Linear regression on log_mktcap + beta → residual
3. 3σ winsorize
4. Rank → Φ⁻¹ (rank-gauss)

Also neutralize labels (future return rank-gauss).

- [ ] **Step 1: Implement**
- [ ] **Step 2: Test**
- [ ] **Step 3: Commit**

### Task 7: Unified feature builder

**Files:**
- Create: `code/src/features/build.py`

Orchestrates: load CSVs → sample pool filter → alpha158/360/valuation → neutralize → cache.
Returns dict of feature sets keyed by model name.

- [ ] **Step 1: Implement**
- [ ] **Step 2: End-to-end test: raw CSV → cached features**
- [ ] **Step 3: Commit**

## Phase 3: Cross-Validation

### Task 8: Walk-forward CV

**Files:**
- Create: `code/src/cv/walk_forward.py`

Implements:
- `generate_cv_splits(dates, n_folds=3, embargo=5, val_days=20, holdout_days=20)`
  Returns list of (train_dates, val_dates) + holdout_dates
- `get_refit_epochs(cv_results)` → median epochs × 1.05

- [ ] **Step 1: Implement**
- [ ] **Step 2: Test with synthetic date range**
- [ ] **Step 3: Commit**

## Phase 4: LightGBM + DoubleEnsemble

### Task 9: LightGBM + DE model

**Files:**
- Create: `code/src/models/lgb_de.py`

Classes:
- `LGBModel`: train(X, y, X_val, y_val) → model; predict(X) → scores
- `DoubleEnsembleModel`: wraps 3 rounds of sample/feature reweighting
  train() → list of sub-models; predict() → averaged scores

- [ ] **Step 1: Implement LGBModel**
- [ ] **Step 2: Implement DoubleEnsembleModel**
- [ ] **Step 3: Test on sample data**
- [ ] **Step 4: Commit**

## Phase 5: MASTER

### Task 10: MASTER model

**Files:**
- Create: `code/src/models/master.py`

Port from SJTU-DMTai/MASTER official code. Key modules:
- `FeatureEncoder` (MLP per stock)
- `MarketGuidedGating` (FiLM conditioning)
- `IntraStockAttention` (temporal self-attention)
- `InterStockAttention` (cross-stock attention)
- `MASTER` (full model: encode → gate → intra → inter → predict)
- `MasterTrainer`: train loop with MSE+RankIC+TopK loss, early stopping

- [ ] **Step 1: Clone MASTER repo, study code**

```bash
cd /tmp && git clone https://github.com/SJTU-DMTai/MASTER.git
```

- [ ] **Step 2: Port model architecture**
- [ ] **Step 3: Port/write training loop with custom loss**
- [ ] **Step 4: Test forward pass shape**
- [ ] **Step 5: Commit**

## Phase 6: StockMixer

### Task 11: StockMixer model

**Files:**
- Create: `code/src/models/stockmixer.py`

Port from SJTU-DMTai/StockMixer. Key modules:
- `TimeMixer`, `FeatureMixer`, `StockMixer` (three parallel MLP-Mixer paths)
- `StockMixerModel` (full model)
- `MixerTrainer`: train loop, same loss as MASTER

- [ ] **Step 1: Clone StockMixer repo, study code**

```bash
cd /tmp && git clone https://github.com/SJTU-DMTai/StockMixer.git
```

- [ ] **Step 2: Port model**
- [ ] **Step 3: Port/write training loop**
- [ ] **Step 4: Test forward pass shape**
- [ ] **Step 5: Commit**

## Phase 7: Ensemble + Pipeline + Docker

### Task 12: Ensemble blender

**Files:**
- Create: `code/src/ensemble/blender.py`

- `rank_normalize(scores)` → cross-section rank in [0,1]
- `blend_scores(model_scores_dict, weights)` → final_score per stock
- `grid_search_weights(model_scores, labels, top_k_list)` → best weights + K

- [ ] **Step 1: Implement**
- [ ] **Step 2: Test**
- [ ] **Step 3: Commit**

### Task 13: Portfolio builder

**Files:**
- Create: `code/src/ensemble/portfolio.py`

- `compute_confidence(scores, model_agreement, rolling_ic)` → c ∈ [0,1]
- `build_portfolio(scores, confidence, alpha, top_k)` → DataFrame[stock_id, weight]

- [ ] **Step 1: Implement**
- [ ] **Step 2: Test**
- [ ] **Step 3: Commit**

### Task 14: Pipeline (train + predict)

**Files:**
- Create: `code/src/pipeline.py`
- Create: `code/src/train.py` (thin wrapper)
- Create: `code/src/predict.py` (thin wrapper)
- Create: `code/src/test.py` (alias for predict, required by competition)
- Create: `code/src/featurework.py` (alias for features/build, required by competition)

`pipeline.py train`:
1. Set seeds
2. Build features (all 3 sets)
3. Generate CV splits
4. For each model × fold × seed: train, evaluate, save
5. Refit on full data
6. Grid-search ensemble weights on holdout
7. Save ensemble_config.json

`pipeline.py predict`:
1. Load features (incremental)
2. Load all models
3. Each model scores latest day
4. Blend → Top-K → confidence → portfolio
5. Write output/result.csv

- [ ] **Step 1: Implement pipeline.py train**
- [ ] **Step 2: Implement pipeline.py predict**
- [ ] **Step 3: Create wrappers (train.py, predict.py, test.py, featurework.py)**
- [ ] **Step 4: End-to-end test**
- [ ] **Step 5: Commit**

### Task 15: Docker + shell scripts + readme

**Files:**
- Create/modify: `Dockerfile`
- Create/modify: `init.sh`, `train.sh`, `test.sh`
- Create: `readme.md`
- Create: `code/src/verify_reproducibility.py`

- [ ] **Step 1: Write Dockerfile**
- [ ] **Step 2: Write shell scripts**
- [ ] **Step 3: Write readme.md (competition format)**
- [ ] **Step 4: Write verify_reproducibility.py**
- [ ] **Step 5: Docker build + test**
- [ ] **Step 6: Commit**
