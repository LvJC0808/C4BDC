# code/src/config.py
import os

SEED = 42
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,2024,7").split(",")]

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
    "num_threads": int(os.environ.get("LGB_NUM_THREADS", "4")),
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
    "min_epochs": 15,
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
    "min_epochs": 15,
}

# Ensemble
ENSEMBLE_INIT_WEIGHTS = {"lgb": 0.4, "master": 0.4, "mixer": 0.2}
TOP_K_CANDIDATES = [3, 4, 5]
POSITION_ALPHAS = [0.3, 0.5, 0.7]
MIN_POSITION = 0.5

# Sample pool filters
MIN_LIST_DAYS = 250
MAX_SUSPEND_DAYS = 5

# ---- Ensemble v2 (ICIR robust) ----
ENSEMBLE_METHOD = os.environ.get("ENSEMBLE_METHOD", "icir_shrink")  # "icir_shrink" | "legacy"
ICIR_LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
ICIR_BOOTSTRAP_N = int(os.environ.get("ICIR_BOOTSTRAP_N", "1000"))
ICIR_MULTI_START_DIRICHLET = 10
