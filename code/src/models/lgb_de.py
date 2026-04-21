"""LightGBM single model and DoubleEnsemble wrapper (KDD 2020).

Sample Reweighting (SR): shrinkage sigmoid on normalized ranks of |residual|.
Feature Reweighting (FR): permutation-importance softmax on MAE deltas,
applied as column scaling proxy.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import lightgbm as lgb

from ..config import (
    LGB_PARAMS,
    LGB_ROUNDS,
    LGB_EARLY_STOP,
    LGB_DE_ROUNDS,
    SEED,
)


class LGBModel:
    def __init__(
        self,
        params: Optional[dict] = None,
        num_rounds: int = LGB_ROUNDS,
        early_stopping: int = LGB_EARLY_STOP,
        seed: Optional[int] = None,
    ) -> None:
        self.params = dict(params) if params is not None else dict(LGB_PARAMS)
        if seed is not None:
            self.params["seed"] = seed
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.booster: Optional[lgb.Booster] = None
        self.best_iteration_: int = 0
        self.feature_weight: Optional[np.ndarray] = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.feature_weight is None:
            return X
        return X * self.feature_weight[np.newaxis, :]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_weight: Optional[np.ndarray] = None,
    ) -> "LGBModel":
        if feature_weight is not None:
            self.feature_weight = np.asarray(feature_weight, dtype=np.float64)
        Xs = self._scale(np.asarray(X))
        dtrain = lgb.Dataset(Xs, label=np.asarray(y), weight=sample_weight)

        valid_sets = [dtrain]
        valid_names = ["train"]
        has_val = X_val is not None and y_val is not None
        if has_val:
            Xvs = self._scale(np.asarray(X_val))
            dval = lgb.Dataset(Xvs, label=np.asarray(y_val), reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")

        callbacks = [lgb.log_evaluation(0)]
        if has_val:
            callbacks.insert(
                0, lgb.early_stopping(self.early_stopping, verbose=False)
            )

        self.booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster.best_iteration or self.num_rounds
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.booster is not None, "Model not trained"
        Xs = self._scale(np.asarray(X))
        return self.booster.predict(Xs, num_iteration=self.best_iteration_)

    def save(self, path: str) -> None:
        assert self.booster is not None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.booster.save_model(path)
        fw_path = path + ".fw.npy"
        if self.feature_weight is not None:
            np.save(fw_path, self.feature_weight)
        elif os.path.exists(fw_path):
            os.remove(fw_path)

    @classmethod
    def load(cls, path: str) -> "LGBModel":
        m = cls()
        m.booster = lgb.Booster(model_file=path)
        m.best_iteration_ = m.booster.best_iteration or 0
        fw_path = path + ".fw.npy"
        if os.path.exists(fw_path):
            m.feature_weight = np.load(fw_path)
        return m


class DoubleEnsembleModel:
    def __init__(
        self,
        n_rounds: int = LGB_DE_ROUNDS,
        seed: int = SEED,
        lgb_params: Optional[dict] = None,
        num_rounds: int = LGB_ROUNDS,
        early_stopping: int = LGB_EARLY_STOP,
        bucket_count: int = 10,
        alpha_sr: float = 1.0,
        beta_fr: float = 1.0,
    ) -> None:
        self.n_rounds = n_rounds
        self.seed = seed
        self.lgb_params = dict(lgb_params) if lgb_params is not None else dict(LGB_PARAMS)
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self.bucket_count = bucket_count
        self.alpha_sr = alpha_sr
        self.beta_fr = beta_fr
        self.submodels: list[LGBModel] = []

    def _sr_weights(self, losses: np.ndarray) -> np.ndarray:
        n = len(losses)
        ranks = losses.argsort().argsort().astype(np.float64)
        r = ranks / max(n - 1, 1)
        w = 1.0 / (1.0 + np.exp(self.alpha_sr * (r - 0.5)))
        w *= n / w.sum()  # mean(w)==1
        return w

    def _fr_weights(
        self,
        model: LGBModel,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n, d = X.shape
        m = min(5000, n)
        idx = rng.choice(n, size=m, replace=False)
        Xs = X[idx]
        ys = y[idx]
        base = model.predict(Xs)
        base_mae = float(np.mean(np.abs(ys - base)))
        delta = np.zeros(d, dtype=np.float64)
        for j in range(d):
            Xp = Xs.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred = model.predict(Xp)
            delta[j] = float(np.mean(np.abs(ys - pred))) - base_mae
        std = delta.std() + 1e-9
        z = delta / std
        logits = self.beta_fr * z
        logits -= logits.max()
        v = np.exp(logits)
        v = v / v.sum() * d
        v = np.maximum(v, 0.1)
        return v

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "DoubleEnsembleModel":
        X = np.asarray(X)
        y = np.asarray(y)
        self.submodels = []
        rng = np.random.default_rng(self.seed)

        # Round 0: uniform
        m0 = LGBModel(
            params=self.lgb_params,
            num_rounds=self.num_rounds,
            early_stopping=self.early_stopping,
            seed=self.seed,
        )
        m0.fit(X, y, X_val, y_val)
        self.submodels.append(m0)

        for r in range(1, self.n_rounds):
            preds = np.mean([m.predict(X) for m in self.submodels], axis=0)
            L = np.abs(y - preds)
            w = self._sr_weights(L)
            sub_rng = np.random.default_rng(self.seed + r)
            v = self._fr_weights(self.submodels[-1], X, y, sub_rng)
            mr = LGBModel(
                params=self.lgb_params,
                num_rounds=self.num_rounds,
                early_stopping=self.early_stopping,
                seed=self.seed + r,
            )
            mr.fit(X, y, X_val, y_val, sample_weight=w, feature_weight=v)
            self.submodels.append(mr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.submodels], axis=0)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for i, m in enumerate(self.submodels):
            m.save(os.path.join(directory, f"sub_{i}.txt"))
        meta = {
            "n_rounds": self.n_rounds,
            "seed": self.seed,
            "num_rounds": self.num_rounds,
            "early_stopping": self.early_stopping,
            "bucket_count": self.bucket_count,
            "alpha_sr": self.alpha_sr,
            "beta_fr": self.beta_fr,
            "n_submodels": len(self.submodels),
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, directory: str) -> "DoubleEnsembleModel":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        obj = cls(
            n_rounds=meta["n_rounds"],
            seed=meta["seed"],
            num_rounds=meta["num_rounds"],
            early_stopping=meta["early_stopping"],
            bucket_count=meta["bucket_count"],
            alpha_sr=meta["alpha_sr"],
            beta_fr=meta["beta_fr"],
        )
        obj.submodels = []
        for i in range(meta["n_submodels"]):
            obj.submodels.append(LGBModel.load(os.path.join(directory, f"sub_{i}.txt")))
        return obj


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    N, D = 2000, 20
    X = rng.standard_normal((N, D)).astype(np.float32)
    beta = rng.standard_normal(D).astype(np.float32)
    y = (X @ beta + 0.5 * rng.standard_normal(N)).astype(np.float32)

    split = int(0.8 * N)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    fast_params = dict(LGB_PARAMS)
    fast_params.update({
        "num_leaves": 16,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "num_threads": 2,
    })

    m = LGBModel(params=fast_params, num_rounds=200, early_stopping=20, seed=0)
    m.fit(X_tr, y_tr, X_val, y_val)
    pred = m.predict(X_val)
    assert pred.shape == (len(y_val),)
    mae = float(np.mean(np.abs(y_val - pred)))
    baseline = float(y_val.std()) / 2
    assert mae < baseline, f"mae {mae} !< baseline {baseline}"
    print(f"LGBModel: best_iter={m.best_iteration_} val_mae={mae:.4f} baseline={baseline:.4f}")

    de = DoubleEnsembleModel(
        n_rounds=2,
        seed=0,
        lgb_params=fast_params,
        num_rounds=200,
        early_stopping=20,
    )
    de.fit(X_tr, y_tr, X_val, y_val)
    assert len(de.submodels) == 2
    p1 = de.predict(X_val)

    with tempfile.TemporaryDirectory() as tmp:
        de.save(tmp)
        de2 = DoubleEnsembleModel.load(tmp)
        p2 = de2.predict(X_val)
        assert np.allclose(p1, p2, atol=1e-8), f"max diff {np.max(np.abs(p1 - p2))}"

    print(f"DoubleEnsemble: n_sub={len(de.submodels)} val_mae={float(np.mean(np.abs(y_val - p1))):.4f}")
    print("OK")
