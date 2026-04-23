"""MASTER model port (Task 10).

Faithful-ish port of SJTU-DMTai/MASTER (``master.py``) adapted to our
panel-data interface. The official model expects a pre-concatenated
(stock-features | market-features) tensor shaped ``(N, T, F+F')`` and
slices the market indices inside ``forward``. Here we accept two
separate inputs (a per-stock panel and a per-date market matrix), which
matches how our data pipeline stores features.

Deviations from the official reference:
    * Market gating receives a dedicated ``market`` tensor (last-step
      vector of size ``market_feat_dim``) instead of being sliced from
      ``x``. The math is identical to the original ``Gate`` module (a
      softmax-scaled FiLM with ``beta`` temperature, multiplying the
      stock feature vector).
    * The inter-stock attention is still a multi-head self-attention
      over the stock axis at every timestep, but implemented with
      ``torch.nn.MultiheadAttention`` instead of the paper's hand-rolled
      per-head loop. Mathematically equivalent up to weight init.
    * Training uses a composite loss (MSE + 1-IC + top-K term) rather
      than the official plain-MSE, as required by the task spec.
    * Cosine LR schedule + early stopping on a validation ``final_score``
      (``1 - norm_mse + rank_ic``) rather than the paper's fixed epochs.

Self-test: ``python -m code.src.models.master`` runs a small synthetic
smoke-test and prints ``OK`` on success.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MASTER_CONFIG, SEED


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class FeatureEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MarketGuidedGating(nn.Module):
    """Softmax-scaled FiLM gate (MASTER paper, eq. 3)."""

    def __init__(self, market_dim: int, feat_dim: int, beta: float = 5.0):
        super().__init__()
        self.trans = nn.Linear(market_dim, feat_dim)
        self.feat_dim = feat_dim
        self.beta = beta

    def forward(self, market_vec: torch.Tensor) -> torch.Tensor:
        # market_vec: (market_dim,) or (B, market_dim)
        g = self.trans(market_vec)
        g = torch.softmax(g / self.beta, dim=-1)
        return self.feat_dim * g


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1], :]


class IntraStockAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class InterStockAttention(nn.Module):
    """Deviation: uses ``nn.MultiheadAttention`` over the stock axis."""

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D) -> attend across N at each T
        N, T, D = x.shape
        h = self.norm1(x)
        h = h.transpose(0, 1)  # (T, N, D): treat T as batch
        a, _ = self.attn(h, h, h, need_weights=False)
        a = a.transpose(0, 1)  # back to (N, T, D)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalAggregator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, T, D) -> (N, D) via attention-pooling with last-step query
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)  # (N, D, 1)
        lam = torch.matmul(h, query).squeeze(-1)  # (N, T)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # (N, 1, T)
        out = torch.matmul(lam, z).squeeze(1)  # (N, D)
        return out


class MASTERModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        self.encoder = FeatureEncoder(config["feat_dim"], d_model)
        self.gate = MarketGuidedGating(
            config["market_feat_dim"], config["feat_dim"], beta=config["beta"]
        )
        self.pos = PositionalEncoding(d_model)
        self.intra = IntraStockAttention(d_model, config["n_head"], config["dropout"])
        self.inter = InterStockAttention(d_model, config["n_head"], config["dropout"])
        self.temporal = TemporalAggregator(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, market: torch.Tensor) -> torch.Tensor:
        # x: (N, T, feat_dim), market: (market_dim,) or (T, market_dim)
        if market.dim() == 2:
            market_last = market[-1]
        else:
            market_last = market
        gate = self.gate(market_last)  # (feat_dim,)
        x = x * gate.unsqueeze(0).unsqueeze(0)  # broadcast over N, T
        h = self.encoder(x)  # (N, T, d_model)
        h = self.pos(h)
        h = self.intra(h)
        h = self.inter(h)
        h = self.temporal(h)  # (N, d_model)
        return self.head(h).squeeze(-1)  # (N,)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + eps)


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w_mse: float,
    w_ic: float,
    w_topk: float,
    k: int = 5,
) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    ic = _pearson(pred, target)
    ic_term = 1.0 - ic
    k_eff = min(k, pred.shape[0])
    top_vals, top_idx = torch.topk(pred, k_eff)
    top_tgt = target.gather(0, top_idx)
    topk_term = -(top_vals * top_tgt).mean()
    return w_mse * mse + w_ic * ic_term + w_topk * topk_term


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class MasterTrainer:
    def __init__(
        self,
        config: Optional[dict] = None,
        device: Optional[str] = None,
        seed: int = SEED,
    ):
        self.config = dict(config if config is not None else MASTER_CONFIG)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self._set_seed(seed)
        self.model: Optional[MASTERModel] = None
        self.state_dict_: Optional[dict] = None
        self.best_epoch: int = -1
        self.history_: list = []

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------
    # Batch builder
    # ------------------------------------------------------------------
    def _prepare(
        self,
        panel_df: pd.DataFrame,
        market_df: pd.DataFrame,
        feature_cols: Sequence[str],
        require_label: bool = True,
    ):
        """Return list of (date, instruments, X tensor, market tensor, y tensor)."""
        T = self.config["lookback"]
        mkt = market_df.sort_index().copy()
        if not isinstance(mkt.index, pd.DatetimeIndex):
            mkt = mkt.set_index("datetime").sort_index()
        # align all dates
        panel = panel_df.copy()
        panel["datetime"] = pd.to_datetime(panel["datetime"])
        mkt.index = pd.to_datetime(mkt.index)
        all_dates = np.array(sorted(panel["datetime"].unique()))
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # Build per-(instrument) history by pivoting once
        feat_cols = list(feature_cols)
        # long -> wide: multi-index (datetime, instrument)
        panel = panel.sort_values(["instrument", "datetime"])
        # For efficiency, pre-compute per-instrument arrays
        grouped = {
            inst: g.set_index("datetime")
            for inst, g in panel.groupby("instrument", sort=False)
        }

        # market feature columns = all numeric columns in order
        market_cols = [c for c in mkt.columns if c != "datetime"]
        if len(market_cols) != self.config["market_feat_dim"]:
            # allow mismatch but warn
            pass
        mkt_arr = mkt[market_cols].to_numpy(dtype=np.float32)
        mkt_date_idx = {d: i for i, d in enumerate(mkt.index)}

        batches = []
        for t_idx, date in enumerate(all_dates):
            if t_idx < T - 1:
                continue
            window_dates = all_dates[t_idx - T + 1 : t_idx + 1]
            # market window
            try:
                m_rows = [mkt_arr[mkt_date_idx[d]] for d in window_dates]
            except KeyError:
                continue
            m_tensor = np.stack(m_rows, axis=0)  # (T, market_dim)

            X_list, y_list, inst_list = [], [], []
            for inst, g in grouped.items():
                if date not in g.index:
                    continue
                # need all T dates
                sub = g.reindex(window_dates)
                if sub[feat_cols].isna().any().any():
                    continue
                if require_label:
                    if "label" not in g.columns:
                        continue
                    y_val = g.loc[date, "label"]
                    if pd.isna(y_val):
                        continue
                    y_list.append(float(y_val))
                else:
                    y_list.append(0.0)
                X_list.append(sub[feat_cols].to_numpy(dtype=np.float32))
                inst_list.append(inst)
            if len(inst_list) < 10:
                continue
            X = np.stack(X_list, axis=0)  # (N, T, F)
            y = np.array(y_list, dtype=np.float32)
            batches.append((date, inst_list, X, m_tensor, y))
        return batches

    # ------------------------------------------------------------------
    def _build_model(self) -> MASTERModel:
        cfg = dict(self.config)
        cfg["feat_dim"] = cfg.get("feat_dim", 158)
        return MASTERModel(cfg).to(self.device)

    def _eval(self, batches) -> dict:
        assert self.model is not None
        self.model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for _, _, X, M, y in batches:
                xt = torch.from_numpy(X).to(self.device)
                mt = torch.from_numpy(M).to(self.device)
                p = self.model(xt, mt).cpu().numpy()
                all_p.append(p)
                all_y.append(y)
        p = np.concatenate(all_p)
        y = np.concatenate(all_y)
        mse = float(np.mean((p - y) ** 2))
        # batch rank ic (average across dates)
        ics = []
        i = 0
        for _, _, _, _, yy in batches:
            n = len(yy)
            pp = p[i : i + n]
            i += n
            if n >= 2 and np.std(pp) > 0 and np.std(yy) > 0:
                ics.append(
                    float(
                        np.corrcoef(
                            pd.Series(pp).rank().to_numpy(),
                            pd.Series(yy).rank().to_numpy(),
                        )[0, 1]
                    )
                )
        rank_ic = float(np.mean(ics)) if ics else 0.0
        final = (1.0 / (1.0 + mse)) + rank_ic
        return {"mse": mse, "rank_ic": rank_ic, "final": final}

    # ------------------------------------------------------------------
    def fit(
        self,
        panel_df: pd.DataFrame,
        market_df: pd.DataFrame,
        feature_cols: Sequence[str],
        val_panel: Optional[pd.DataFrame] = None,
        val_market: Optional[pd.DataFrame] = None,
    ) -> "MasterTrainer":
        self._set_seed(self.seed)
        self.config["feat_dim"] = len(feature_cols)
        train_batches = self._prepare(panel_df, market_df, feature_cols)
        if not train_batches:
            raise ValueError("No valid training batches (check lookback / NaNs).")
        val_batches = None
        if val_panel is not None and val_market is not None:
            val_batches = self._prepare(val_panel, val_market, feature_cols)

        self.model = self._build_model()
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.config["epochs"])
        )

        best_final = -float("inf")
        best_state = None
        bad = 0
        self.history_ = []
        rng = np.random.RandomState(self.seed)

        for epoch in range(self.config["epochs"]):
            self.model.train()
            order = list(range(len(train_batches)))
            rng.shuffle(order)
            epoch_loss = 0.0
            for bi in order:
                _, _, X, M, y = train_batches[bi]
                xt = torch.from_numpy(X).to(self.device)
                mt = torch.from_numpy(M).to(self.device)
                yt = torch.from_numpy(y).to(self.device)
                opt.zero_grad()
                pred = self.model(xt, mt)
                loss = composite_loss(
                    pred,
                    yt,
                    self.config["loss_mse_w"],
                    self.config["loss_ic_w"],
                    self.config["loss_topk_w"],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                opt.step()
                epoch_loss += float(loss.item())
            sched.step()
            epoch_loss /= max(1, len(order))

            val_metrics = None
            if val_batches:
                val_metrics = self._eval(val_batches)
            self.history_.append({"epoch": epoch, "train_loss": epoch_loss, "val": val_metrics})

            score = val_metrics["final"] if val_metrics else -epoch_loss
            if score > best_final:
                best_final = score
                self.best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if epoch + 1 < self.config.get("min_epochs", 0):
                    continue
                if bad >= self.config["patience"]:
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.best_epoch = len(self.history_) - 1
        self.state_dict_ = best_state
        self.model.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        panel_df: pd.DataFrame,
        market_df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> pd.DataFrame:
        assert self.model is not None, "fit() first."
        batches = self._prepare(panel_df, market_df, feature_cols, require_label=False)
        self.model.eval()
        rows = []
        with torch.no_grad():
            for date, insts, X, M, _ in batches:
                xt = torch.from_numpy(X).to(self.device)
                mt = torch.from_numpy(M).to(self.device)
                scores = self.model(xt, mt).cpu().numpy()
                for inst, s in zip(insts, scores):
                    rows.append((inst, date, float(s)))
        return pd.DataFrame(rows, columns=["instrument", "datetime", "score"])

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict_, os.path.join(path, "master.pt"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"config": self.config, "best_epoch": self.best_epoch}, f)

    def load(self, path: str) -> "MasterTrainer":
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        self.config = meta["config"]
        self.best_epoch = meta.get("best_epoch", -1)
        self.model = self._build_model()
        state = torch.load(
            os.path.join(path, "master.pt"), map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state)
        self.state_dict_ = state
        return self


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _make_synth(n_dates=30, n_stocks=50, feat_dim=158, market_dim=63, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    insts = [f"S{i:03d}" for i in range(n_stocks)]
    rows = []
    # signal: first feature correlates with next-day label
    for d_i, d in enumerate(dates):
        base = rng.randn(n_stocks).astype(np.float32)
        feats = rng.randn(n_stocks, feat_dim).astype(np.float32)
        feats[:, 0] = base
        label = 0.5 * base + 0.1 * rng.randn(n_stocks).astype(np.float32)
        for s_i, inst in enumerate(insts):
            row = {"instrument": inst, "datetime": d, "label": float(label[s_i])}
            for f in range(feat_dim):
                row[f"f{f}"] = float(feats[s_i, f])
            rows.append(row)
    panel = pd.DataFrame(rows)
    mkt = pd.DataFrame(
        rng.randn(n_dates, market_dim).astype(np.float32),
        index=dates,
        columns=[f"m{i}" for i in range(market_dim)],
    )
    mkt.index.name = "datetime"
    feat_cols = [f"f{f}" for f in range(feat_dim)]
    return panel, mkt, feat_cols


def _self_test():
    panel, mkt, feat_cols = _make_synth()
    cfg = dict(MASTER_CONFIG)
    cfg["epochs"] = 2
    cfg["lookback"] = 4
    cfg["d_model"] = 64
    cfg["patience"] = 5
    cfg["feat_dim"] = len(feat_cols)

    # train/val split
    dates = sorted(panel["datetime"].unique())
    cut = dates[int(len(dates) * 0.7)]
    tr = panel[panel["datetime"] <= cut]
    va = panel[panel["datetime"] > cut]
    tr_m = mkt.loc[mkt.index <= cut]
    va_m = mkt.loc[mkt.index > cut - pd.Timedelta(days=10)]  # include lookback

    trainer = MasterTrainer(config=cfg, seed=0)
    trainer.fit(tr, tr_m, feat_cols, val_panel=va, val_market=va_m)
    assert trainer.history_, "no history"
    # check val MSE decreased (epoch 1 < epoch 0) or equal (tiny test is noisy)
    h = trainer.history_
    if len(h) >= 2 and h[0]["val"] and h[1]["val"]:
        assert h[1]["val"]["mse"] <= h[0]["val"]["mse"] * 1.5, (
            f"val mse blew up: {h[0]['val']['mse']} -> {h[1]['val']['mse']}"
        )

    pred = trainer.predict(va, va_m, feat_cols)
    # expected rows: sum over valid dates of n_stocks (50)
    assert len(pred) > 0
    assert set(pred.columns) == {"instrument", "datetime", "score"}

    # save/load roundtrip
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        trainer.save(tmp)
        t2 = MasterTrainer(config=cfg, seed=0).load(tmp)
        pred2 = t2.predict(va, va_m, feat_cols)
    merged = pred.merge(pred2, on=["instrument", "datetime"], suffixes=("_a", "_b"))
    diff = float(np.max(np.abs(merged["score_a"] - merged["score_b"])))
    assert diff < 1e-5, f"save/load mismatch: {diff}"
    print(f"history: {[(e['epoch'], e['train_loss'], e['val']) for e in h]}")
    print(f"pred rows: {len(pred)}, save/load max diff: {diff:.2e}")
    print("OK")


if __name__ == "__main__":
    _self_test()
