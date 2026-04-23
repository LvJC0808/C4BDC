"""StockMixer model (Task 11) — port of SJTU-DMTai StockMixer (AAAI 2024).

Reference: /tmp/StockMixer/src/model.py (official implementation). We port the
three-path idea — TimeMixer / FeatureMixer / cross-stock Mixer — and adapt it to
our panel interface.

Reshape contract
----------------
Our Alpha360 feature panel (``code/src/features/alpha360.py``) produces 360 flat
columns ``alpha360_0 .. alpha360_359`` where the layout is row-major
``(T=60, F=6)``, i.e. column index ``r * 6 + f`` corresponds to the r-th past
day (0 = oldest, 59 = most recent) and f-th raw field in order
``[open, high, low, close, volume, amount]``.

For StockMixer we want a ``(T=lookback, 6)`` tensor per (date, stock). We take
the most recent ``lookback`` rows, i.e. columns
``alpha360_{(60-lookback)*6} .. alpha360_359`` reshaped to ``(lookback, 6)``.

If the upstream pipeline supplies a different ``lookback`` than the Alpha360
default 60, the trainer expects the 360-column layout and truncates the oldest
rows. No other layout is supported here.

Trainer exposes the same fit/predict/save/load interface as MasterTrainer.
"""
from __future__ import annotations

import json
import math
import pathlib
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import MIXER_CONFIG, SEED


# ---------------------------------------------------------------------------
# composite loss (duplicated from master.py spec; replace with import when
# master.py lands and exports it).
# ---------------------------------------------------------------------------

def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_w: float = 0.5,
    ic_w: float = 0.5,
    topk_w: float = 0.1,
    topk: int = 5,
) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    mse = F.mse_loss(pred, target)

    # IC term (pearson)
    pc = pred - pred.mean()
    tc = target - target.mean()
    denom = torch.sqrt((pc * pc).sum() * (tc * tc).sum() + 1e-12)
    ic = (pc * tc).sum() / denom
    ic_loss = 1.0 - ic

    # top-k term: negative mean of true returns among top-k predicted
    k = min(topk, pred.shape[0])
    if k > 0:
        _, idx = torch.topk(pred, k)
        topk_loss = -target[idx].mean()
    else:
        topk_loss = torch.tensor(0.0, device=pred.device)

    return mse_w * mse + ic_w * ic_loss + topk_w * topk_loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MixerBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        if self.drop:
            x = F.dropout(x, p=self.drop, training=self.training)
        x = self.fc2(x)
        return x


class TimeFeatureMixer(nn.Module):
    """One Mixer2d block: mix across T then across feature channels."""

    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.ln1 = nn.LayerNorm([time_steps, channels])
        self.ln2 = nn.LayerNorm([time_steps, channels])
        self.time_mixer = MixerBlock(time_steps, time_steps)
        self.feat_mixer = MixerBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C)
        y = self.ln1(x)
        y = y.permute(0, 2, 1)
        y = self.time_mixer(y)
        y = y.permute(0, 2, 1)
        x = x + y
        z = self.feat_mixer(self.ln2(x))
        return x + z


class PyramidalTimeMixer(nn.Module):
    """Multi-scale path: halve T ``scale`` times and run per-scale Mixer2d."""

    def __init__(self, time_steps: int, channels: int, scale: int):
        super().__init__()
        self.scale = scale
        self.time_steps = time_steps
        self.mixers = nn.ModuleList()
        self.scale_dims = []
        for i in range(scale):
            ts = max(1, time_steps // (2 ** i))
            self.scale_dims.append(ts)
            self.mixers.append(TimeFeatureMixer(ts, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns concat along T axis of per-scale outputs: (N, sum(ts), C)
        outs = []
        cur = x
        for i, m in enumerate(self.mixers):
            if i == 0:
                h = m(cur)
            else:
                # average-pool by factor 2 along T
                pooled = F.avg_pool1d(cur.permute(0, 2, 1), kernel_size=2, stride=2)
                cur = pooled.permute(0, 2, 1)
                h = m(cur)
            outs.append(h)
        return torch.cat(outs, dim=1)


class NoGraphStockMixer(nn.Module):
    """Cross-stock MLP: mixes along the N (stock) axis. market_scale = hidden."""

    def __init__(self, n_stocks: int, market_hidden: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_stocks)
        self.fc1 = nn.Linear(n_stocks, market_hidden)
        self.fc2 = nn.Linear(market_hidden, n_stocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) -> treat N as a dim to mix
        x = x.permute(1, 0)          # (D, N)
        x = self.ln(x)
        x = F.hardswish(self.fc1(x))
        x = self.fc2(x)              # (D, N)
        return x.permute(1, 0)       # (N, D)


class StockMixer(nn.Module):
    """Three parallel paths: time-mixer, feature-mixer, cross-stock-mixer.

    Input: (N_stocks, T, F) per date.
    Output: (N_stocks,) prediction.
    """

    def __init__(
        self,
        time_steps: int,
        feat_dim: int,
        hidden: int,
        scale: int,
        market_scale: int,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.feat_dim = feat_dim
        self.hidden = hidden

        # lift raw features -> hidden channels
        self.lift = nn.Linear(feat_dim, hidden)

        self.pyramidal = PyramidalTimeMixer(time_steps, hidden, scale)
        total_T = sum(max(1, time_steps // (2 ** i)) for i in range(scale))

        # feature-only mixer path
        self.feat_path = TimeFeatureMixer(time_steps, hidden)

        # channel reduction to scalar per (stock, time)
        self.channel_fc = nn.Linear(hidden, 1)
        self.time_fc_time = nn.Linear(total_T, 1)
        self.time_fc_feat = nn.Linear(time_steps, 1)

        self._market_scale = market_scale  # used to size NoGraphStockMixer hidden

        # cross-stock path is created lazily once we know N_stocks per batch;
        # we instead use hidden=market_scale * hidden as stable market_hidden.
        self.stock_mixer_hidden = max(8, market_scale * 8)
        self._stock_mixer: NoGraphStockMixer | None = None

        # final head: sum of three scalars
        self.out_scale = nn.Parameter(torch.ones(3))

    def _ensure_stock_mixer(self, n: int, device: torch.device) -> NoGraphStockMixer:
        if self._stock_mixer is None or self._stock_mixer.fc1.in_features != n:
            self._stock_mixer = NoGraphStockMixer(n, self.stock_mixer_hidden).to(device)
        return self._stock_mixer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, F_raw)
        h = self.lift(x)                 # (N, T, H)

        y_time = self.pyramidal(h)       # (N, total_T, H)
        y_time = self.channel_fc(y_time).squeeze(-1)  # (N, total_T)
        score_time = self.time_fc_time(y_time).squeeze(-1)  # (N,)

        y_feat = self.feat_path(h)       # (N, T, H)
        y_feat = self.channel_fc(y_feat).squeeze(-1)  # (N, T)
        score_feat = self.time_fc_feat(y_feat).squeeze(-1)  # (N,)

        # cross-stock: use y_feat as per-stock embedding (N, T)
        sm = self._ensure_stock_mixer(h.shape[0], h.device)
        y_stock = sm(y_feat)             # (N, T)
        score_stock = self.time_fc_feat(y_stock).squeeze(-1)  # (N,)

        w = self.out_scale
        return w[0] * score_time + w[1] * score_feat + w[2] * score_stock


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MixerTrainer:
    """Trainer with panel interface matching MasterTrainer.

    ``feature_cols`` is expected to be the Alpha360 360-column list. See
    module-level reshape contract.
    """

    def __init__(self, config: dict[str, Any] | None = None, device: str | None = None):
        cfg = dict(MIXER_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: StockMixer | None = None
        self._feature_cols: list[str] | None = None

    # ---- panel -> tensor per date ----------------------------------------

    def _reshape_row(self, feats: np.ndarray) -> np.ndarray:
        """Convert (N, 360) alpha360 rows to (N, T, 6) with T=lookback."""
        lookback = int(self.config["lookback"])
        feat_dim = int(self.config["feat_dim"])  # 6
        total = feats.shape[1]
        if total % feat_dim != 0:
            raise ValueError(f"feature cols {total} not divisible by feat_dim {feat_dim}")
        T_full = total // feat_dim
        arr = feats.reshape(feats.shape[0], T_full, feat_dim)
        if T_full < lookback:
            # pad with oldest row
            pad = np.repeat(arr[:, :1, :], lookback - T_full, axis=1)
            arr = np.concatenate([pad, arr], axis=1)
        else:
            arr = arr[:, T_full - lookback:, :]
        return arr.astype(np.float32)

    def _iter_dates(self, panel_df: pd.DataFrame, feature_cols: list[str]):
        date_col = "datetime" if "datetime" in panel_df.columns else "日期"
        for date, g in panel_df.groupby(date_col, sort=True):
            feats = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            arr = self._reshape_row(feats)  # (N, T, F)
            y = g["label"].to_numpy(dtype=np.float32) if "label" in g.columns else None
            yield date, arr, y

    # ---- fit -------------------------------------------------------------

    def fit(
        self,
        panel_df: pd.DataFrame,
        market_df: pd.DataFrame | None,
        feature_cols: list[str],
        val_panel: pd.DataFrame | None = None,
        val_market: pd.DataFrame | None = None,
    ) -> "MixerTrainer":
        _set_seed(SEED)
        self._feature_cols = list(feature_cols)
        cfg = self.config

        self.model = StockMixer(
            time_steps=int(cfg["lookback"]),
            feat_dim=int(cfg["feat_dim"]),
            hidden=int(cfg["hidden"]),
            scale=int(cfg["scale"]),
            market_scale=int(cfg["market_scale"]),
        ).to(self.device)

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        sched = CosineAnnealingLR(opt, T_max=int(cfg["epochs"]))

        train_batches = list(self._iter_dates(panel_df, feature_cols))
        val_batches = (
            list(self._iter_dates(val_panel, feature_cols)) if val_panel is not None else None
        )

        best_val = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        patience = int(cfg["patience"])
        bad = 0

        for epoch in range(int(cfg["epochs"])):
            self.model.train()
            total = 0.0
            np.random.shuffle(train_batches)
            for _date, x, y in train_batches:
                if y is None or x.shape[0] < 2:
                    continue
                xt = torch.from_numpy(x).to(self.device)
                yt = torch.from_numpy(y).to(self.device)
                opt.zero_grad()
                pred = self.model(xt)
                loss = composite_loss(pred, yt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                opt.step()
                total += float(loss.item())
            sched.step()

            if val_batches is not None:
                self.model.eval()
                v_total = 0.0
                n_v = 0
                with torch.no_grad():
                    for _date, x, y in val_batches:
                        if y is None or x.shape[0] < 2:
                            continue
                        xt = torch.from_numpy(x).to(self.device)
                        yt = torch.from_numpy(y).to(self.device)
                        pred = self.model(xt)
                        v_total += float(composite_loss(pred, yt).item())
                        n_v += 1
                v_loss = v_total / max(1, n_v)
                if v_loss < best_val - 1e-6:
                    best_val = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if epoch + 1 < cfg.get("min_epochs", 0):
                        continue
                    if bad >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # ---- predict ---------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        panel_df: pd.DataFrame,
        market_df: pd.DataFrame | None,
        feature_cols: list[str],
    ) -> np.ndarray:
        assert self.model is not None, "fit() first"
        self.model.eval()
        date_col = "datetime" if "datetime" in panel_df.columns else "日期"
        preds = np.zeros(len(panel_df), dtype=np.float32)
        # row index pointer per original order
        # group-by preserves indices, so we rebuild from positions
        out = pd.Series(index=panel_df.index, dtype=np.float32)
        for date, g in panel_df.groupby(date_col, sort=True):
            feats = g[feature_cols].to_numpy(dtype=np.float32, copy=False)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            arr = self._reshape_row(feats)
            xt = torch.from_numpy(arr).to(self.device)
            p = self.model(xt).detach().cpu().numpy()
            out.loc[g.index] = p
        preds = out.to_numpy(dtype=np.float32)
        return preds

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        assert self.model is not None
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "feature_cols": self._feature_cols,
            "state_dict": self.model.state_dict(),
            "stock_mixer_state": (
                self.model._stock_mixer.state_dict() if self.model._stock_mixer is not None else None
            ),
            "stock_mixer_n": (
                self.model._stock_mixer.fc1.in_features if self.model._stock_mixer is not None else None
            ),
        }
        torch.save(payload, path)

    def load(self, path: str | pathlib.Path) -> "MixerTrainer":
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.config = payload["config"]
        self._feature_cols = payload["feature_cols"]
        cfg = self.config
        self.model = StockMixer(
            time_steps=int(cfg["lookback"]),
            feat_dim=int(cfg["feat_dim"]),
            hidden=int(cfg["hidden"]),
            scale=int(cfg["scale"]),
            market_scale=int(cfg["market_scale"]),
        ).to(self.device)
        if payload.get("stock_mixer_n") is not None:
            sm = NoGraphStockMixer(int(payload["stock_mixer_n"]), self.model.stock_mixer_hidden).to(self.device)
            sm.load_state_dict(payload["stock_mixer_state"])
            self.model._stock_mixer = sm
        self.model.load_state_dict(payload["state_dict"])
        return self


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _set_seed(SEED)
    rng = np.random.RandomState(0)
    n_dates, n_stocks = 30, 50
    dates = pd.date_range("2020-01-01", periods=n_dates)
    codes = [f"S{i:03d}" for i in range(n_stocks)]

    rows = []
    feat_cols = [f"alpha360_{i}" for i in range(360)]
    for d in dates:
        for c in codes:
            feat = rng.randn(360).astype(np.float32)
            label = float(rng.randn() * 0.01)
            rows.append((c, d, *feat, label))
    panel = pd.DataFrame(rows, columns=["instrument", "datetime", *feat_cols, "label"])

    market = pd.DataFrame({"datetime": dates, "mkt_ret": rng.randn(n_dates) * 0.01})

    trainer = MixerTrainer(config={"epochs": 2, "lookback": 8, "patience": 2})
    trainer.fit(panel, market, feat_cols)

    preds = trainer.predict(panel, market, feat_cols)
    assert preds.shape[0] == len(panel), f"row count mismatch {preds.shape[0]} vs {len(panel)}"

    tmp = pathlib.Path("/tmp/_mixer_ckpt.pt")
    trainer.save(tmp)
    t2 = MixerTrainer().load(tmp)
    preds2 = t2.predict(panel, market, feat_cols)
    diff = float(np.max(np.abs(preds - preds2)))
    assert diff < 1e-5, f"save/load mismatch {diff}"
    print("OK")
