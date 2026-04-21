"""Walk-forward CV with embargo.

3-fold rolling-origin split with a 5-day embargo between train and val
(required because labels span T+1..T+5 → 5 future trading days of leakage).

A final 20-day holdout at the end is reserved for ensemble-weight /
position-alpha / top-K grid search.

Refit epochs = median CV epochs × 1.05, applied when fitting the final
full-data model used for production prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Split:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


@dataclass
class CVPlan:
    splits: List[Split]
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp


def generate_cv_splits(
    dates: pd.Series,
    n_folds: int = 3,
    embargo: int = 5,
    val_days: int = 20,
    holdout_days: int = 20,
    fold_gap: int = 50,
) -> CVPlan:
    """Generate walk-forward CV splits from a sorted series of trading dates.

    Layout (from the end going back):
        [..train..] [embargo] [val_fold_k] ... [holdout]

    Fold `k` (k = 0 is earliest, k = n_folds-1 is latest) ends at
    (last_date − holdout_days − (n_folds−1−k) * fold_gap).
    Train data for each fold is everything up to its train_end.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(pd.Series(sorted(pd.unique(dates))))
    else:
        dates = pd.Series(sorted(dates))

    if len(dates) < holdout_days + val_days + embargo + 100:
        raise ValueError(
            f"Not enough trading days ({len(dates)}) for CV config")

    last = dates.iloc[-1]
    holdout_end = last
    holdout_start = dates.iloc[-holdout_days]

    splits: List[Split] = []
    for k in range(n_folds):
        # latest fold is closest to holdout
        offset_from_end = holdout_days + (n_folds - 1 - k) * fold_gap
        val_end_idx = len(dates) - 1 - offset_from_end
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - embargo - 1

        if val_start_idx <= 0 or train_end_idx <= 100:
            raise ValueError(f"Fold {k}: insufficient history")

        splits.append(Split(
            fold=k,
            train_start=dates.iloc[0],
            train_end=dates.iloc[train_end_idx],
            val_start=dates.iloc[val_start_idx],
            val_end=dates.iloc[val_end_idx],
        ))

    return CVPlan(
        splits=splits,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )


def get_refit_epochs(cv_best_epochs: List[int], slack: float = 1.05) -> int:
    """Median CV best-epochs × 1.05 (rounded up).

    Used to set the stopping epoch for the final refit on full data,
    since there is no validation set during refit.
    """
    if not cv_best_epochs:
        return 40
    med = int(np.median(cv_best_epochs))
    return int(np.ceil(med * slack))


def filter_panel(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                 date_col: str = 'datetime') -> pd.DataFrame:
    """Return rows with date_col ∈ [start, end] inclusive."""
    mask = (panel[date_col] >= start) & (panel[date_col] <= end)
    return panel.loc[mask]


def describe_cv(plan: CVPlan) -> str:
    out = []
    for s in plan.splits:
        out.append(f"  fold {s.fold}: train {s.train_start.date()}..{s.train_end.date()}  "
                   f"val {s.val_start.date()}..{s.val_end.date()}")
    out.append(f"  holdout: {plan.holdout_start.date()}..{plan.holdout_end.date()}")
    return "\n".join(out)


# ---------------- Self-test ----------------

if __name__ == '__main__':
    dates = pd.bdate_range('2020-01-01', '2025-12-31')
    plan = generate_cv_splits(dates)
    print(describe_cv(plan))
