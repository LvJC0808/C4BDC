"""Tradability filter: exclude stocks hitting price limits on target date."""
from __future__ import annotations
import numpy as np
import pandas as pd


def is_limit_up(pct_chg: float, code: str, threshold_main: float = 9.9,
                threshold_special: float = 19.9) -> bool:
    """Return True if stock hit upper price limit."""
    # ChiNext (300xxx) and STAR (688xxx) have 20% limits
    if code.startswith("30") or code.startswith("68"):
        return pct_chg >= threshold_special
    return pct_chg >= threshold_main


def is_limit_down(pct_chg: float, code: str, threshold_main: float = -9.9,
                  threshold_special: float = -19.9) -> bool:
    if code.startswith("30") or code.startswith("68"):
        return pct_chg <= threshold_special
    return pct_chg <= threshold_main


def get_tradable_ids(stock_df: pd.DataFrame, target_date: pd.Timestamp,
                     exclude_limit_up: bool = True,
                     exclude_limit_down: bool = True) -> set:
    """Return set of stock codes tradable on target_date.

    A stock is untradable if on target_date it hit the price limit
    (涨停 or 跌停), because next-day open will gap and may not fill.

    Parameters
    ----------
    stock_df : DataFrame with columns 股票代码, 日期, 涨跌幅
    target_date : the T date (we check T's price change)
    """
    stock_df = stock_df.copy()
    stock_df["日期"] = pd.to_datetime(stock_df["日期"])
    stock_df["股票代码"] = stock_df["股票代码"].astype(str).str.zfill(6)
    day = stock_df[stock_df["日期"] == target_date]
    if day.empty:
        return set(stock_df["股票代码"].unique())

    excluded = set()
    for _, row in day.iterrows():
        code = row["股票代码"]
        pct = float(row.get("涨跌幅", 0))
        if exclude_limit_up and is_limit_up(pct, code):
            excluded.add(code)
        if exclude_limit_down and is_limit_down(pct, code):
            excluded.add(code)
    all_codes = set(day["股票代码"].unique())
    return all_codes - excluded


def filter_tradable_topk(scores_df: pd.DataFrame, tradable_ids: set,
                         top_k: int, expand: int = 5,
                         score_col: str = "final_score",
                         id_col: str = "instrument") -> list:
    """Pick top-K from scores, skipping untradable stocks.

    If after filtering fewer than max(top_k-2, 1) remain, relax to
    whatever is available.
    """
    ranked = scores_df.sort_values(score_col, ascending=False)
    tradable = [r for _, r in ranked.iterrows()
                if r[id_col] in tradable_ids]
    n = max(min(top_k, len(tradable)), max(top_k - 2, 1))
    return [r[id_col] for r in tradable[:n]]


def _self_test():
    import pandas as pd
    codes = ["000001", "300001", "688001", "600001"]
    df = pd.DataFrame({
        "股票代码": codes,
        "日期": pd.Timestamp("2026-03-20"),
        "涨跌幅": [5.0, 20.1, 19.95, -10.1],
    })
    t = get_tradable_ids(df, pd.Timestamp("2026-03-20"))
    assert "000001" in t  # normal, not limit
    assert "300001" not in t  # ChiNext 20.1 >= 19.9 → limit up
    assert "688001" not in t  # STAR 19.95 >= 19.9 → limit up
    assert "600001" not in t  # main board -10.1 <= -9.9 → limit down
    print("OK")


if __name__ == "__main__":
    _self_test()
