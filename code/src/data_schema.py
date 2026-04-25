"""Schema helpers for judge-mounted CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STOCK_BASE_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]

STOCK_VALUATION_COLUMNS = [
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
]

STOCK_REQUIRED_COLUMNS = STOCK_BASE_COLUMNS + STOCK_VALUATION_COLUMNS


def stock_data_schema_missing(csv_path: str | Path) -> list[str]:
    """Return required columns missing from stock_data.csv."""
    header = pd.read_csv(csv_path, nrows=0)
    cols = set(header.columns)
    return [col for col in STOCK_REQUIRED_COLUMNS if col not in cols]


def stock_data_schema_ok(csv_path: str | Path) -> bool:
    """Whether stock_data.csv has the full 16-column schema required by runtime."""
    return not stock_data_schema_missing(csv_path)
