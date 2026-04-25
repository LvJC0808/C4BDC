from __future__ import annotations

from pathlib import Path

from code.src.data_schema import (
    STOCK_REQUIRED_COLUMNS,
    stock_data_schema_missing,
    stock_data_schema_ok,
)


def test_stock_data_schema_ok_with_full_16_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "stock_data.csv"
    csv_path.write_text(",".join(STOCK_REQUIRED_COLUMNS) + "\n", encoding="utf-8")

    assert stock_data_schema_missing(csv_path) == []
    assert stock_data_schema_ok(csv_path) is True


def test_stock_data_schema_detects_missing_valuation_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "stock_data.csv"
    csv_path.write_text(",".join(STOCK_REQUIRED_COLUMNS[:-4]) + "\n", encoding="utf-8")

    assert stock_data_schema_missing(csv_path) == ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    assert stock_data_schema_ok(csv_path) is False
