"""Reconstruct stock_data.csv with valuation columns from cached valuation parquet.

When `data/stock_data.csv` was clobbered by git restore to the baseline 12-col
version, we can still recover the 4 valuation columns (peTTM, pbMRQ, psTTM,
pcfNcfTTM) from the cached `temp/features/valuation_*.parquet` which contains
pe_raw / pb_raw / ps_raw / pcf_raw per (instrument, datetime).

Usage:
    python scripts/rebuild_stock_data_from_cache.py \
        --baseline data/stock_data.csv \
        --valuation temp/features/valuation_dc22ebe9597f.parquet \
        --out data/stock_data.csv
"""
from __future__ import annotations
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="data/stock_data.csv",
                    help="Baseline 12-col stock_data.csv to augment")
    ap.add_argument("--valuation", required=True,
                    help="Path to cached valuation parquet")
    ap.add_argument("--out", default="data/stock_data.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    print(f"Reading baseline: {args.baseline}")
    base = pd.read_csv(args.baseline)
    # Strip BOM from first col name
    base.columns = [c.lstrip("﻿") for c in base.columns]
    base["股票代码"] = base["股票代码"].astype(str).str.zfill(6)
    base["日期"] = pd.to_datetime(base["日期"])
    print(f"  rows: {len(base)}, cols: {list(base.columns)}")

    print(f"Reading valuation cache: {args.valuation}")
    val = pd.read_parquet(args.valuation)
    val["instrument"] = val["instrument"].astype(str).str.zfill(6)
    val["datetime"] = pd.to_datetime(val["datetime"])
    val = val[["instrument", "datetime", "pe_raw", "pb_raw", "ps_raw", "pcf_raw"]]
    val = val.rename(columns={
        "instrument": "股票代码",
        "datetime": "日期",
        "pe_raw": "peTTM",
        "pb_raw": "pbMRQ",
        "ps_raw": "psTTM",
        "pcf_raw": "pcfNcfTTM",
    })
    print(f"  val rows: {len(val)}, dates: {val['日期'].nunique()}")

    merged = base.merge(val, on=["股票代码", "日期"], how="left")
    nan_ratio = merged[["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]].isna().mean()
    print(f"Merged: {len(merged)} rows")
    print("NaN ratios (valuation cols):")
    print(nan_ratio)

    # Reorder: baseline cols first, then new valuation cols
    out_cols = list(base.columns) + ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
    merged = merged[out_cols]

    # Format date back to string if baseline used string dates
    merged["日期"] = merged["日期"].dt.strftime("%Y/%-m/%-d")  # match baseline format '2024/1/2'

    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(merged.columns)} columns")
    print(f"  columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
