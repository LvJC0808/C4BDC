#!/usr/bin/env python3
"""
BDC2026 离线数据抓取（Z1 扩展版）

在联网环境一次性执行，产物打包进 docker，运行时不再联网。

爬取内容:
  1. stock_data.csv      HS300 日线 OHLCV + 换手率 + 涨跌幅 + 估值 (peTTM/pbMRQ/psTTM/pcfNcfTTM)
  2. industry_map.csv    申万一级行业映射
  3. stock_basic.csv     上市日期 / 状态
  4. hs300_history.csv   沪深300历史成分股（半年频快照）
  5. csi300_index.csv    沪深300指数日线（sh.000300）
  6. trade_calendar.csv  交易日历

用法:
    python fetch_all.py --start 2018-01-01 --end 2026-04-18 --out ./data
"""

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timedelta

import baostock as bs
import pandas as pd
from tqdm import tqdm


# ---------------- baostock 会话 ----------------

def login():
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    print("baostock logged in.")


def logout():
    bs.logout()
    print("baostock logged out.")


def _rs_to_df(rs, label=""):
    if rs.error_code != '0':
        raise RuntimeError(f"{label} query failed: {rs.error_msg}")
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def _retry(fn, *args, retries=3, sleep=2, **kwargs):
    last = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(sleep * (i + 1))
    raise last


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _report(path):
    size = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ {path}  [{size:.2f} MB]  md5={_md5(path)}")


# ---------------- 1. HS300 当前成分股（用于主数据抓取） ----------------

def get_current_hs300():
    rs = bs.query_hs300_stocks()
    df = _rs_to_df(rs, 'hs300')
    return df  # columns: updateDate, code, code_name


# ---------------- 2. 日线 + 估值 ----------------

def fetch_stock_data(start, end, out_dir, stock_list):
    """HS300 日线 + 估值。字段与 baseline 兼容并新增 peTTM/pbMRQ/psTTM/pcfNcfTTM。"""
    out_path = os.path.join(out_dir, "stock_data.csv")
    fields = ("date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,"
              "peTTM,pbMRQ,psTTM,pcfNcfTTM")

    all_frames = []
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="stock_data"):
        bs_code = row['code']
        try:
            rs = _retry(bs.query_history_k_data_plus, bs_code, fields,
                        start_date=start, end_date=end,
                        frequency='d', adjustflag='1')  # 后复权，与 baseline 一致
            df = _rs_to_df(rs, bs_code)
            if df.empty:
                continue

            numeric = ['open', 'high', 'low', 'close', 'preclose', 'volume',
                       'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            for c in numeric:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(4)
            df['涨跌额'] = (df['close'] - df['preclose']).round(4)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%-m/%-d')
            df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)

            df = df.rename(columns={
                'code': '股票代码', 'date': '日期',
                'open': '开盘', 'close': '收盘', 'high': '最高', 'low': '最低',
                'volume': '成交量', 'amount': '成交额',
                'turn': '换手率', 'pctChg': '涨跌幅',
            })
            cols = ['股票代码', '日期', '开盘', '收盘', '最高', '最低',
                    '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
                    'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            all_frames.append(df[cols])
        except Exception as e:
            print(f"  ! {bs_code} failed: {e}", file=sys.stderr)

    result = pd.concat(all_frames, ignore_index=True)
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    _report(out_path)
    return out_path


# ---------------- 3. 行业映射 ----------------

def fetch_industry_map(out_dir):
    out_path = os.path.join(out_dir, "industry_map.csv")
    rs = _retry(bs.query_stock_industry)
    df = _rs_to_df(rs, 'industry')
    # fields: updateDate, code, code_name, industry, industryClassification
    df['stock_id'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    out = df[['stock_id', 'industry', 'industryClassification']].rename(
        columns={'industry': 'sw_industry'})
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    _report(out_path)
    return out_path


# ---------------- 4. 上市信息 ----------------

def fetch_stock_basic(out_dir, stock_list):
    out_path = os.path.join(out_dir, "stock_basic.csv")
    rows = []
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="stock_basic"):
        bs_code = row['code']
        try:
            rs = _retry(bs.query_stock_basic, code=bs_code)
            df = _rs_to_df(rs, bs_code)
            if not df.empty:
                rows.append(df.iloc[0])
        except Exception as e:
            print(f"  ! basic {bs_code}: {e}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df['stock_id'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    out = df[['stock_id', 'code_name', 'ipoDate', 'outDate', 'type', 'status']]
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    _report(out_path)
    return out_path


# ---------------- 5. HS300 历史成分股 ----------------

def fetch_hs300_history(start, end, out_dir):
    """每月第一个交易日抓一次成分股快照，避免幸存者偏差。"""
    out_path = os.path.join(out_dir, "hs300_history.csv")
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    # 生成每月 1 日的日期列表
    dates = []
    d = start_dt.replace(day=1)
    while d <= end_dt:
        dates.append(d.strftime("%Y-%m-%d"))
        # 下一个月
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)

    all_rows = []
    for dstr in tqdm(dates, desc="hs300_history"):
        try:
            rs = _retry(bs.query_hs300_stocks, date=dstr)
            df = _rs_to_df(rs, f'hs300@{dstr}')
            if df.empty:
                continue
            df['stock_id'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
            df['date'] = dstr
            all_rows.append(df[['date', 'stock_id']])
        except Exception as e:
            print(f"  ! hs300 {dstr}: {e}", file=sys.stderr)

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=['date', 'stock_id'])
    result.to_csv(out_path, index=False, encoding='utf-8-sig')
    _report(out_path)
    return out_path


# ---------------- 6. 沪深300 指数日线 ----------------

def fetch_csi300_index(start, end, out_dir):
    out_path = os.path.join(out_dir, "csi300_index.csv")
    fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    rs = _retry(bs.query_history_k_data_plus, 'sh.000300', fields,
                start_date=start, end_date=end,
                frequency='d', adjustflag='3')  # 指数无需复权
    df = _rs_to_df(rs, 'csi300')
    numeric = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(4)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    _report(out_path)
    return out_path


# ---------------- 7. 交易日历 ----------------

def fetch_trade_calendar(start, end, out_dir):
    out_path = os.path.join(out_dir, "trade_calendar.csv")
    rs = _retry(bs.query_trade_dates, start_date=start, end_date=end)
    df = _rs_to_df(rs, 'calendar')
    # fields: calendar_date, is_trading_day
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    _report(out_path)
    return out_path


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'))
    parser.add_argument('--out', default='./data')
    parser.add_argument('--hs300_list', default=None,
                        help='optional CSV path with column `code` (bs format sh.XXXXXX)')
    parser.add_argument('--skip', nargs='*', default=[],
                        choices=['stock_data', 'industry', 'basic',
                                 'hs300_history', 'csi300', 'calendar'],
                        help='sub-tasks to skip')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Window: {args.start} ~ {args.end}  Output: {args.out}")

    login()
    try:
        if args.hs300_list and os.path.exists(args.hs300_list):
            stock_list = pd.read_csv(args.hs300_list)
        else:
            stock_list = get_current_hs300()
            stock_list.to_csv(os.path.join(args.out, 'hs300_stock_list.csv'),
                              index=False, encoding='utf-8-sig')
        print(f"Universe: {len(stock_list)} stocks")

        if 'stock_data' not in args.skip:
            fetch_stock_data(args.start, args.end, args.out, stock_list)
        if 'industry' not in args.skip:
            fetch_industry_map(args.out)
        if 'basic' not in args.skip:
            fetch_stock_basic(args.out, stock_list)
        if 'hs300_history' not in args.skip:
            fetch_hs300_history(args.start, args.end, args.out)
        if 'csi300' not in args.skip:
            fetch_csi300_index(args.start, args.end, args.out)
        if 'calendar' not in args.skip:
            fetch_trade_calendar(args.start, args.end, args.out)

        print("\nAll done.")
    finally:
        logout()


if __name__ == '__main__':
    main()
