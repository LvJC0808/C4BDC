import pandas as pd
import numpy as np
from code.src.ensemble.regime import detect_regime


def _make_panel(dates, n_large=10, n_small=10,
                large_daily=0.005, small_daily=-0.002):
    rows = []
    for i in range(n_large):
        sid = f"L{i:03d}"
        price = 100.0
        for d in dates:
            price *= (1 + large_daily)
            rows.append({"datetime": d, "instrument": sid,
                         "close": price, "log_mktcap": 20.0})
    for i in range(n_small):
        sid = f"S{i:03d}"
        price = 100.0
        for d in dates:
            price *= (1 + small_daily)
            rows.append({"datetime": d, "instrument": sid,
                         "close": price, "log_mktcap": 10.0})
    return pd.DataFrame(rows)


def test_regime_large_cap_rotation_detected():
    dates = pd.date_range("2026-04-01", periods=15, freq="B")
    panel = _make_panel(dates, large_daily=0.005, small_daily=-0.002)
    out = detect_regime(panel, as_of=dates[-1], lookback_days=10,
                        threshold=0.01)
    assert out["regime"] == "large_cap_rotation"
    assert out["diff"] > 0.01


def test_regime_neutral_when_small_diff():
    dates = pd.date_range("2026-04-01", periods=15, freq="B")
    panel = _make_panel(dates, large_daily=0.0005, small_daily=0.0003)
    out = detect_regime(panel, as_of=dates[-1], lookback_days=10,
                        threshold=0.01)
    assert out["regime"] == "neutral"
