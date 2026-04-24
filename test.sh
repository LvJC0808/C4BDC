#!/bin/bash
set -e
cd /app
export MCAP_MIN_LARGE=${MCAP_MIN_LARGE:-3}
export MCAP_CAND_K=${MCAP_CAND_K:-10}
export MCAP_LARGE_Q=${MCAP_LARGE_Q:-0.5}
python scripts/predict_lgb_only.py
