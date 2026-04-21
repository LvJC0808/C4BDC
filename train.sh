#!/bin/bash
set -e
cd /app/code/src
python pipeline.py train \
  --data_path /app/data \
  --model_dir /app/model \
  --temp_dir /app/temp
