#!/bin/bash
set -e
cd /app/code/src
python pipeline.py predict \
  --data_path /app/data \
  --model_dir /app/model \
  --temp_dir /app/temp \
  --output_path /app/output/result.csv
