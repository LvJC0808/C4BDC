#!/bin/bash
set -e
mkdir -p /app/model /app/output /app/temp
python -c "import lightgbm, numpy, pandas, scipy, sklearn, pyarrow, talib; print('deps OK')"
