#!/bin/bash
# Switch the "active" model set that pipeline.py predict reads from.
# Usage:  ./scripts/activate_model.sh {5090|4060}
set -e
cd "$(dirname "$0")/.."

if [[ -z "${1:-}" ]]; then
  echo "usage: $0 {5090|4060}"
  exit 1
fi
tag="$1"
src="model/$tag"
[[ -d "$src" ]] || { echo "ERROR: archive not found: $src"; exit 1; }

# Remove active copies
rm -rf model/lgb model/master model/mixer model/ensemble_config.json

# Shallow copy (hard-link to save disk)
cp -al "$src/lgb"                 model/lgb
cp -al "$src/master"              model/master
cp -al "$src/mixer"               model/mixer
cp    "$src/ensemble_config.json" model/ensemble_config.json

echo "[activate] active model set: $tag"
