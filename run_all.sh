#!/usr/bin/env bash
# Rebuild everything from a clean state: new venv, pinned install, all scripts, report.
# Deletes outputs/ and derived data so nothing stale can be reused.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf outputs data/wisconsin_original_clean.csv data/feature_target_correlation.csv venv
python3 -m venv venv
. venv/bin/activate
pip install -q -r requirements-lock.txt
for s in 01_data_eda 02_model_training 03_explainability 04_robustness_testing \
         05_threshold_analysis 06_temporal_validation 07_validation_audit; do
  echo "== $s"; python src/$s.py > /dev/null
done
if command -v node >/dev/null; then
  [ -d node_modules ] || npm install --silent
  node src/build_report.js
else
  echo "node not found: skipping Word report"
fi
echo "Done."
