#!/usr/bin/env bash
# Formal sweep for the revision (seeds 0-9). Writes only to results_rev1/formal/.
# Sequential on purpose: one dataset/classifier at a time, logs per stage.
# Paths: run from the package root, or set FAIRAL_CODE_ROOT to it. PYTHON_BIN selects the
# interpreter (default: the active one on PATH).
set -euo pipefail
cd "${FAIRAL_CODE_ROOT:-$(dirname "$0")/..}"
export PYTHONPATH=src
export PYTHON_BIN="${PYTHON_BIN:-python}"
export SEEDS=0,1,2,3,4,5,6,7,8,9
LOG=results_rev1/formal
mkdir -p "$LOG"

echo "[1/4] lambda/tau grid"
OUT_DIR="$LOG/lambda_tau" bash scripts/17_lambda_tau_sensitivity.sh
echo "[2/4] sensitive-attribute removal"
OUT_DIR="$LOG/drop_sensitive" bash scripts/18_drop_sensitive_inputs.sh
echo "[3/4] batch-prior vs intra-batch"
OUT_DIR="$LOG/intra_batch" bash scripts/19_intra_batch_ablation.sh
echo "[4/4] continuous vs delayed control"
OUT_DIR="$LOG/delayed_control" bash scripts/20_delayed_control.sh
echo "FORMAL SWEEP DONE"
