#!/usr/bin/env bash
# Revision script 34 (2026-11-03): FAL-CUR implementation sensitivity.
#
# Only the FAL-CUR arm is re-run; the comparison arms come from fal_cur_v2.
# Variants (none of them is the manuscript's main comparator, see
# 02_FAL_CUR_FIDELITY_AUDIT.md D1/D3/D4):
#   fal_cur_rawscale   no within-cluster normalisation (literal Eq. 7 scaling)
#   fal_cur_entclust   entropy appended to the clustering space (authors' script)
#   fal_cur_l05        lambda multiplier x0.25
#   fal_cur_l2         lambda multiplier x4
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-results_rev1/formal/fal_cur_sensitivity}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
VARIANTS="${VARIANTS:-fal_cur_rawscale,fal_cur_entclust,fal_cur_l05,fal_cur_l2}"
mkdir -p "$OUT_DIR"
for pair in "ABIDE_A lr" "ABIDE_B svm"; do
  set -- $pair; ds=$1; clf=$2
  extra=(); [[ "$ds" == "ABIDE_B" ]] && extra=(--sensitive-index 46)
  for v in ${VARIANTS//,/ }; do
    "$PYTHON_BIN" -m baalexp.run_experiment \
      --data-dir "${DATA_DIR:-data}" --dataset "$ds" "${extra[@]}" \
      --backend sklearn --classifier "$clf" --strategies "$v" --seeds "$SEEDS" \
      --initial-size 40 --budget 120 --batch-size 10 --k-candidates 200 \
      --output "$OUT_DIR/${ds}_${clf}_${v}.csv"
  done
done
echo "FAL-CUR sensitivity written to $OUT_DIR"
