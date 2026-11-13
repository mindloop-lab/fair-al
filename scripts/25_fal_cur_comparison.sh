#!/usr/bin/env bash
# Revision script 25 (second round, 2026-11-03): external fair-AL comparison — FAL-CUR.
#
# This is OUR implementation of the published FAL-CUR algorithm (FairKM marginal assignment,
# Eq. 4 cluster ranking, Eq. 7 score, Algorithm 1 round-robin selection), NOT the authors'
# released script. See src/baalexp/fal_cur.py and 02_FAL_CUR_FIDELITY_AUDIT.md for the
# documented choices D1-D6. The first-round implementation (fixed cluster penalty) is
# superseded: its results stay untouched in results_rev1/formal/fal_cur/.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-results_rev1/formal/fal_cur_v2}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
CLASSIFIERS="${CLASSIFIERS:-lr,svm}"
STRATEGIES="${STRATEGIES:-fal_cur,uncertainty,fairal_weighted,quota,stratified_uncertainty}"
mkdir -p "$OUT_DIR"

for ds in ABIDE_A ABIDE_B; do
  extra=()
  if [[ "$ds" == "ABIDE_B" ]]; then extra=(--sensitive-index 46); fi
  for clf in ${CLASSIFIERS//,/ }; do
    "$PYTHON_BIN" -m baalexp.run_experiment \
      --data-dir "${DATA_DIR:-data}" --dataset "$ds" "${extra[@]}" \
      --backend sklearn --classifier "$clf" \
      --strategies "$STRATEGIES" --seeds "$SEEDS" \
      --initial-size 40 --budget 120 --batch-size 10 --k-candidates 200 \
      --save-history \
      --output "$OUT_DIR/${ds}_${clf}_falcur.csv"
  done
done

echo "FAL-CUR comparison written to $OUT_DIR"
echo "Label results as 'FAL-CUR (reimpl.)', never as the authors' code."
