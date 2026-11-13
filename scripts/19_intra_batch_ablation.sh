#!/usr/bin/env bash
# Revision (stage-2) script 19: intra-batch update ablation (S2-4).
#
# Historical FAIR-AL scores every candidate of a batch against the PRE-batch cumulative
# queried distribution. The quota / stratified-uncertainty baselines update it after every
# pick. This ablation adds the sequential (intra-batch) variant for FAIR-AL. Historical
# results are NOT overwritten or reinterpreted; both versions are reported side by side.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-results_rev1/intra_batch}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
CLASSIFIERS="${CLASSIFIERS:-lr,svm}"
STRATEGIES="${STRATEGIES:-uncertainty,quota,stratified_uncertainty,fairal_pareto,fairal_weighted,fairal_constrained}"
mkdir -p "$OUT_DIR"

for ds in ABIDE_A ABIDE_B; do
  extra=()
  if [[ "$ds" == "ABIDE_B" ]]; then extra=(--sensitive-index 46); fi
  for clf in ${CLASSIFIERS//,/ }; do
    for mode in batch_prior intra_batch; do
      flags=()
      if [[ "$mode" == "intra_batch" ]]; then flags=(--intra-batch); fi
      "$PYTHON_BIN" -m baalexp.run_experiment \
        --data-dir "${DATA_DIR:-data}" --dataset "$ds" "${extra[@]}" \
        --backend sklearn --classifier "$clf" \
        --strategies "$STRATEGIES" --seeds "$SEEDS" \
        --initial-size 40 --budget 120 --batch-size 10 --k-candidates 200 \
        "${flags[@]}" \
        --output "$OUT_DIR/${ds}_${clf}_${mode}.csv"
    done
  done
done

echo "intra-batch ablation written to $OUT_DIR"
