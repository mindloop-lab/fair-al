#!/usr/bin/env bash
# Revision (stage-2) script 18: sensitive-attribute removal ablation (S2-3).
#
# The protected-attribute columns are removed from the MODEL INPUT only; the acquisition
# strategies still receive the attribute and the splits / initial labeled set are unchanged
# (verified by tests/test_rev1.py::test_drop_sensitive_keeps_split_and_attribute).
#
# ABIDE-A: SEX_1 / SEX_2 (complementary one-hot pair).
# ABIDE-B: PHENO feature 46 and its complement 47.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-results_rev1/drop_sensitive}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
CLASSIFIERS="${CLASSIFIERS:-lr,svm}"
STRATEGIES="${STRATEGIES:-uncertainty,quota,stratified_uncertainty,fairal_pareto,fairal_weighted,fairal_constrained}"
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
      --drop-sensitive-inputs \
      --output "$OUT_DIR/${ds}_${clf}_dropsens.csv"
  done
done

echo "sensitive-attribute removal runs written to $OUT_DIR"
echo "Report: Q-Bias and balanced accuracy vs the same arms WITHOUT the ablation."
echo "Do not pre-commit to the claim that results are unchanged."
