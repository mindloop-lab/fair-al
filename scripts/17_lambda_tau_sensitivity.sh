#!/usr/bin/env bash
# Revision (stage-2) script 17: lambda / tau sensitivity grid.
#
# Purpose: answer R2-1 with evidence instead of asserting the argparse defaults.
# The default operating point (lambda=0.5, tau=0.03) is NEVER changed here; the grid
# only shows sensitivity. Every run records the explicit value it used.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-results_rev1/lambda_tau}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
CLASSIFIERS="${CLASSIFIERS:-lr,svm}"
DATASETS="${DATASETS:-ABIDE_A,ABIDE_B}"
LAMBDAS="${LAMBDAS:-0.2,0.35,0.5,0.65,0.8}"
TAUS="${TAUS:-0.01,0.02,0.03,0.05,0.08}"
mkdir -p "$OUT_DIR"

for ds in ${DATASETS//,/ }; do
  extra=()
  if [[ "$ds" == "ABIDE_B" ]]; then extra=(--sensitive-index 46); fi
  for clf in ${CLASSIFIERS//,/ }; do
    for lam in ${LAMBDAS//,/ }; do
      "$PYTHON_BIN" -m baalexp.run_experiment \
        --data-dir "${DATA_DIR:-data}" --dataset "$ds" "${extra[@]}" \
        --backend sklearn --classifier "$clf" \
        --strategies fairal_weighted,uncertainty \
        --seeds "$SEEDS" --initial-size 40 --budget 120 --batch-size 10 --k-candidates 200 \
        --lambda-weight "$lam" \
        --save-history \
        --output "$OUT_DIR/${ds}_${clf}_lambda_${lam}.csv"
    done
    for tau in ${TAUS//,/ }; do
      "$PYTHON_BIN" -m baalexp.run_experiment \
        --data-dir "${DATA_DIR:-data}" --dataset "$ds" "${extra[@]}" \
        --backend sklearn --classifier "$clf" \
        --strategies fairal_constrained,uncertainty \
        --seeds "$SEEDS" --initial-size 40 --budget 120 --batch-size 10 --k-candidates 200 \
        --epsilon "$tau" \
        --save-history \
        --output "$OUT_DIR/${ds}_${clf}_tau_${tau}.csv"
    done
  done
done

echo "lambda/tau grid written to $OUT_DIR"
echo "NOTE: infeasibility is recorded per round in the column 'constraint_dropped';"
echo "      report the trigger rate, do not silently drop those runs."
