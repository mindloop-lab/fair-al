#!/usr/bin/env bash
# Revision (stage-2) script 20: continuous vs delayed drift control (S2-6 / R2-3).
#
# Both FAIR-AL-Weighted and FAIR-AL-Constrained are run with control enabled
#   from the start (0.0), after 50% of the budget, and after 75% of the budget.
# This is a delayed-ENABLEMENT heuristic, not an optimiser with an end-point guarantee.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-results_rev1/delayed_control}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
CLASSIFIERS="${CLASSIFIERS:-lr,svm}"
STRATEGIES="${STRATEGIES:-uncertainty,fairal_weighted,fairal_constrained}"
FRACS="${FRACS:-0.0,0.5,0.75}"
MODES="${MODES:-batch_prior,intra_batch}"
mkdir -p "$OUT_DIR"

for ds in ABIDE_A ABIDE_B; do
  extra=()
  if [[ "$ds" == "ABIDE_B" ]]; then extra=(--sensitive-index 46); fi
  for clf in ${CLASSIFIERS//,/ }; do
    for mode in ${MODES//,/ }; do
      mode_flags=()
      if [[ "$mode" == "intra_batch" ]]; then mode_flags=(--intra-batch); fi
      for frac in ${FRACS//,/ }; do
        "$PYTHON_BIN" -m baalexp.run_experiment \
          --data-dir "${DATA_DIR:-data}" --dataset "$ds" "${extra[@]}" \
          --backend sklearn --classifier "$clf" \
          --strategies "$STRATEGIES" --seeds "$SEEDS" \
          --initial-size 40 --budget 120 --batch-size 10 --k-candidates 200 \
          --control-start-frac "$frac" \
          "${mode_flags[@]}" \
          --save-history \
          --output "$OUT_DIR/${ds}_${clf}_${mode}_frac_${frac}.csv"
      done
    done
  done
done

echo "delayed-control runs written to $OUT_DIR"
echo "Report final Q-Bias, the budget-axis integral of Q-Bias, and whether the remaining"
echo "budget / group supply could still reach the target at the switch point."
