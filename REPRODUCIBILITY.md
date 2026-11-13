# Reproducibility scope and verification provenance

This document separates capabilities, published evidence, and tests performed at different
stages. Downloading the code is not equivalent to reproducing every manuscript artifact.

## A. Synthetic workflow and data-free tests

    python tools/run_smoke_synthetic.py
    python tests/run_tests.py

The authors measured 32/32 smoke checks and 18 passed / 4 skipped / 0 failed from the
extracted v1.0.1 candidate using the previously tested Python 3.12.13 environment. Module
resolution was checked inside that extraction. This correction did not perform a new
dependency installation. The four skips require authorized data and frozen results;
they are not evidence of a passing data-dependent regression.

The synthetic workflow tests execution and selection containment on generated data.
It does not reproduce a published biomedical result or establish a fairness conclusion.

## B. Experiments with authorized feature inputs

Experiment reruns require the documented ABIDE-derived feature contract and recorded
protocol. The historical acceptance records separately report 22 passed / 0 skipped with
authorized feature files. That check was not repeated in this packaging correction.
The exact original per-run invocation arguments were not archived; defaults and runner
records identify historical parameter settings with the limitations stated in the paper.

## C. Existing table-building entry points

    python scripts/36_build_final_tables.py --plan plan/comparisons_manifest.csv --out out/
    python scripts/37_build_support_tables.py --out out/

These commands expect the full historical/revision input tree, which is not distributed
in this code repository. Historical acceptance records report 514 comparisons, 5140 paired
rows, 98 families, 418 testable contrasts and 96 zero-variance contrasts, with 11 rebuilt
tables matching the frozen revision results. Those full-input checks are historical
evidence, not new training or table-building runs in this correction.

## D. Independent recomputation from the released evidence

The results-evidence release asset includes comparison_endpoint_pairs.csv and the declared
comparison plan. The endpoint pairs permit independent recomputation of the released
paired means, differences, nominal confidence intervals, effect sizes and adjusted p-values,
including the stated handling of zero-variance contrasts and comparison families.

The asset also includes process_trajectory_points.csv. These aggregate per-query points
permit recomputation of the released process-drift integrals and summaries. D4_pairs.csv
similarly supports verification of the separately released exploratory D4 summaries.

These files cannot be passed directly to the existing 36/37 commands as a substitute for
their original input tree. Their availability does not imply that every table, plot,
training curve or experiment can be rebuilt from the evidence ZIP alone. Recomputing
statistics from stored values does not independently verify the original training runs.

## E. Other limits

- The current Figure 1 source is not distributed; scripts/33 produces an earlier alternative.
- Figure regeneration via scripts/28 requires additional stored result inputs.
- The main FAL-CUR comparisons have the D4 flag off; the separate exploratory D4 runs have
  it on. D4 is outside the 514-comparison manifest. Raw p-values and nominal intervals must
  be distinguished from any exploratory eight-outcome Holm calculation.
- The DeepASD adapters are withheld because upstream redistribution permission was not
  verified. The separate DeepASD reference analyses are outside this core package. The
  FAIR-AL acquisition experiments represented by this package use the sklearn backend.
- The historical lock and revision tested requirements identify different environments;
  ENVIRONMENTS.md explains them. requirements-tested.txt is the recommended revision setup.
