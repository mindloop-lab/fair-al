# Reproducibility statement

What each path in `README.md` can and cannot establish. Every claim below was executed on
2026-11-13 from a fresh extraction of the release archive, in a new Python 3.12.13 virtual
environment built from `requirements-tested.txt`. The acceptance log is
`PUBLICATION_ACCEPTANCE.csv` in the review bundle; the raw logs are kept by the authors.

## A. Synthetic smoke — PASS, no restricted input

    python tools/run_smoke_synthetic.py

Expected, and observed: `checks passed: 32/32`. The run generates a synthetic dataset,
executes five acquisition rules for one seed, and asserts evaluation-point count, one history
record per query event, Q-Bias within range, constrained fallback counters, cross-round
uniqueness of selections, index bounds, and that every selection lies inside the training
pool and outside the initial/validation/test sets.

Establishes: the acquisition machinery runs and its internal invariants hold.
Does **not** establish: any reported number, any comparison, any fairness conclusion.

## A'. Data-free test suite — PASS with four explicit skips

    python tests/run_tests.py

Observed: `passed=18 skipped=4 failed=0`. The four skips are the data-dependent tests; each
names the environment variable it needs (`FAIRAL_DATA_DIR`, `FAIRAL_FROZEN_DIR`). Skips are
reported as skips, never as passes.

## B. Experiment re-run — capability only

Needs authorised ABIDE-derived feature files. With them the same suite reports
`passed=22 skipped=0 failed=0`; the authors observed exactly that on their authorised copy.

Establishes: the pipeline runs end-to-end on the declared feature contract, and the loader
enforces the column contract for both cohorts.
Does **not** establish: the published numbers. Those came from seeds 0–9, budget 120,
initial 40, batch 10, k-candidates 200, λ = 0.5, ε = 0.03 on files the authors held at a
specific SHA-256; different files or seeds give different numbers by design.

## C. Table rebuild — PASS, needs the historical result records

    python scripts/36_build_final_tables.py --plan plan/comparisons_manifest.csv --out out/
    python scripts/37_build_support_tables.py --out out/

Observed with the authors' authorised result records: `declared = evaluated_ok = 514`,
`ids_match_manifest = true`, 5140 pair rows (5141 lines with the header), 98 families, 418
testable contrasts, 96 zero-variance contrasts; the rebuilt tables match the frozen revision
tables value for value (max |Δ| = 0.0 over 11 tables).

Establishes: the statistics in the paper follow mechanically from the stored per-comparison
endpoints plus the declared plan.
Does **not** establish: that the stored endpoints themselves are correct — that depends on
the runs that produced them, whose environment is recorded in `ENVIRONMENTS.md`.

The result records are not in this repository. The authors can supply them where the
redistribution terms allow, or a reviewer can be given access through the journal.

## What is deliberately not reproducible from this repository

- **Figure 1** is an edited vector graphic; `scripts/33` produces an early alternative, not
  the published figure.
- **Figure regeneration** (`scripts/28`) needs `matplotlib` and the stored per-seed curves.
- **D4 sensitivity analysis**: the switch and its documentation are here, and the archived
  per-seed evidence was checked against it; the 20 paired runs are not re-executed here
  because that analysis is not part of the declared comparison plan.
- **DeepASD backend**: the adapter modules are withheld (upstream licence unverified), so
  `--backend deepasd` will not run. Every reported number uses the sklearn backend.
- **Process-level trajectory integrals**: the 5140-row paired table does not contain the
  query-by-query trajectories; the trajectory points are a separate artefact.

## Environment

`ENVIRONMENTS.md` lists three recordings: the frozen submission runs (numpy 2.3.5 / pandas
3.0.3 / scipy 1.17.1 / scikit-learn 1.9.0), the revision runs (2.5.4 / 3.1.0 / 1.18.1 / 1.9.1)
and the tested set in `requirements-tested.txt` (the revision set plus matplotlib). `uv.lock`
is the submission-era resolution and is kept as a historical record, not as the recommended
install path.
