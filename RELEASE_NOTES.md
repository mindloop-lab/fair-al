# Release notes — cmpb-rev1-v1 (2026-11-13)

Reference code for the revision of *FAIR-AL: A reproducible active-learning framework for
auditing demographic query drift in feature-based ASD screening* (CMPB-D-26-04362).

## What this release is

- Acquisition rules (uncertainty, Pareto, weighted, constrained with entropy fallback), the
  strong composition controls, and the adapted FAL-CUR arm.
- The strict paired-comparison core and the table builders `scripts/36` and `scripts/37`.
- The declared 514-comparison plan as `plan/comparisons_manifest.csv`.
- A synthetic end-to-end example that runs with no restricted input (`tools/run_smoke_synthetic.py`).

## Verification performed on these bytes

Unpacked from the release archive into a new directory, new Python 3.12.13 virtual
environment, `pip install -r requirements-tested.txt` (exit 0):

- `tools/run_smoke_synthetic.py` → `checks passed: 32/32`
- `tests/run_tests.py` without data → `passed=18 skipped=4 failed=0`
- `tests/run_tests.py` with the authors' authorised feature files → `passed=22 skipped=0 failed=0`
- `scripts/36` → 514 declared, 514 evaluated, 5140 pair rows, 98 families, 418 testable,
  96 zero-variance
- `scripts/37` → support tables rebuilt
- 11 rebuilt tables match the frozen revision tables: max |Δ| = 0.0
- missing-plan and missing-source probes both exit non-zero without writing a partial table

## What it does not contain

- No participant-level data: no feature tables, labels, indices or per-subject predictions.
- No result records. Rebuilding the reported tables needs the per-run records held by the
  authors; `REPRODUCIBILITY.md` explains the dependency.
- No DeepASD adapters (upstream licence unverified) and not the FAL-CUR authors' script
  (`src/baalexp/fal_cur.py` is an adapted reimplementation).
- No licence. No licence has been granted, so this tree may not be redistributed; a licence
  will be added in a later release if and when the rights holder decides.

## Known limits

- `scripts/33_make_framework_figure.py` produces an early alternative of the framework
  schematic, not the figure in the manuscript.
- The D4 sensitivity switch is exploratory, off by default, and outside the declared plan.
- `scripts/26_make_revision_tables.py` is the legacy statistics path; use `36`/`37`.

## Citing

See `CITATION.cff`. Please cite the article; the DOI is intentionally unset in this candidate
rather than invented.
