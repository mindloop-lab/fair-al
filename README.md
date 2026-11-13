# FAIR-AL — reference code for

**FAIR-AL: A reproducible active-learning framework for auditing demographic query drift in
feature-based ASD screening** (manuscript CMPB-D-26-04362, revision 1).

This repository contains the acquisition rules, the strict paired-comparison statistics and
the table builders behind the revision. It contains **no participant-level data** and **no
result records**: the derived ABIDE feature files cannot be redistributed by the authors
(see `DATA_ACCESS.md`).

> **Licence: pending an author decision.** No open-source licence has been granted yet, so
> this tree is not licensed for reuse or redistribution. See "Licence status" below.

## What this code does

FAIR-AL audits *demographic query drift*: the change in the protected- or proxy-group
composition of the cohort an active-learning rule chooses to annotate, measured by a
Total-Variation Q-Bias between the cumulative queried distribution and a target distribution.
Four acquisition families are implemented — uncertainty, Pareto (FAIR-AL-Pareto), weighted
scalarisation (FAIR-AL-Weighted) and constrained with entropy fallback (FAIR-AL-Constrained) —
plus strong composition controls (quota, stratified uncertainty) and the adapted FAL-CUR arm.

The paper's claims are about the *acquisition process*, not about diagnostic sensitivity of a
downstream model. Nothing here establishes clinical validity.

## Layout

    src/baalexp/      acquisition rules, backends, metrics, strict pairing core, statistics
    tests/            plain-assert tests; run_tests.py reports passed / skipped / failed
    scripts/          experiment drivers and table builders
    tools/            synthetic-data generator, data-free smoke runner
    configs/          run configuration records
    plan/             the declared 514-comparison plan (comparisons_manifest.csv)

    pyproject.toml            package metadata (version 0.8.0)
    requirements-tested.txt   the exact set this tree was verified with (start here)
    ENVIRONMENTS.md           submission-era, revision-era and tested environments
    uv.lock                   submission-era lock — historical, NOT the tested set

## Three different things people mean by "reproduce"

| Path | Needs | Command | What you get |
|---|---|---|---|
| **A. Synthetic smoke** | nothing | `python tools/run_smoke_synthetic.py` | 32/32 checks on generated data: five rules, one seed, index bounds, cross-round uniqueness, fallback counters, Q-Bias range. Reproduces no published number. |
| **B. Re-run an experiment** | authorised ABIDE-derived feature files (`DATA_ACCESS.md`) | `python -m baalexp.run_experiment ...` (below) | New result CSVs for your own seeds. Restores capability, not the published numbers: the published runs used seeds 0–9, budget 120, initial 40, batch 10. |
| **C. Rebuild the paper tables** | the historical result records (`results_as_submitted/`, `results_rev1/formal/`) | `python scripts/36_build_final_tables.py --plan plan/comparisons_manifest.csv --out out/` then `python scripts/37_build_support_tables.py --out out/` | The contrast, family-ledger and support tables, recomputed from the stored per-comparison endpoints. |

Downloading this repository does **not** reproduce the paper's full results. Path C needs the
result records, which are not in this repository; path B needs the feature files, which the
authors cannot redistribute. `REPRODUCIBILITY.md` states exactly what each path can and
cannot establish.

## Quick start (no restricted data)

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements-tested.txt
    export PYTHONPATH=src
    python tools/run_smoke_synthetic.py     # expects: checks passed: 32/32
    python tests/run_tests.py               # expects: passed=18 skipped=4 failed=0

The four skips are the data-dependent tests; they skip with an explicit reason and are not
evidence of a passing regression.

## Re-running an experiment

    export FAIRAL_DATA_DIR=/path/to/abide-derived
    python -m baalexp.run_experiment --data-dir $FAIRAL_DATA_DIR --dataset ABIDE_A \
        --classifier lr --strategies uncertainty,baal,baal_weighted,baal_hard \
        --seeds 0,1,2,3,4,5,6,7,8,9 --budget 120 --batch-size 10 --initial-size 40 \
        --k-candidates 200 --lambda-weight 0.5 --epsilon 0.03 \
        --output results/ABIDE_A_lr_core.csv --save-history

ABIDE-B additionally needs `--sensitive-index 46` (zero-based). Column 46 and column 47 are
exact complements; the pair defines a binary proxy group whose clinical meaning has **not**
been verified against a data dictionary. See `DATA_ACCESS.md`.

## Building the tables (needs the authorised result records)

    export FAIRAL_CODE_ROOT=/path/to/this/tree-with-results   # must hold results_rev1/
    export FAIRAL_PROJECT_ROOT=/path/to/project               # holds results_as_submitted/
    export PYTHONPATH=$FAIRAL_CODE_ROOT/src
    python scripts/36_build_final_tables.py --plan plan/comparisons_manifest.csv --out out/
    python scripts/37_build_support_tables.py --out out/

`--plan` is required: `36` otherwise looks for
`<code root>/results_rev1/stat/comparisons_manifest.csv`. Without the result records, both
scripts exit non-zero with `comparison plan not found` / `source not found` rather than
writing a partial table.

`36` writes `comparison_endpoint_pairs.csv` (514 × 10 = 5140 rows), `05_CONTRASTS_FINAL.csv`,
`06_FAMILY_LEDGER.csv`, `guard_probes.csv` and `verification.json`. A complete run reports
`declared = evaluated_ok = 514`, `98` statistical families, `418` testable contrasts and `96`
zero-variance contrasts; `wc -l out/comparison_endpoint_pairs.csv` gives 5141 lines including
the header.

`scripts/26_make_revision_tables.py` is the **legacy** statistics path from an earlier
revision round. It is kept for provenance only: its results differ from `36` wherever the
earlier loader is used, and it is not an authoritative entry point. Use `36`/`37`.

## Sensitivity switch (off by default, exploratory only)

    --fal-cur-cluster-drop-sensitive

With the flag on, the two group-encoding columns are replaced by their own column mean before
the distance computations used by the FAL-CUR acquisition rule. Both the fair-clustering
assignment **and** the within-cluster representativeness term use that matrix, so the group
encoding is neutralised in both; the predictor input, the group penalty, the train/test
splits, the initial labelled set, the budget, the batch size and the seeds are unchanged. The
flag is refused together with `--drop-sensitive-inputs`, which removes the columns from the
predictor as well — a different intervention.

The flag is off in every reported number. It backs an exploratory sensitivity analysis that
is **not** part of the declared 514-comparison plan and carries no multiplicity adjustment.
The interval for the one setting it moves is reported as nominal and per-comparison.

## Figure 1

`scripts/33_make_framework_figure.py` generates an **early alternative** of the framework
schematic, not the figure in the current manuscript. The manuscript figure is an edited
vector graphic; `scripts/33` is shipped only so the underlying layout is documented. Do not
use it to claim reproduction of Figure 1.

## Licence

The code, scripts and documentation in this repository are licensed under the **Apache
License 2.0** (see `LICENSE`); copyright 2026 Zongqing Chen, Chongqing Normal University, Chongqing, China, recorded in `NOTICE`.

The fourteen aggregate result CSVs and their documentation, published as a release asset, are
licensed under **CC BY 4.0** (see `LICENCE_DATA.md` in that directory). That licence covers the
aggregate files only: it does not extend to ABIDE or to any ABIDE-derived feature table, which
remain subject to the ABIDE data-use terms.

## Citation

`CITATION.cff` carries the title, author list and version. Please cite the manuscript once it
is published; the DOI is intentionally left unset rather than invented.
