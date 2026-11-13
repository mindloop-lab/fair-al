# FAIR-AL

**FAIR-AL: A reproducible active-learning framework for auditing demographic query drift in
feature-based ASD screening** — manuscript CMPB-D-26-04362 (revision 1), *Computer Methods and
Programs in Biomedicine*.

Reference implementation: acquisition rules, the strict paired-comparison core, and the table
builders. No participant-level data, no checkpoints, no per-run result records.

| | |
|---|---|
| Repository | `https://github.com/mindloop-lab/fair-al` (ID 1456286167) |
| Verified release | [`cmpb-rev1-v1.0.1`](https://github.com/mindloop-lab/fair-al/releases/tag/cmpb-rev1-v1.0.1) |
| Code archive | `fair-al-cmpb-26-04362-cmpb-rev1-v1.0.1.zip` (214 558 B, `aa49cc0a…7d27`) |
| Results archive | `fair-al-results-evidence-cmpb-rev1-v1.0.1.zip` (460 989 B, `3f81bacd…3ce0`) |
| Package version | 0.8.0 (`pyproject.toml`) |

`main` carries documentation and comment cleanup on top of that tag; the code path is identical.
Cite the release, not a moving branch.

## Quick start (no restricted data)

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements-tested.txt
    export PYTHONPATH=src
    python tools/run_smoke_synthetic.py     # checks passed: 32/32
    python tests/run_tests.py               # passed=18 skipped=4 failed=0

The four skips are the data-dependent tests; a skip is not a pass.

## Data and results

| Artefact | Where | Licence |
|---|---|---|
| Code, scripts, tests, plan, documentation | this repository | Apache-2.0 |
| 14 aggregate result CSVs + schema/provenance | `fair-al-results-evidence-…-v1.0.1.zip` release asset | CC BY 4.0 |
| Per-run result records | not distributed | — |
| ABIDE features, labels, split/selection indices, per-subject predictions | not distributed (ABIDE data-use terms) | — |
| Figure 1 source | not distributed | — |

See `DATA_ACCESS.md` for what a user must supply themselves.

## Reproduction scope

| Path | Needs | Establishes |
|---|---|---|
| **A. Synthetic smoke** | nothing | the acquisition machinery runs, its invariants hold; no reported number |
| **B. Re-run an experiment** | authorised ABIDE-derived feature files | capability on your own seeds; not the published numbers |
| **C. Rebuild the tables** | the per-run result records | the reported contrasts recomputed from stored endpoints |

The fourteen aggregate CSVs contain paired endpoint values and aggregate per-query trajectories.
They support independent recomputation of the released paired-comparison statistics and the
process-drift integrals; they do **not** replace the per-run input tree that `scripts/36` and
`scripts/37` expect, and downloading the repository does not reproduce the paper's results.

`REPRODUCIBILITY.md` states exactly what each path can and cannot establish; `ENVIRONMENTS.md`
records the submission-era, revision-era and tested environments.

## Re-running an experiment

    export FAIRAL_DATA_DIR=/path/to/abide-derived
    python -m baalexp.run_experiment --data-dir $FAIRAL_DATA_DIR --dataset ABIDE_A \
        --classifier lr --strategies uncertainty,baal,baal_weighted,baal_hard \
        --seeds 0,1,2,3,4,5,6,7,8,9 --budget 120 --batch-size 10 --initial-size 40 \
        --k-candidates 200 --lambda-weight 0.5 --epsilon 0.03 \
        --output results/ABIDE_A_lr_core.csv --save-history

ABIDE-B additionally needs `--sensitive-index 46` (zero-based). Columns 46 and 47 are exact
complements and define a binary proxy group whose clinical meaning is unverified; ABIDE-B results
are proxy-group analyses, not sex-specific clinical fairness evidence.

## Notes

- `scripts/26_make_revision_tables.py` is the legacy statistics path; use `36`/`37`.
- The `--fal-cur-cluster-drop-sensitive` switch backs the separately reported D4 exploratory
  analysis (off for the main comparisons, on for D4). D4 is outside the declared 514 comparison
  plan; its intervals are nominal and its Holm check across eight outcomes is exploratory.
- `scripts/33_make_framework_figure.py` draws an early alternative schematic, not the manuscript's
  Figure 1.
- `plan/comparisons_manifest.csv` is frozen; `FAL_CUR_NOTES.md` explains its vocabulary without
  changing it.

## Licence

Code, scripts, tests, configuration, plan and documentation: **Apache-2.0** — full text in
`LICENSE`, copyright statement in `NOTICE` (copyright 2026 Zongqing Chen, Chongqing Normal
University, Chongqing, China). The fourteen aggregate CSVs and their documentation: **CC BY 4.0**,
stated in `LICENCE_DATA.md` inside that asset. Neither licence extends to ABIDE or to any
ABIDE-derived feature table.

## Citation

`CITATION.cff` carries title, authors, licence and version. Cite the article; no DOI is asserted
here because none has been minted.
