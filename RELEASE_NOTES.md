# Release notes — cmpb-rev1-v1.0.1 (release-integrity correction)

Reference code for the revision of *FAIR-AL: A reproducible active-learning framework for
auditing demographic query drift in feature-based ASD screening* (manuscript CMPB-D-26-04362,
revision 1).

This release supersedes `cmpb-rev1-v1`. The old tag, commit and assets are retained; its
release description receives an erratum and a link to this correction. No scientific
result, statistic, plan entry or source file changes: the difference is packaging and
licensing.

## Fixed since cmpb-rev1-v1

1. **`LICENSE` and `NOTICE` were absent from the repository and from the archive.** The
   documentation of `cmpb-rev1-v1` declared Apache-2.0 while the licence text was not shipped.
   Both files are now present in the Git tree and in the archive, and every file is covered by
   an explicit licence statement.
2. **Two files failed their own `SHA256SUMS` in a fresh clone** — `MANIFEST.csv` and
   `plan/comparisons_manifest.csv`. The publishing host normalised CRLF to LF at commit time
   while the sums were computed on the CRLF working tree, so the committed blobs differed from
   the hashed bytes. Both files are now written with LF throughout, `.gitattributes` pins the
   text and binary policy, and the sums were recomputed **after** that normalisation and
   verified in a fresh clone.
3. **Contradictory licence statements**: the README opened with "licence pending" while its
   own licence section said Apache-2.0, and these notes said no licence had been granted.
   Every statement now reflects the granted licences.
4. **"No result records" overstated the restriction.** Fourteen aggregate result CSVs are
   published as a separate release asset under CC BY 4.0. The README now separates what is
   published (code tree; aggregate CSVs) from what is not (per-run records; ABIDE-derived
   features, labels, indices, predictions; Figure 1 source).

## What this release is

- Acquisition rules (uncertainty, Pareto, weighted, constrained with entropy fallback), the
  strong composition controls, and the adapted FAL-CUR arm.
- The strict paired-comparison core and the table builders `scripts/36` and `scripts/37`.
- The declared 514-comparison plan as `plan/comparisons_manifest.csv`.
- A synthetic end-to-end example that runs with no restricted input
  (`tools/run_smoke_synthetic.py`).

## Assets

- `fair-al-cmpb-26-04362-cmpb-rev1-v1.0.1.zip` — this code tree (Apache-2.0).
- `fair-al-results-evidence-cmpb-rev1-v1.0.1.zip` — 14 aggregate result CSVs plus schema,
  provenance and licence documents (CC BY 4.0).

## Verification performed on these bytes

For this correction, the candidate commit was cloned and the actual archive was extracted
into new directories. Smoke and data-free tests used the previously tested Python 3.12.13
environment, with imports verified to resolve inside the extracted candidate. No new
dependency installation is claimed for this correction. Earlier clean-environment
installation and authorized-data/table-building checks are historical evidence, reported
separately in ENVIRONMENTS.md and the authors' acceptance records.

- `sha256sum -c SHA256SUMS` in the fresh clone: all entries verify, under both
  `core.autocrlf=input` and `core.autocrlf=true` checkouts
- `tools/run_smoke_synthetic.py` → `checks passed: 32/32`
- `tests/run_tests.py` without restricted data → `passed=18 skipped=4 failed=0`
- the archive extracted from this release reproduces the committed tree byte for byte

The results-evidence asset is checked separately: file set, per-file digests and per-record
content equality with the approved CSVs.

## What it does not contain

- No participant-level data: no feature tables, labels, split or selection indices, per-subject
  predictions.
- No complete per-run result tree. The released endpoint pairs and trajectories support
  independent statistical recomputation, but do not directly provide the full input layout
  required by `36`/`37`. `REPRODUCIBILITY.md` explains this distinction.
- No DeepASD adapters (upstream licence unverified) and not the FAL-CUR authors' script
  (`src/baalexp/fal_cur.py` is an adapted reimplementation).
- No figure source for Figure 1.

## Known limits

- `scripts/33_make_framework_figure.py` produces an early alternative of the framework
  schematic, not the figure in the manuscript.
- The D4 sensitivity switch is exploratory, off by default, and outside the declared plan.
- `scripts/26_make_revision_tables.py` is the legacy statistics path; use `36`/`37`.

## Citing

See `CITATION.cff`. Please cite the article; no DOI is asserted for this release because none
has been minted. If a DOI is minted later it will be recorded in an updated release note rather
than by moving this tag.
