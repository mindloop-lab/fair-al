# Changelog

All entries concern packaging and documentation unless stated. No reported number, seed,
split, budget or statistical family changed in any of these steps.

## 2026-11-13 — public release candidate

- Documentation re-cut for public reading: exact manuscript title, the three reproduction
  paths separated, environment entry point made explicit (`requirements-tested.txt` first,
  `uv.lock` marked historical), `scripts/26` marked legacy.
- D4 wording corrected everywhere it survived from the reviewer pack: the switch neutralises
  the group-encoding columns in the distance computations used by the FAL-CUR rule — the fair
  clustering **and** the within-cluster representativeness term — with the predictor input,
  group penalty, splits, initial set, budget, batch size and seeds unchanged.
- Figure provenance stated: `scripts/33` is an early alternative schematic, not the figure in
  the manuscript.
- Added `CITATION.cff`, `REPRODUCIBILITY.md`, `RELEASE_IDENTITY.json`, `.gitignore`.
- Removed from the public tree: `LICENSE_PENDING.md` (licence still undecided; the public
  tree therefore ships **no** licence and says so), `CHANGES_V4.md` (superseded by this file).
- No source file changed.

## 2026-11-12 — reviewer candidate v4

- `scripts/36` is now reachable by the documented command: the declared plan travels in the
  archive as `plan/comparisons_manifest.csv` and is passed with `--plan`.
- Removed machine-specific absolute paths from `scripts/21`, `22`, `26`, `28`; they resolve
  `FAIRAL_CODE_ROOT` / `FAIRAL_PROJECT_ROOT` / `FAIRAL_FROZEN_DIR` and accept `--out`.
- `scripts/39` (packager) and other internal-only tools no longer ship.
- `ENVIRONMENTS.md` corrected: the submission-era set is numpy 2.3.5 / pandas 3.0.3 /
  scipy 1.17.1 / scikit-learn 1.9.0; the revision set is 2.5.4 / 3.1.0 / 1.18.1 / 1.9.1. An
  earlier version of the file labelled 2.5.4 as historical, which was wrong.
- D4 switch documentation corrected in the CLI help and module comments (the intervention
  also removes the columns from the representativeness term).
- Figure scripts: 10 pt base, 9 pt ticks, 8.5 pt legend, curve legends below the axes, so the
  figures stay at or above 8 pt at the insertion widths used in the manuscript.
- `requirements-tested.txt` gained `matplotlib==3.11.2`.

## 2026-11-10 — reviewer candidate v3

- First reviewer-facing archive: code, tests, tools, declared plan, environment notes.
- Withheld the DeepASD adapter modules and the FAL-CUR authors' script (licences unverified);
  `fal_cur.py` labelled an adapted reimplementation.
