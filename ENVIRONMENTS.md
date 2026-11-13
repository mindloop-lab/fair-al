# Environments

Three recordings on purpose: the frozen submission runs, the revision runs, and the
environment this package was tested in.

## Frozen submission runs (the numbers in the submitted manuscript)

Taken from the `.env.json` written next to every result file in `results_as_submitted/`
(40 files, all identical):

    python 3.12.13   numpy 2.3.5   pandas 3.0.3   scipy 1.17.1   scikit-learn 1.9.0

Note: this is the set `uv.lock` resolves to, and it is **not** the set the revision runs
used.  An earlier version of this file labelled 2.5.4 as the historical environment; that
was wrong - 2.5.4 is the revision-run set below.

## Revision runs (the added analyses, 156 result files)

    python 3.12.13   numpy 2.5.4   pandas 3.1.0   scipy 1.18.1   scikit-learn 1.9.1

## Candidate tested environment (this package)

`requirements-tested.txt` pins the revision set plus matplotlib for the figure scripts.
On 2026-11-12 the package was unpacked into a fresh directory, a new `python3.12 -m venv`
was created (Python 3.12.13) and `pip install -r requirements-tested.txt` resolved every
pin; the README quick start then reported `checks passed: 32/32` for the synthetic smoke
and `passed=18 skipped=4 failed=0` without data (`passed=22 skipped=0 failed=0` with the
authorised data directories set).  The table builders `scripts/36` and `scripts/37`
reproduce the frozen contrast, family, seven-tier, process-drift and Table-5 tables
bit-for-bit (max |delta| = 0.0) in that environment.

## Why `uv.lock` is not the tested set

`uv.lock` was resolved for the submission-era environment (numpy 2.3.5, pandas 3.0.3,
scikit-learn 1.9.0, scipy 1.17.1).  It is kept because the submitted runs were launched
from it, and it is stated here as the submission-era record rather than as a replica of
the revision runs.  If you need to reproduce the submitted numbers, prefer that
resolution; if you need to re-run the revision analyses, use `requirements-tested.txt`.

## Not installed here

`torch` (only needed for `--backend deepasd`, whose adapter modules are not
redistributed) and `statsmodels` (optional; the multiple-comparison cross-check falls back
to a printed notice when it is absent).
