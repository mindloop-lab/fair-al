# Third-party notices

## Withheld
`src/baalexp/deepasd_*.py` - adapters around the DeepASD authors' code; the upstream
licence could not be verified for redistribution.  Only needed for `--backend deepasd`;
every manuscript result uses the sklearn backend.  Obtain DeepASD from its authors and
supply the adapter yourself.

`src/baalexp/fal_cur_legacy.py` is retained because `strategies.py` imports it at module
scope.  It is marked SUPERSEDED, and no reported number uses it.

## Not redistributed
The FAL-CUR authors' released script (no verified licence, and it was not used to produce
any number here).  `src/baalexp/fal_cur.py` is an adapted reimplementation of the published
algorithm within this protocol and is labelled "FAL-CUR (adapted reimpl.)" everywhere.

## Dependencies
numpy, pandas, scipy, scikit-learn, tqdm (optionally statsmodels).  `uv.lock` and
`requirements-tested.txt` pin two different resolutions - see `ENVIRONMENTS.md`.
