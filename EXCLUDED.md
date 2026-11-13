# Exclusions

| withheld | reason |
|---|---|
| src/baalexp/deepasd_style.py | adapts DeepASD authors' code; licence unverified |
| src/baalexp/deepasd_original_adapter.py | same |
| src/baalexp/deepasd_static_sex_bias.py | same |
| src/baalexp/deepasd_uncertainty_sex_analysis.py | same |
| src/baalexp/enhanced_artifacts copy.py | duplicate working copy, not imported |
| data, results, frozen submission tree, per-seed predictions | restricted / not redistributable |
| internal review material, manuscript PDFs, virtualenvs, caches | not part of a code review |

`pyproject.toml` in this package no longer declares the `deepasd-*` console entry points
that pointed at the withheld modules.
