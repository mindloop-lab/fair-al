# Notes on the frozen comparison plan

`plan/comparisons_manifest.csv` is the declared 514-comparison plan. It is frozen: `analysis_id`,
comparison direction, family assignment, seeds and source relations cannot be edited without
changing the analysis. This file explains the vocabulary in its `rationale` column so a reader
does not have to guess, and it deliberately adds no definition of its own.

## Equation and algorithm references

Some rationales describe the FAL-CUR comparison as "FairKM marginal clustering, Eq. 4 cluster
ranking, Eq. 7 score, Algorithm 1 round-robin". Those numbers refer to **the published FAL-CUR
paper**, not to this manuscript and not to files in this repository:

> Y. Yan, S. Liu, et al., *Fairness-aware active learning*, and the FAL-CUR algorithm as published
> there; the marginal-clustering step (FairKM) and its cluster-ranking equation, the scoring
> equation, and the round-robin selection algorithm are used as described in that paper.

`src/baalexp/fal_cur.py` is an **adapted reimplementation** written for this repository. It is not
the FAL-CUR authors' script, it was not redistributed to us, and the comparison numbers should be
read as "this reimplementation versus our arms", not as a reproduction of the original
implementation's published values.

## "02_FAL_CUR_FIDELITY_AUDIT.md" and the D1-D6 choices

That file is an internal audit record and is **not distributed**. It listed six implementation
decisions (D1-D6) taken when reimplementing FAL-CUR: how cluster assignment is initialised, how the
marginal-clustering objective is optimised, how clusters are ranked against the target
distribution, how the score is combined with the uncertainty term, how ties are broken during
round-robin selection, and how a round ends when a cluster is exhausted. The rationale strings
name them so that the audit trail stays traceable; the choices themselves are visible in the code.

## Terms used in the rationales

| Term | Meaning |
|---|---|
| batch-prior (historical) | the pre-revision acquisition rule that scores a whole batch against the prior composition |
| intra-batch | the revision variant that re-scores within the batch at each query |
| pre-set default | a value fixed before the runs and never re-selected from their results (0.5 for lambda, 0.03 for tau) |
| RETROSPECTIVE_REVISION_ANALYSIS | the analysis was defined during the revision, not preregistered |
| post hoc | added after the runs were inspected; reported as post hoc and not in the 514 family |
| descriptive | reported without a multiplicity-adjusted test |

Nothing here changes the plan. If a future revision needs different comparisons, it needs a new
manifest, not an edit to this one.
