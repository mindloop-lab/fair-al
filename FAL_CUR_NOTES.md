# FAL-CUR adaptation and frozen-plan documentation errata

The 514-row comparison manifest is a frozen revision analysis record. This note corrects
documentation surrounding it; it changes no comparison, statistical family, seed, result,
or acquisition implementation.

## Reference and equation mapping

The relevant publication is Fajri, R. M., Saxena, A., Pei, Y., and Pechenizkiy, M. (2024),
*FAL-CUR: Fair Active Learning using Uncertainty and Representativeness on Fair Clustering*,
Expert Systems with Applications 242, 122842.
https://doi.org/10.1016/j.eswa.2023.122842

The publisher's version of record is also available from the authors' university:
https://pure.tue.nl/ws/portalfiles/portal/327474055/1-s2.0-S0957417423033444-main.pdf

Some frozen rationale strings use outdated equation labels. The correct mapping in that
version of record is:

| Component | Published location |
|---|---|
| Fair-clustering objective and nonsensitive-coordinate distance | Eq. (1) |
| Cluster-specific fairness deviation and fairness ranking | Eq. (3), Section 4.1 |
| An intermediate uncertainty-plus-distance selection objective | Eq. (4) |
| Predictive entropy | Eq. (5) |
| Final acquisition score | Eq. (6) |
| Classification accuracy metric | Eq. (7), Section 5.3 |
| Round-robin acquisition procedure | Algorithm 1 |

Thus the legacy phrases “Eq. 4 cluster ranking” and “Eq. 7 score” must not be read as
accurate references to the journal version. The frozen strings are retained for traceability;
this table provides the correction. They do not refer to equation numbers in the FAIR-AL
manuscript.

## Adapted implementation and D1-D6

The distributed implementation is an **adapted reimplementation**, not the original authors'
released script and not evidence that the original method would produce the same results.
The upstream script is not redistributed. The detailed implementation choices are documented
in src/baalexp/fal_cur.py. The labels used there mean:

| Label | Choice or boundary in this repository |
|---|---|
| D1 | The fair-clustering penalty uses a FairKM-inspired heuristic with a fixed multiplier; the project tests multipliers 0.25, 1 and 4. These multipliers are project choices, not reported FAL-CUR hyperparameters. |
| D2 | The number of clusters is set to batch size, with reclustering in the shared active-learning protocol. |
| D3 | The acquisition combines within-cluster min-max-normalized representativeness and entropy using the documented sign convention. A raw-scale sensitivity variant is supplied. |
| D4 | Main runs use model features, including group encodings, for distance computations. The separate D4 switch neutralizes those encodings in both clustering and within-cluster representativeness distances, while preserving predictor inputs and the group penalty. |
| D5 | Singleton clusters are retained; documented entropy filling applies if cluster selections are exhausted before the batch is full. |
| D6 | Acquisition uses unlabeled features, predicted probabilities and the group attribute; labels and test information are not acquisition inputs. |

The FAIR-AL weighted-acquisition parameter lambda=0.5 is distinct from the clustering
penalty discussed in D1. The main FAL-CUR comparisons have the D4 switch off; its exploratory
sensitivity runs have the switch on.

The manuscript's limitations and sensitivity evidence apply to this adapted implementation.
They must not be interpreted as a faithful reproduction claim or as conclusions about
the original FAL-CUR method in all settings.

## Analysis status and historical parameter evidence

- RETROSPECTIVE_REVISION_ANALYSIS means an analysis defined during revision, not preregistered.
- The frozen 514-row manifest includes **32** contrasts labeled
  POST_HOC_AFTER_SEEING_S24_RESULTS. They remain in their recorded families and are not
  converted into preregistered analyses by inclusion in the manifest.
- D4 is a separate exploratory sensitivity analysis outside those 514 comparisons.
- DESCRIPTIVE rows are reported without a multiplicity-adjusted inferential test.
- Archived runner defaults and scripts document FAIR-AL lambda=0.5 and tau=0.03.
  The complete historical per-run argument vectors were not retained, so defaults alone
  do not prove the absence of every historical override or every possible selection step.

The internal 02_FAL_CUR_FIDELITY_AUDIT.md is not part of the public distribution.
The public source documentation and this note state the adaptation choices needed to
interpret the released comparisons; access to that internal filename is not assumed.

