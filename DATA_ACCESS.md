# Data access and column contract

No participant-level data is in this archive: no feature tables, labels, index lists or
per-subject predictions.

## Cohort 1
`ABIDE_A/processed_standard_data.csv` - one row per subject, numeric features, last column
the diagnosis code `1`/`2` (mapped to `0`/`1` by the loader).  Group membership is a
complementary one-hot pair (`SEX_1`, `SEX_2`), detected by column name; both columns are
recorded as protected-attribute inputs and both are removed by `--drop-sensitive-inputs`.
`ABIDE_A/modal_feat_dict.npy` - pickled `{modality: [feature names]}`.

## Cohort 2
`ABIDE_B/RFE_512_processed_standard_data.npz` - `data` (n x 1584), `label` (1/2).  There is
no `sex`/`gender` key: the loader refuses to guess and requires `--sensitive-index 46`,
whose complement is column 47. These are zero-based column indices. Column 46 defines a
binary proxy group with unverified clinical meaning: the identification rests on the
complementary one-hot relation between columns 46 and 47, not on a data dictionary entry,
and the manuscript treats ABIDE-B as a secondary proxy-group analysis accordingly.

## Obtaining them
Both cohorts derive from ABIDE under its data-use terms; the derived tables cannot be
redistributed by the authors.  Rebuild them with the preprocessing described in the
manuscript data statement and record the sha256 of each file you obtain.

## Prohibited
No re-identification attempts, no joins with public record-level metadata, no
redistribution outside the terms you obtained the data under.
