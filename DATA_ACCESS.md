# Data access, provenance and input contract

This distribution contains no participant-level feature tables, diagnosis labels,
participant identifiers, split/selection indices or predictions. The separate evidence
asset contains original aggregate analysis outputs, not participant records.

## Source access and licensing

ABIDE source data are available through the original ABIDE/INDI access routes:
https://fcon_1000.projects.nitrc.org/indi/abide/ . The ABIDE I Usage Agreement describes
non-commercial research use, registration, attribution and funding acknowledgements,
and links to CC BY-NC-SA 3.0. Follow the terms for the actual source release and access
route. ABIDE II is described separately on its official page; the local file names
below do not establish which official release or sites supplied a participant.

The authors do not redistribute the processed participant-level files in this project.
This is a distribution decision, not a claim that ABIDE universally prohibits all
redistribution. Apache-2.0 covers this software. CC BY 4.0 covers the authors' original
aggregate analysis outputs in the evidence asset. Neither relicenses source ABIDE data
or conveys rights to third-party processed feature files.

## Supported processed artifacts

`ABIDE_A` and `ABIDE_B` are local processed-artifact names. They must not be assumed to
mean ABIDE I and ABIDE II or to denote independent source participants.

### ABIDE_A

- `processed_standard_data.csv`: one row per participant, numeric features, diagnosis
  code in the final column.
- `modal_feat_dict.npy`: a serialized feature-name/modality dictionary. Load only files
  obtained from a trusted source because the historical loader enables pickle.
- The audited named schema has `SEX_1`/`SEX_2` at zero-based columns 46/47. Group 0 is
  SEX_1=0 (144 participants), group 1 is SEX_1=1 (727). Both columns remain predictor
  inputs in the historical default and are removed by `--drop-sensitive-inputs`.
- The clinical code-to-sex mapping was not authenticated for the processed artifact.
  These are sex-related encoding groups; no clinical sex direction is asserted.

### ABIDE_B

- `RFE_512_processed_standard_data.npz`: `data` has 1584 columns, `label` contains
  diagnosis codes. In the audited artifact there is no explicit `sex`/`gender` key.
- Pass `--sensitive-index 46` (zero-based). Column 47 is the complementary one-hot
  column. Group counts are 146/803. The 48 leading binary columns support the PHENO
  layout but do not establish clinical sex meaning.
- Results are secondary proxy-group analyses. Both proxy columns are predictor inputs
  by default. The predictor-input ablation and the FAL-CUR distance-input sensitivity
  are distinct interventions.

These contracts describe the audited files, not a generic loader for arbitrary ABIDE
exports. The ABIDE-A loader uses broad name matching and can fall back to feature 1
for unfamiliar schemas; inspect the reported sensitive name and pass the verified
index rather than accepting that guess. Validate shape, labels, group counts and
complementarity against the input provenance before rerunning.

## Diagnosis polarity and metric definitions

For input labels {1, 2}, the loader subtracts one. Metric functions treat normalized
class 1 (therefore original input code 2) as positive. Original ABIDE legends describe
DX_GROUP 1=Autism and 2=Control, but that legend does not prove that the upstream
processed artifacts preserved the original coding. Obtain the preprocessing mapping
before calling the released sensitivity an ASD detection rate. Without that mapping,
report numeric-class sensitivity, specificity and EOD, not ASD missed-case rates.

## Reproduction limits

The downstream sklearn scaler is fitted on the currently labeled training samples.
The full upstream standardization/RFE implementation and its fit scope are not in this
distribution; a training-only upstream fit and absence of preprocessing leakage have
not been established. This release supplies neither a complete raw-image-to-feature
pipeline nor a participant mapping. Accessing raw ABIDE data alone is insufficient to
reconstruct these exact processed artifacts. Obtain the matching inputs and provenance
from their original providers subject to applicable terms; record their checksums.

For reuse, retain source attribution and required acknowledgements, avoid
re-identification, and do not redistribute outside the terms of the actual input files.
