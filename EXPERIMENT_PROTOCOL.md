# ECG classification experiment protocol

This protocol defines the comparisons before the held-out test results are inspected. Changes made after inspecting test performance must be documented as exploratory and evaluated on a newly reserved test set.

## Prediction task

The primary task is multilabel classification over `SR`, `AF`, `1AVB`, `LBBB`, `RBBB`, `APB`, `VPB`, `STDD`, and `STE`. One original WFDB record is one example, and every selected diagnosis is represented in a multi-hot target. This preserves legitimate diagnostic co-occurrence instead of copying one waveform into contradictory exclusive classes. A secondary single-label analysis may include only records having exactly one selected diagnosis and must be reported separately.

## Data controls

- Build the manifest only from `Data/WFDBRecords`; do not learn from copied class directories.
- Keep all 12 leads together in a tensor shaped `[12, time]`.
- Group by patient identifier when one is available; otherwise group by immutable original record ID.
- Assert that group and record intersections between training, validation, and test are empty.
- Fit normalization and any learned preprocessing on training records only.
- Apply augmentation to training data only.
- Save the exact split manifest alongside every run.

## Model selection

The primary validation metric is macro-AUPRC. Macro-F1, macro-AUROC, accuracy, and per-class precision/recall/F1 are secondary. Hyperparameters, early stopping, and architecture selection use no test examples.

For multilabel classification, each label's decision threshold is selected on validation predictions using a fixed coarse grid. Final reports also include Brier score, expected calibration error, and 95% grouped bootstrap confidence intervals. Thresholds are never tuned on test predictions.

Every run exports record-aligned validation and test probabilities. These tables permit audit of sample ordering and paired bootstrap comparisons between finalists without rerunning or rounding model outputs.

The classical branch compares only classical architectures. The hybrid branch must compare the same CNN encoder with:

1. a linear classical head;
2. a parameter-matched MLP head;
3. a variational quantum head; and
4. a QCNN-style quantum head.

Every comparison uses the same split manifest, preprocessing, optimization budget where practical, and prespecified seeds.

## Test policy

Evaluate the held-out test set only after choosing the complete training procedure. Report all prespecified seeds, not the best seed. Include parameter counts, training time, confusion matrices, per-class metrics, and patient- or record-level bootstrap confidence intervals.

## Current limitations

- The public release does not expose an explicit patient identifier in the current pipeline; record ID is used as the grouping key until a validated patient mapping is available.
- The original exploratory notebook contains lead-level expansion, random row splitting, and ambiguous dataframe slicing. It remains in the repository for provenance but is not the registered experiment implementation.
- This research code is not validated for clinical use.
