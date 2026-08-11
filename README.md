# Machine Learning for Cardiac Arrhythmia Detection

This repository explores automated cardiac arrhythmia classification from 12-lead electrocardiogram (ECG) signals. The project combines ECG data preparation with classical, deep-learning, and quantum-machine-learning experiments.

The main implementation is an exploratory Jupyter notebook that:

- reads paired WFDB header (`.hea`) and MATLAB signal (`.mat`) files;
- extracts diagnostic SNOMED CT codes from record metadata;
- maps diagnostic codes to readable arrhythmia labels;
- reorganizes records into class-specific folders;
- constructs tabular training data from ECG signals;
- trains and evaluates a PyTorch 1D convolutional neural network (CNN);
- compares the CNN with a random forest baseline; and
- investigates dimensionality reduction, quantum encodings, and a PennyLane quantum convolutional neural network (QCNN).

> [!IMPORTANT]
> This is a research project, not a medical device. Its models and results must not be used for diagnosis, treatment, or other clinical decisions.

## Repository contents

```text
.
|-- Code/
|   |-- Scientific_ECG_Experiment.ipynb
|   |-- ML_for_Cardiac_Arrythmia_Detection.ipynb
|   |-- ML_for_Cardiac_Arrythmia_Detection.html
|   `-- ML_for_Cardiac_Arrythmia_Detection.zip
|-- Data/
|   |-- WFDBRecords/            # Raw downloaded records (not tracked)
|   `-- <arrhythmia classes>/   # Class-specific records (not tracked)
|-- Papers/                     # Background literature
|-- Performance/                # Saved model metrics and learning curves
|-- Poster/                     # Research poster source and exported PDF
|-- References/
|   |-- ConditionNames_SNOMED-CT.csv
|   |-- LICENSE.txt
|   |-- RECORDS
|   `-- SHA256SUMS.txt
`-- README.md
```

`Scientific_ECG_Experiment.ipynb` is the leakage-safe, reproducible entry point. The larger notebook whose filename retains the original `Arrythmia` spelling is preserved as an exploratory record and should not be used for final model comparisons.

The prespecified evaluation rules are documented in `EXPERIMENT_PROTOCOL.md`.

## Dataset

The repository's research materials cite the PhysioNet dataset:

> J. Zheng, H. Guo, and H. Chu, “A large scale 12-lead electrocardiogram database for arrhythmia study (version 1.0.0),” PhysioNet, 2022. DOI: [10.13026/wgex-er52](https://doi.org/10.13026/wgex-er52).

The project materials describe 45,152 12-lead ECG records, each approximately 10 seconds long and sampled at 500 Hz. The preprocessing workflow uses the supplied headers and the SNOMED CT mapping in `References/ConditionNames_SNOMED-CT.csv` to assign records to diagnostic categories. Because diagnoses can co-occur, the registered experiment treats this as multilabel classification and keeps each original ECG exactly once.

The ECG files are intentionally excluded from Git because the dataset contains tens of thousands of records and is too large for normal GitHub storage. Git tracks only `.gitkeep` placeholders for the category directories.

### Expected local data layout

After obtaining the dataset from its official source, place the raw records under:

```text
Data/WFDBRecords/
```

The sorting portion of the notebook creates or populates directories such as:

```text
Data/1AVB/
Data/AF/
Data/APB/
Data/LBBB/
Data/RBBB/
Data/SR/
Data/STDD/
Data/STE/
Data/VPB/
```

Each record normally consists of files with the same basename:

```text
JS00001.hea
JS00001.mat
```

Anything placed inside `Data/` remains local and should not appear in a Git commit. The category folders remain visible in Git through their `.gitkeep` files.

## Environment setup

Python 3.12 was recorded in the current notebook metadata. A virtual environment is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas scipy matplotlib scikit-learn scikit-optimize torch torcheval wfdb pennylane tqdm
jupyter notebook Code/Scientific_ECG_Experiment.ipynb
```

GPU acceleration is optional. The CNN code selects CUDA when available and contains support for Apple's Metal Performance Shaders (MPS). The quantum experiments can be computationally expensive when run in simulation.

## Experiment task guide

Run commands from the repository root. Use a different `--output-dir` for every configuration. Passing `--overwrite` intentionally replaces an existing validation run.

| Task | File or branch to use |
| --- | --- |
| Review the registered protocol | `EXPERIMENT_PROTOCOL.md` |
| Inspect the classical workflow | `Code/Scientific_ECG_Experiment.ipynb` |
| Run a classical experiment | `Code/run_classical_experiment.py` |
| Run automated checks | `tests/` |
| Review historical exploration only | `Code/ML_for_Cardiac_Arrythmia_Detection.ipynb` |
| Run hybrid CNN/quantum experiments | Switch to the `hybrid-model` branch |

### 1. Install and verify the classical environment

```powershell
git switch main
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
python Code/run_classical_experiment.py --help
```

### 2. Inspect the workflow interactively

```powershell
jupyter notebook Code/Scientific_ECG_Experiment.ipynb
```

The larger `ML_for_Cardiac_Arrythmia_Detection.ipynb` notebook is retained for provenance. Do not use its lead-level split or dataframe slicing for final comparisons.

### 3. Train and validate the classical CNN

The default task is nine-label multilabel classification. Validation does not open the test waveforms:

```powershell
python Code/run_classical_experiment.py `
  --stage validate `
  --seed 43 `
  --output-dir artifacts/classical/seed-43
```

Optional training-only augmentation:

```powershell
python Code/run_classical_experiment.py `
  --stage validate `
  --seed 43 `
  --augment `
  --output-dir artifacts/classical-augmented/seed-43
```

Secondary single-label analysis must use a separate output directory and must not be mixed with multilabel results:

```powershell
python Code/run_classical_experiment.py `
  --stage validate `
  --task single-label `
  --target-classes "SR,AF,1AVB,LBBB,RBBB,APB,VPB,STDD,STE" `
  --seed 43 `
  --output-dir artifacts/classical-single-label/seed-43
```

### 4. Repeat a confirmed configuration across seeds

```powershell
$seeds = 13, 23, 33, 43, 53
foreach ($seed in $seeds) {
  python Code/run_classical_experiment.py `
    --stage validate `
    --seed $seed `
    --output-dir "artifacts/classical/seed-$seed"
}
```

Do not select the best seed. Report the complete prespecified set.

### 5. Open the classical test set once

Use the same seed, task, target classes, and output directory as validation:

```powershell
python Code/run_classical_experiment.py `
  --stage test `
  --seed 43 `
  --output-dir artifacts/classical/seed-43
```

The runner refuses to test without a completed validation checkpoint and matching configuration.

### 6. Run hybrid and quantum comparisons

The controlled linear, parameter-matched MLP, VQC, and QCNN experiments live on `hybrid-model`. Its README contains commands for the experiment grid, frozen-encoder comparison, re-uploading, finite-shot/noise ablations, sealed testing, and paired finalist analysis:

```powershell
git switch hybrid-model
python -m pip install -r requirements-hybrid.txt
python -m pip install pytest
python -m pytest -q
jupyter notebook Code/Hybrid_ECG_Experiment.ipynb
```

### 7. Understand generated files

Each classical experiment directory may contain:

| File | Purpose |
| --- | --- |
| `config.json` | Data and training configuration |
| `split_manifest.csv` | Exact record and split assignment |
| `class_counts.json` | Label prevalence in every split |
| `normalization.json` | Training-only channel statistics |
| `best_model.pt` | Best validation checkpoint |
| `training.json` | Epoch history and timing |
| `decision_thresholds.json` | Per-label thresholds tuned on validation only |
| `validation_predictions.csv` | Record-aligned validation probabilities |
| `validation_metrics.json` | Validation results used for selection |
| `test_predictions.csv` | Record-aligned sealed-test probabilities |
| `test_metrics.json` | Final metrics, calibration, and confidence intervals |

Generated artifacts are ignored by Git. Preserve the selected result directories separately if they are needed for a paper or audit.

## Legacy notebook paths

The notebook currently contains absolute Windows paths from the original development computer. Update the configuration cell near the beginning of the notebook so that these locations match your clone:

```python
from pathlib import Path

repo_dir = Path.cwd()
if repo_dir.name == "Code":
    repo_dir = repo_dir.parent

source_dir = repo_dir / "Data" / "WFDBRecords"
data_dir = repo_dir / "Data"
ref_dir = repo_dir / "References"
```

Also search the notebook for other absolute paths beginning with `C:/Users/` before executing export or cached-data cells.

The notebook documents several alternative dataframe-building and model configurations. Some later cells redefine earlier functions and classes. Run the notebook section you intend to use in order, and review its parameters rather than assuming every experimental branch should be executed in one pass.

## Experimental workflow

### 1. Sort and label ECG records

The preprocessing section traverses the raw WFDB record tree, reads header metadata with `wfdb`, extracts numeric diagnostic codes, and cross-references those codes against the SNOMED CT condition table. Valid `.hea`/`.mat` pairs are then grouped into class directories.

### 2. Construct a training dataframe

The notebook includes several approaches for loading signal arrays and forming a Pandas dataframe. The current experiments flatten or select ECG samples, attach a numeric class label, randomize the rows using a fixed seed, and replace missing tensor values before training.

One documented classical experiment uses nine categories for comparison:

```text
SR, AF, 1AVB, LBBB, RBBB, APB, VPB, STDD, STE
```

Other notebook cells select categories dynamically according to the number of available records.

### 3. Train the classical models

The deep-learning path uses PyTorch data loaders and a 1D CNN with configurable filter counts, kernel size, dropout, batch size, and learning rate. The notebook includes Bayesian hyperparameter optimization with `scikit-optimize` and reports accuracy, precision, recall, F1, AUROC, AUPRC, and confusion matrices.

A `RandomForestClassifier` with 100 estimators is included as a classical baseline.

### 4. Explore the quantum model

The quantum section evaluates preprocessing and encoding approaches including PCA, autoencoding, undersampling, Laplacian eigenmaps, locality-preserving projections, compressive sensing, and FFT-based compression. PennyLane circuits provide multiple convolution and pooling configurations plus a QCNN hyperparameter-search loop.

The poster describes the multiclass QCNN as ongoing research constrained by simulator and hardware costs; it should not be interpreted as a production-ready classifier.

## Reported results

The research poster reports the following averages over seeds 13, 23, 33, 43, and 53:

| Model | Accuracy | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| Experimental 1D CNN | 99.84% ± 0.04% | 99.70% ± 0.36% | 99.78% ± 0.09% | 99.72% ± 0.22% |
| Baseline model | 90.05% ± 7.99% | 62.82% ± 25.76% | 67.25% ± 23.77% | 62.17% ± 25.77% |

These values are transcribed from `Poster/main.tex`; they have not been independently reproduced as part of this README update. Dataset composition, preprocessing, class balance, leakage controls, and split strategy should be reviewed before comparing them with other work.

## Reproducibility notes

- The notebook uses explicit random seeds in several experiments, but some cells change the selected seed.
- The primary split is approximately 80% training and 20% testing, followed by a validation split within the training set.
- Some comments and experimental cells describe different signal windows or sampling rates. Confirm the active cell configuration before training.
- Generated CSV files, raw ECG records, and reorganized dataset files belong under `Data/` and are ignored by Git.
- Saved figures in `Performance/` and the research poster provide a record of prior experiments, but they are not substitutes for rerunning the evaluation.

## References

- J. Zheng, H. Guo, and H. Chu, “A large scale 12-lead electrocardiogram database for arrhythmia study,” PhysioNet, 2022. DOI: [10.13026/wgex-er52](https://doi.org/10.13026/wgex-er52).
- T. Vu et al., “Real-time arrhythmia detection using convolutional neural networks,” *Frontiers in Big Data*, vol. 6, 2023. DOI: [10.3389/fdata.2023.1270756](https://doi.org/10.3389/fdata.2023.1270756).
- Additional background papers are stored in `Papers/`.

## Licensing

`References/LICENSE.txt` contains the Creative Commons Attribution 4.0 license distributed with the dataset reference files. No separate license for the repository's original source code is currently declared. Dataset users should review and follow the official PhysioNet terms and attribution requirements.
