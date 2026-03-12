## Overview

This repository documents the AML survival challenge workflow that ranked second in the QRT / Institut Gustave Roussy competition. The goal is to order adult acute myeloid leukemia (AML) patients by survival risk using clinical, cytogenetic, and molecular features. The evaluation metric is the IPCW-C-index truncated at seven years, so submissions only need to provide `ID` and `risk_score` columns with lower scores indicating longer predicted survival.

## Repository structure

- **challenge_code/scripts/stages/**: production-ready stages. `prepare_data.py` validates raw tables, merges external cohorts, engineers auxiliary features, and runs the imputation + preprocessing pipeline before writing processed CSVs. `train_models.py` performs GroupKFold/KFold cross-validation, saves IPCW-C fold predictions, ranks ensembles, and retrains each estimator on the full dataset. `train_models_clean.py` reuses the same training flow after discarding the highest-error patients identified by the most recent error analysis, producing `_clean` artifacts. `optimize_models.py` runs Optuna-style searches for the configurable estimators. `predict.py` loads persisted models, rank-averages their predictions on the processed test set, and writes a submission CSV. `run_pipeline.py` orchestrates these stages via CLI so automation can call multiple steps without legacy wrappers.
- **challenge_code/scripts/experiments/**: exploratory tooling (drift/correlation reports, variant downloads, error analysis) kept separate from the deployment pipeline.
- **challenge_code/datas/**: original source data (`X_train/`, `X_test/`, `target_train.csv`, `external/clinvar.vcf`, etc.), preserved for audits and reruns.
- **challenge_code/datasets_processed/**: outputs from `prepare_data.py` such as `X_train_processed.csv`, `y_train_processed.csv`, `X_test_processed.csv`, and intermediate manifests used during imputation.
- **challenge_code/models/**: saved joblib models (`model_*.joblib`), ensemble metadata (`ensemble_meta*.json`), imputer metadata, and experiment result folders.
- **challenge_code/reports/**: ranking tables, IPCW drift diagnostics, correlation heatmaps, and other analysis artifacts generated during preprocessing or training.
- **challenge_code/submissions/**: timestamped prediction exports plus `latest_submission.csv` for quick access.
- **challenge_code/src/**: reusable domain logic (`data/`, `modeling/`, `utils/`, `visualization/`) and `config.py`, which centralizes dataset paths, preprocessing toggles, curated feature definitions, and experiment helpers.
- **challenge_code/requirements.txt**: Python dependency list (scikit-survival, LightGBM, optuna, etc.).

There are no notebooks in this repository; reproducible analysis lives under `scripts/experiments/` or `reports/data_explore/`.

## Installation

```bash
cd challenge_code
python -m venv .venv
.venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
```

Adjust the activation command for other operating systems when needed.

## Detailed preprocessing flow

`prepare_data.py` carries out the following steps:
1. Load and clean the clinical, molecular, and target tables, validating IDs and reconciling `target_train.csv` with the clinical table.
2. Create auxiliary indicators (missingness, ratios, statistics) and optionally merge data from the Beat AML / TCGA cohorts to bolster imputation or augment training.
3. Apply `ExternalDataManager` lookups, compute cytogenetic risk features, molecular signatures, and combinatorial cyto-molecular interactions.
4. Prune redundant columns, enforce float32 casting, and execute the imputation pipeline defined in `src/config.py` (early imputation plus auxiliary columns or advanced estimators).
5. Run `get_preprocessing_pipeline()` to scale numerics, encode categoricals, and align the final column order.
6. Persist `X_train_processed.csv`, `y_train_processed.csv`, `X_test_processed.csv`, feature manifests, and hash files under the experiment directory for traceability.

Each step logs which auxiliary features were generated, which cohorts were fused, and the active feature subset mode so downstream stages can rely on deterministic column lists.

## Training and ensembling

`train_models.py` loads the processed datasets, drops identifiers, and runs 5-fold CV using `CENTER_GROUP` when available. During CV:
- Fit every configured estimator (`get_survival_models()` in `src/modeling/train.py`).
- Store IPCW-C predictions on the validation splits and compute fold-wise IPCW scores.
- Remove models that produced NaN predictions and evaluate every combination of the remaining estimators by rank averaging.

The stage writes `ensemble_ranking.csv` (saved both in `reports/` and the experiment folder) plus metadata files (`ensemble_meta.json`, `fold_scores.json`, `training_report.json`). After CV, each estimator is retrained on the full training set and saved under `challenge_code/models/`.

`train_models_clean.py` wraps the same workflow but first reads `error_analysis_detailed.csv` to identify the top `--outliers` samples that contributed the most to IPCW errors. It drops those IDs, logs them, and reruns the CV/ensemble process so the `clean` artifacts (`_clean` suffix) reflect a filtered cohort.

`optimize_models.py` runs Optuna-style sweeps for RSF, Gradient Boosting, ExtraTrees, and other configurable estimators. The best parameters are written to JSON files for adoption by `train_models*.py` or downstream experiments.

## Prediction and submission

`predict.py` reads `datasets_processed/X_test_processed.csv` and drops IDs. It accepts `--models` (comma-separated names or `all`). Each model’s prediction is converted to ranks, the ranks are averaged, and the result is saved as `risk_score`. The submission is exported as `submissions/submission_<models>_<timestamp>.csv`, mirrored to `latest_submission.csv`, and recorded in the current experiment folder via `save_predictions()`.

`run_pipeline.py` glues everything together. Pass `--stages prepare train optimize predict` (or `all`) to execute all stages, `--skip-predict` to stop after training, or a custom subset of stages to match automated runs. The orchestrator uses `src/config.py` to resolve paths and logs each step so the CI logs show which artifacts were produced.

## Remarks

- Artifact directories (`models/`, `reports/`, `submissions/`) stay outside the source tree and can be cleared between experiments without affecting the pipeline code.
- Global toggles for imputation policies, auxiliary features, float32 enforcement, and feature subset modes live in `src/config.py`.
- The pipeline is CPU-bound but some steps (evaluating every ensemble combination, running hyperparameter optimizers) can be time-consuming.
- The benchmark metric is the IPCW-C-index (scikit-survival implementation), truncated at 7 years to balance censored and uncensored patients. Higher scores mean better ranking of risk.
- The challenge dataset consists of 3,323 training and 1,193 test patients from 24 centers, with clinical/lab data, cytogenetic risk, and molecular mutations involving gene, variant effect, and VAF. Submissions need only `ID` and `risk_score`.
- The second-place finish underscores the robustness of the preprocessing, imputation, ensembling, and submission strategy described in this README.
