#!/usr/bin/env python3
"""
Script 2/3: Model Training, Evaluation & Selection via Cross-Validation

This script performs a robust evaluation of multiple survival models using
k-fold cross-validation to identify the best performing model based on the
IPCW concordance index. The champion model is then retrained on the entire
dataset and saved as a complete, ready-to-use prediction pipeline.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_ipcw
from sklearn.pipeline import Pipeline

# --- Import from your project structure ---
# Make sure these paths are correct for your project layout
from src.modeling.train import load_training_dataset_csv, get_survival_models
from src.modeling.pipeline_components import (
    get_preprocessing_pipeline,
)  # We need to create this function


def main():
    """Main function to run the training and evaluation workflow."""
    print("=" * 80)
    print(" SCRIPT 2/3: MODEL TRAINING & EVALUATION (with Cross-Validation)")
    print("=" * 80)

    # --- 1. LOAD PREPARED DATASET ---
    print("\n[STEP 1/5] Loading the prepared dataset...")
    try:
        X, y = load_training_dataset_csv(
            X_train_path="datasets_featured/X_train_featured.csv",
            y_train_path="datasets_featured/y_train_featured.csv",
        )
        # Drop ID from features, but keep it for later if needed
        train_ids = X["ID"].copy()
        X = X.drop(columns=["ID"])
        features = X.columns.tolist()
        print(
            f"   -> Dataset loaded successfully: {X.shape[0]} samples, {X.shape[1]} features."
        )
    except FileNotFoundError as e:
        print(f"   [ERROR] {e}")
        print("   Please run the data preparation script (1_prepare_data.py) first.")
        return

    # --- 2. CONFIGURE CROSS-VALIDATION ---
    print("\n[STEP 2/5] Configuring Cross-Validation...")
    models_to_evaluate = get_survival_models()
    N_SPLITS = 5  # 5 folds is a robust standard
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    cv_results = {name: [] for name in models_to_evaluate.keys()}

    print(f"   -> {len(models_to_evaluate)} models will be evaluated.")
    print(f"   -> Using a {N_SPLITS}-fold cross-validation strategy.")

    # --- 3. RUN CROSS-VALIDATION LOOP ---
    print("\n[STEP 3/5] Starting the cross-validation loop...")

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- FOLD {i+1}/{N_SPLITS} ---")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        for name, model in models_to_evaluate.items():
            print(f"  > Training and evaluating: {name}")
            try:
                # We use the raw (but feature-engineered) data here.
                # The preprocessing pipeline will handle imputation and scaling inside the fit.
                preprocessor = get_preprocessing_pipeline(X)

                # Create a full pipeline for this fold
                fold_pipeline = Pipeline(
                    steps=[("preprocessor", preprocessor), ("model", model)]
                )

                # Fit the entire pipeline on the training fold
                fold_pipeline.fit(X_train_fold, y_train_fold)

                # Predict on the validation fold
                # Note: For scikit-survival, predict() returns a risk score.
                predictions = fold_pipeline.predict(X_val_fold)

                # Calculate IPCW C-index
                # y_train_fold is used to learn the censoring distribution
                # y_val_fold is used for the actual evaluation
                score = concordance_index_ipcw(y_train_fold, y_val_fold, predictions)[0]
                cv_results[name].append(score)
                print(f"    IPCW C-index on this fold: {score:.4f}")

            except Exception as e:
                print(f"    /!\\ ERROR for model {name} on this fold: {e}")
                cv_results[name].append(np.nan)

    # --- 4. ANALYZE RESULTS AND SELECT BEST MODEL ---
    print("\n\n" + "=" * 80)
    print("[STEP 4/5] Analyzing cross-validation results")
    print("=" * 80)

    final_scores = {}
    for name, scores in cv_results.items():
        if all(np.isnan(s) for s in scores):
            mean_score, std_score = np.nan, np.nan
        else:
            mean_score = np.nanmean(scores)
            std_score = np.nanstd(scores)
        final_scores[name] = {"mean_ipcw": mean_score, "std_ipcw": std_score}
        print(
            f"  Model: {name:<20} | Mean IPCW C-index: {mean_score:.4f} (+/- {std_score:.4f})"
        )

    best_model_name = max(
        final_scores,
        key=lambda k: (
            final_scores[k]["mean_ipcw"]
            if not np.isnan(final_scores[k]["mean_ipcw"])
            else -1
        ),
    )
    print(
        f"\n---> Best performing model: '{best_model_name}' with a mean score of {final_scores[best_model_name]['mean_ipcw']:.4f}"
    )

    # --- 5. FINALIZE AND SAVE THE PREDICTION PIPELINE ---
    print("\n" + "=" * 80)
    print(
        f"[STEP 5/5] Finalizing: Retraining '{best_model_name}' on all data and saving pipeline"
    )
    print("=" * 80)

    # Instantiate the final preprocessor and the champion model
    final_preprocessor = get_preprocessing_pipeline(X)
    final_best_model = get_survival_models()[best_model_name]

    # Create the final, complete pipeline
    final_prediction_pipeline = Pipeline(
        steps=[("preprocessor", final_preprocessor), ("model", final_best_model)]
    )

    # Fit the entire pipeline on 100% of the training data
    # This ensures the model learns as much as possible before prediction
    print("  > Fitting the final pipeline on the entire training dataset...")
    final_prediction_pipeline.fit(X, y)

    # Save the single, powerful pipeline object
    os.makedirs("models", exist_ok=True)
    pipeline_path = os.path.join("models", "final_prediction_pipeline.joblib")
    joblib.dump(final_prediction_pipeline, pipeline_path)

    print(f"  > Final prediction pipeline saved successfully to: '{pipeline_path}'")

    print("\n" + "=" * 80)
    print("SCRIPT 2/3 COMPLETED SUCCESSFULLY!")
    print("The final pipeline is ready for making predictions on new data.")
    print("=" * 80)


if __name__ == "__main__":
    main()
