#!/usr/bin/env python3
"""
Script 2b: Training & Evaluation of Base Models and their Ensemble

This script performs a robust cross-validation to compare the performance of
RandomSurvivalForest, GradientBoostingSurvivalAnalysis, and a rank-averaging
ensemble of the two. It provides a direct comparison of their IPCW C-index scores.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_ipcw
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline

# --- Import from your project structure ---
from src.modeling.train import load_training_dataset_csv, get_survival_models
from src.modeling.pipeline_components import get_preprocessing_pipeline
from src.visualization.visualize import (
    plot_cv_results,
)


def main():
    """Main function to run the training and evaluation workflow for the ensemble."""
    print("=" * 80)
    print(" SCRIPT 2b: TRAINING & EVALUATION OF BASE MODELS AND ENSEMBLE")
    print("=" * 80)

    # --- 1. LOAD PREPARED DATASET ---
    print("\n[STEP 1/4] Loading the prepared dataset...")
    try:
        X, y = load_training_dataset_csv(
            X_train_path="datasets_processed/X_train_processed.csv",
            y_train_path="datasets_processed/y_train_processed.csv",
        )
        if "ID" in X.columns:
            X = X.drop(columns=["ID"])

        print(
            f"   -> Dataset loaded successfully: {X.shape[0]} samples, {X.shape[1]} features."
        )
    except FileNotFoundError as e:
        print(f"   [ERROR] {e}")
        return

    # --- 2. CONFIGURE MODELS AND CROSS-VALIDATION ---
    print("\n[STEP 2/4] Configuring Models and Cross-Validation...")
    all_models = get_survival_models()
    # We only want to evaluate RSF and GradientBoosting for this comparison
    models_to_evaluate = {
        "RSF": all_models["RSF"],
        "GradientBoosting": all_models["GradientBoosting"],
    }

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # We will store results for RSF, XGB, and the Ensemble
    cv_results = {
        name: [] for name in list(models_to_evaluate.keys()) + ["Ensemble_Rank"]
    }

    print(f"   -> Models to be ensembled: {list(models_to_evaluate.keys())}")
    print(f"   -> Using a {N_SPLITS}-fold cross-validation strategy.")

    # --- 3. RUN CROSS-VALIDATION LOOP ---
    print("\n[STEP 3/4] Starting the cross-validation loop...")

    preprocessor = get_preprocessing_pipeline(X)

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- FOLD {i+1}/{N_SPLITS} ---")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Fit the preprocessor on this fold's training data
        preprocessor.fit(X_train_fold)
        X_train_fold_processed = preprocessor.transform(X_train_fold)
        X_val_fold_processed = preprocessor.transform(X_val_fold)

        fold_predictions = {}

        # Train and predict for each base model
        for name, model in models_to_evaluate.items():
            print(f"  > Training and predicting with: {name}")
            try:
                model.fit(X_train_fold_processed, y_train_fold)
                predictions = model.predict(X_val_fold_processed)
                fold_predictions[name] = predictions

                # Evaluate the base model
                score = concordance_index_ipcw(y_train_fold, y_val_fold, predictions)[0]
                cv_results[name].append(score)
                print(f"    IPCW C-index for {name}: {score:.4f}")

            except Exception as e:
                print(f"    /!\\ ERROR for model {name} on this fold: {e}")
                cv_results[name].append(np.nan)

        # Create and evaluate the ensemble if both models succeeded
        if "RSF" in fold_predictions and "GradientBoosting" in fold_predictions:
            print("  > Creating and evaluating the Ensemble...")

            # Convert scores to ranks
            ranks_rsf = rankdata(fold_predictions["RSF"])
            ranks_xgb = rankdata(fold_predictions["GradientBoosting"])

            # Simple average of ranks
            ensemble_score = 0.6 * ranks_rsf + 0.4 * ranks_xgb

            # Evaluate the ensemble
            ensemble_ipcw = concordance_index_ipcw(
                y_train_fold, y_val_fold, ensemble_score
            )[0]
            cv_results["Ensemble_Rank"].append(ensemble_ipcw)
            print(f"    IPCW C-index for Ensemble: {ensemble_ipcw:.4f}")
        else:
            cv_results["Ensemble_Rank"].append(np.nan)

    # --- 4. ANALYZE AND DISPLAY FINAL RESULTS ---
    print("\n\n" + "=" * 80)
    print("[STEP 4/4] Final Cross-Validation Results")
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

    # --- Generate visualization ---
    print("\n" + "=" * 80)
    print("  Generating Visualization Report")
    print("=" * 80)

    os.makedirs("reports", exist_ok=True)
    plot_cv_results(cv_results)

    print("\n" + "=" * 80)
    print("SCRIPT 2b COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
