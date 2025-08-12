#!/usr/bin/env python3
"""
Script 3/3: Final Model Training and Ensemble Prediction

This script trains the champion ensemble models (identified via cross-validation)
on the entire training dataset. It then generates predictions on the test set,
combines them using a rank-averaging strategy, and produces the final
submission file.
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# --- Import from your project structure ---
from src.modeling.train import load_training_dataset_csv, get_survival_models
from src.modeling.pipeline_components import get_preprocessing_pipeline


def main():
    """Main function to train final models and generate the ensemble submission."""
    print("=" * 80)
    print(" SCRIPT 3: FINAL MODEL TRAINING & ENSEMBLE PREDICTION")
    print("=" * 80)

    # --- 1. LOAD ALL NECESSARY DATA ---
    print("\n[STEP 1/5] Loading all necessary datasets...")
    try:
        # Load the full training data
        X_train_full, y_train_full = load_training_dataset_csv(
            X_train_path="datasets_processed/X_train_processed.csv",
            y_train_path="datasets_processed/y_train_processed.csv",
        )
        if "ID" in X_train_full.columns:
            X_train_full = X_train_full.drop(columns=["ID"])

        # Load the test data
        X_test = pd.read_csv("datasets_processed/X_test_processed.csv")
        test_ids = X_test["ID"].copy()
        if "ID" in X_test.columns:
            X_test = X_test.drop(columns=["ID"])

        print(f"   -> Full training data: {X_train_full.shape}")
        print(f"   -> Test data: {X_test.shape}")

        # Ensure test columns match training columns
        X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)

    except FileNotFoundError as e:
        print(f"   [ERROR] {e}")
        print(
            "   Please ensure all processed datasets exist by running script 1 first."
        )
        return

    # --- 2. DEFINE THE CHAMPION ENSEMBLE MODELS ---
    print("\n[STEP 2/5] Defining the champion ensemble models...")
    all_models = get_survival_models()
    champion_model_names = ["RSF", "GradientBoosting", "CoxNet", "ComponentwiseGB"]
    champion_models = {name: all_models[name] for name in champion_model_names}
    print(f"   -> Models to be trained: {list(champion_models.keys())}")

    # --- 3. TRAIN FINAL MODELS ON 100% OF TRAINING DATA ---
    print("\n[STEP 3/5] Training final models on the entire training dataset...")

    trained_models = {}
    for name, model in champion_models.items():
        print(f"  > Training final model: {name}...")
        try:
            model.fit(X_train_full, y_train_full)
            trained_models[name] = model
            print(f"    ✓ {name} trained successfully.")
        except Exception as e:
            print(f"    /!\\ ERROR training final model {name}: {e}")
            # If a model fails here, we can't create the submission
            return

    # --- 4. GENERATE PREDICTIONS ON THE TEST SET ---
    print("\n[STEP 4/5] Generating predictions on the test set...")

    test_predictions = {}
    for name, model in trained_models.items():
        print(f"  > Predicting with: {name}...")
        predictions = model.predict(X_test)
        test_predictions[name] = predictions

    # --- 5. CREATE AND SAVE THE FINAL ENSEMBLE SUBMISSION ---
    print("\n[STEP 5/5] Creating final ensemble submission via rank averaging...")

    # Convert individual predictions to ranks
    ranks_df = pd.DataFrame(
        {name: rankdata(preds) for name, preds in test_predictions.items()}
    )

    # Calculate the simple average of the ranks
    final_ensemble_score = ranks_df.mean(axis=1).values

    # Create the submission DataFrame
    submission_df = pd.DataFrame({"ID": test_ids, "risk_score": final_ensemble_score})

    # Save the submission file
    os.makedirs("submissions", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submissions/submission_FINAL_ENSEMBLE_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"  > Final submission file saved to: {submission_path}")
    print("\n  > Final prediction score statistics:")
    print(submission_df["risk_score"].describe())

    print("\n" + "=" * 80)
    print("  PROJECT COMPLETE - FINAL SUBMISSION IS READY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
