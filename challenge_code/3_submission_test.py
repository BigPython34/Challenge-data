#!/usr/bin/env python3
"""
Script 3d: Final Submission Generation with a Diversified Ensemble (RSF + CoxNet)

This script trains the two most diverse and robust models (RandomSurvivalForest
and CoxnetSurvivalAnalysis) on the entire training dataset. It then generates
predictions on the test set, combines them using a rank-averaging strategy,
and produces the final submission file.
"""
import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# --- Import from your project structure ---
from src.modeling.train import load_training_dataset_csv, get_survival_models


def main():
    """Main function to train final models and generate the RSF + CoxNet ensemble submission."""
    print("=" * 80)
    print(" SCRIPT 3d: FINAL SUBMISSION - DIVERSIFIED ENSEMBLE (RSF + CoxNet)")
    print("=" * 80)

    # --- 1. LOAD ALL NECESSARY DATA ---
    print("\n[STEP 1/4] Loading all necessary datasets...")
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

    # --- 2. DEFINE THE ENSEMBLE MODELS ---
    print("\n[STEP 2/4] Defining the RSF + CoxNet ensemble models...")
    all_models = get_survival_models()
    # --- MODIFICATION ICI ---
    # On ne sélectionne que les deux modèles qui nous intéressent
    champion_model_names = ["RSF", "CoxNet"]
    champion_models = {name: all_models[name] for name in champion_model_names}
    print(f"   -> Models to be trained: {list(champion_models.keys())}")

    # --- 3. TRAIN FINAL MODELS ON 100% OF TRAINING DATA ---
    print("\n[STEP 3/4] Training final models on the entire training dataset...")

    trained_models = {}
    for name, model in champion_models.items():
        print(f"  > Training final model: {name}...")
        try:
            model.fit(X_train_full, y_train_full)
            trained_models[name] = model
            print(f"    ✓ {name} trained successfully.")
        except Exception as e:
            print(f"    /!\\ ERROR training final model {name}: {e}")
            return

    # --- 4. CREATE AND SAVE THE FINAL ENSEMBLE SUBMISSION ---
    print("\n[STEP 4/4] Creating final ensemble submission via rank averaging...")

    # Prédire avec chaque modèle
    preds_rsf = trained_models["RSF"].predict(X_test)
    preds_coxnet = trained_models["CoxNet"].predict(X_test)

    # Convertir en classements
    ranks_rsf = rankdata(preds_rsf)
    ranks_coxnet = rankdata(preds_coxnet)

    # Moyenne simple des classements (la plus robuste)
    final_ensemble_score = 0.5 * ranks_rsf + 0.5 * ranks_coxnet

    # Créer le DataFrame de soumission
    submission_df = pd.DataFrame({"ID": test_ids, "risk_score": final_ensemble_score})

    # Sauvegarder le fichier
    os.makedirs("submissions", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"submissions/submission_ENSEMBLE_RSF_CoxNet_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"  > Final submission file saved to: {submission_path}")
    print("\n  > Final prediction score statistics:")
    print(submission_df["risk_score"].describe())

    print("\n" + "=" * 80)
    print("  PROJECT COMPLETE - FINAL SUBMISSION IS READY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
