#!/usr/bin/env python3
"""
Script 2c: Systematic Ensemble Finder via Out-of-Fold Predictions

This script performs a single robust cross-validation run to train multiple base
models. It stores their out-of-fold (OOF) predictions and then uses these
predictions to efficiently test dozens of rank-averaging ensemble combinations,
identifying the blend with the highest overall IPCW C-index.
"""
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_ipcw
from scipy.stats import rankdata

# --- Import from your project structure ---
from src.modeling.train import load_training_dataset_csv, get_survival_models
from src.modeling.pipeline_components import get_preprocessing_pipeline


def main():
    print("=" * 80)
    print(" SCRIPT 2c: SYSTEMATIC ENSEMBLE FINDER")
    print("=" * 80)

    # --- 1. LOAD DATASET ---
    print("\n[STEP 1/4] Loading the prepared dataset...")
    X, y = load_training_dataset_csv(
        X_train_path="datasets_processed/X_train_processed.csv",
        y_train_path="datasets_processed/y_train_processed.csv",
    )
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
    print(f"   -> Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

    # --- 2. CONFIGURE MODELS AND CV ---
    print("\n[STEP 2/4] Configuring Models and Cross-Validation...")
    models_to_train = get_survival_models()
    # On peut exclure les modèles les moins performants si on le souhaite
    # models_to_train.pop("ComponentwiseGB", None)

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # DataFrame pour stocker les prédictions Out-of-Fold (OOF)
    oof_predictions = pd.DataFrame(index=X.index, columns=models_to_train.keys())

    print(f"   -> {len(models_to_train)} base models will be trained.")
    print(f"   -> Using a {N_SPLITS}-fold cross-validation strategy.")

    # --- 3. GENERATE OUT-OF-FOLD PREDICTIONS ---
    print("\n[STEP 3/4] Generating Out-of-Fold (OOF) predictions...")
    preprocessor = get_preprocessing_pipeline(X)

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- FOLD {i+1}/{N_SPLITS} ---")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Fit preprocessor on this fold's training data
        preprocessor.fit(X_train_fold)
        X_train_fold_processed = preprocessor.transform(X_train_fold)
        X_val_fold_processed = preprocessor.transform(X_val_fold)

        for name, model in models_to_train.items():
            print(f"  > Training {name}...")
            try:
                model.fit(X_train_fold_processed, y_train_fold)
                # Faire des prédictions et les stocker à la bonne place dans le df OOF
                predictions = model.predict(X_val_fold_processed)
                oof_predictions.loc[val_idx, name] = predictions
            except Exception as e:
                print(f"    /!\\ ERROR for model {name}: {e}")

    # Vérifier si des prédictions ont échoué
    oof_predictions.dropna(axis=1, how="all", inplace=True)
    print("\n✓ Out-of-Fold predictions generated for all models.")

    # --- 4. EVALUATE ALL ENSEMBLE COMBINATIONS ---
    print("\n[STEP 4/4] Evaluating all ensemble combinations...")

    # Évaluer d'abord les modèles de base sur l'ensemble des prédictions OOF
    base_model_scores = {}
    for name in oof_predictions.columns:
        preds = oof_predictions[name].values.astype(float)
        score = concordance_index_ipcw(y, y, preds)[0]
        base_model_scores[name] = score
        print(f"  - Overall OOF Score for {name:<20}: {score:.5f}")

    print("\n--- Testing Ensemble Combinations ---")

    ensemble_results = []
    model_names = list(oof_predictions.columns)

    # Itérer sur toutes les tailles de combinaison possibles (paires, triplets, etc.)
    for k in range(2, len(model_names) + 1):
        # Générer toutes les combinaisons de k modèles
        for combo in itertools.combinations(model_names, k):
            combo_name = " + ".join(combo)
            print(f"  > Testing: {combo_name}")

            # Récupérer les prédictions OOF pour les modèles de la combinaison
            combo_preds_df = oof_predictions[list(combo)]

            # Transformer en classements
            combo_ranks_df = combo_preds_df.rank()

            # Faire la moyenne des classements
            ensemble_score_ranks = combo_ranks_df.mean(axis=1).values

            # Évaluer l'ensemble
            ensemble_ipcw = concordance_index_ipcw(y, y, ensemble_score_ranks)[0]
            ensemble_results.append((combo_name, ensemble_ipcw))
            print(f"    IPCW C-index: {ensemble_ipcw:.5f}")

    # --- Afficher le classement final ---
    print("\n\n" + "=" * 80)
    print("  FINAL ENSEMBLE RANKING")
    print("=" * 80)

    # Combiner les scores de base et d'ensemble
    final_ranking = {**base_model_scores}
    for name, score in ensemble_results:
        final_ranking[name] = score

    # Trier par score, du meilleur au moins bon
    sorted_ranking = sorted(
        final_ranking.items(), key=lambda item: item[1], reverse=True
    )

    for name, score in sorted_ranking:
        print(f"  {score:.5f} - {name}")

    best_ensemble_name, best_ensemble_score = sorted_ranking[0]
    print(
        f"\n---> Best combination found: '{best_ensemble_name}' with a score of {best_ensemble_score:.5f}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
