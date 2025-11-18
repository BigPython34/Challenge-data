#!/usr/bin/env python3
"""
Script d'optimisation des hyperparamètres - RSF INTENSIF
Optimise intensivement les hyperparamètres du Random Survival Forest
"""
import pickle
import os
from src.modeling.optimize_hyperparameters import (
    resume_or_start_rsf_optimization,
)


def main():
    """Optimise intensivement les hyperparamètres du Random Survival Forest"""
    print("=== OPTIMISATION INTENSIVE RSF ===")
    print("Objectif : Optimiser RSF avec recherche intensive")
    print("=" * 50)

    # 1. Chargement du dataset
    print("\n1. Chargement du dataset...")
    dataset_path = "datasets/training_dataset.pkl"

    if not os.path.exists(dataset_path):
        print("ERREUR : Dataset introuvable !")
        print(f"   Fichier attendu : {dataset_path}")
        print("   Veuillez d'abord executer : python 1_prepare_data.py")
        return

    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        print("   Dataset chargé avec succès")
        print(
            f"   {dataset['metadata']['n_samples_train']} échantillons d'entraînement"
        )
        print(f"   {dataset['metadata']['n_features']} features")
    except Exception as e:
        print(f"ERREUR lors du chargement : {e}")
        return


    X_train = dataset["X_train"]
    y_train = dataset["y_train"]

    # 2. Configuration de l'optimisation
    print("\n2. Configuration de l'optimisation RSF intensive...")
    n_trials = 300
    n_splits = 5

    print(f"   Nombre d'essais RSF : {n_trials}")
    print(f"   Validation croisée : {n_splits} folds")
    print(f"   Estimation du temps : 3-4 heures")
    print(f"   Total d'évaluations : {n_trials * n_splits} = {n_trials * n_splits}")
    print(f"   ({n_trials} trials × {n_splits} folds)")
    print("   🔥 OPTIMISATION RSF INTENSIVE ACTIVÉE 🔥")

    # 3. Optimisation intensive du RSF
    print("\n3. Lancement de l'optimisation RSF intensive...")
    print("🔍 Recherche d'optimisations existantes...")

    try:
        best_params, best_score, csv_path = resume_or_start_rsf_optimization(
            X_train=X_train, y_train=y_train, n_trials=n_trials, n_splits=n_splits
        )

        print("\n" + "=" * 50)
        print("✅ OPTIMISATION RSF TERMINÉE AVEC SUCCÈS")
        print("=" * 50)
        print(f"Meilleur score RSF : {best_score:.5f}")
        print(f"Meilleurs paramètres RSF : {best_params}")
        if csv_path:
            print(f"Fichier CSV généré : {csv_path}")

    except Exception as e:
        print(f"\nERREUR lors de l'optimisation : {e}")
        raise


if __name__ == "__main__":
    main()
