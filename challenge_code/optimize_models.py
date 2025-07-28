#!/usr/bin/env python3
"""
Script d'optimisation des hyperparamètres
Optimise les hyperparamètres des modèles Gradient Boosting et Random Survival Forest
"""
import pickle
import os
from src.modeling.optimize_hyperparameters import optimize_both_models
from src.utils.helpers import set_seed


def main():
    """Optimise les hyperparamètres des modèles de survie"""
    print("=== OPTIMISATION DES HYPERPARAMETRES ===")
    print("Objectif : Optimiser GB et RSF avec Optuna")
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

    # Extraction des données
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    # 2. Configuration de l'optimisation
    print("\n2. Configuration de l'optimisation...")
    n_trials = 200  # Optimisation intensive - environ 5-6h au total
    n_splits = 5  # Validation croisée 5-fold

    print(f"   Nombre d'essais par modèle : {n_trials}")
    print(f"   Validation croisée : {n_splits} folds")
    print(f"   Estimation du temps : 5-6 heures")
    print(
        f"   Total d'évaluations : {n_trials * 2 * n_splits} = {n_trials * 2 * n_splits}"
    )
    print(f"   (2 modèles × {n_trials} trials × {n_splits} folds)")
    print("   🔥 OPTIMISATION INTENSIVE ACTIVÉE 🔥")

    # 3. Optimisation des hyperparamètres
    print("\n3. Lancement de l'optimisation...")
    try:
        optimization_results = optimize_both_models(
            X_train=X_train, y_train=y_train, n_trials=n_trials, n_splits=n_splits
        )

        print("\n" + "=" * 50)
        print("✅ OPTIMISATION TERMINÉE AVEC SUCCÈS")
        print("=" * 50)
        print(f"Meilleur modèle : {optimization_results['best_model']}")
        print(
            f"Score CV du meilleur modèle : {optimization_results[optimization_results['best_model']]['score']:.5f}"
        )
        print(
            "\n📊 Fichiers CSV générés avec tous les résultats d'optimisation dans le dossier 'models/'"
        )

    except Exception as e:
        print(f"\nERREUR lors de l'optimisation : {e}")
        raise


if __name__ == "__main__":
    main()
