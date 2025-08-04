#!/usr/bin/env python3
"""
Script 2/3 (Version Finale) : Entraînement du Modèle de Production

Ce script charge le dataset préparé, et entraîne le MEILLEUR modèle (Random Survival Forest)
sur 100% des données disponibles pour maximiser la performance.
"""
import pandas as pd
import joblib
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from src.modeling.train import load_training_dataset_csv
import os
from src.config import RSF_PARAMS


def main():
    print("=== SCRIPT 2/3 (Mode Production) : ENTRAÎNEMENT DU MODÈLE FINAL ===")
    print("Objectif : Entraîner le meilleur modèle sur 100% des données")
    print("=" * 60)

    # 1. Charger le dataset préparé complet (X et y)
    print("\n1. Chargement du dataset d'entraînement complet...")
    X_train_path = "datasets_processed/X_train_processed.csv"
    y_train_path = "datasets_processed/y_train_processed.csv"

    try:
        # On utilise votre fonction de chargement, mais on renomme les variables
        # pour bien marquer qu'il s'agit de l'ensemble des données.
        X_full, y_full_df = load_training_dataset_csv(X_train_path, y_train_path)

        # Conversion de y au format structuré de sk-surv
        y_full_sksurv = Surv.from_arrays(
            event=y_full_df["OS_STATUS"], time=y_full_df["OS_YEARS"]
        )
        print(
            f"   Données chargées avec succès : {X_full.shape[0]} patients et {X_full.shape[1]} features."
        )

    except FileNotFoundError as e:
        print(e)
        return

    # NOTE : Pas de train_test_split. On utilise toutes les données pour l'entraînement.
    print("\n2. Utilisation de 100% des données pour l'entraînement final.")

    # 3. Définir les hyperparamètres du meilleur modèle (RSF régularisé)
    print(
        "\n3. Définition des hyperparamètres du modèle final (Random Survival Forest)..."
    )

    print(f"   Paramètres utilisés : {RSF_PARAMS}")

    # 4. Entraîner le modèle final
    print("\n4. Entraînement du modèle final en cours...")
    try:
        # Instanciation du modèle avec les paramètres finaux
        final_model = RandomSurvivalForest(**RSF_PARAMS)

        # Entraînement sur TOUTES les données
        final_model.fit(X_full, y_full_sksurv)

        print("   Entraînement terminé avec succès !")
    except Exception as e:
        print(f"ERREUR durant l'entraînement final : {e}")
        return

    # 5. Sauvegarder le modèle et les informations nécessaires pour la prédiction
    print("\n5. Sauvegarde du modèle final et du package de prédiction...")

    # On recrée une structure similaire à votre "model_package" pour que votre
    # script de prédiction n'ait pas besoin d'être modifié.
    model_package = {
        "best_model": {"model": final_model},
        "features": X_full.columns.tolist(),
        "best_model_name": "final_rsf_model",  # Nom descriptif
    }

    # Créer le dossier s'il n'existe pas
    os.makedirs("models", exist_ok=True)

    # Sauvegarder le package complet
    package_path = "models/model_package.pkl"
    joblib.dump(model_package, package_path)
    print(f"   Package de prédiction sauvegardé dans : {package_path}")

    # Sauvegarder aussi le modèle seul pour archivage
    model_path = "models/final_rsf_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"   Modèle seul sauvegardé dans : {model_path}")

    print("\n" + "=" * 60)
    print("SCRIPT 2/3 (Mode Production) TERMINÉ AVEC SUCCÈS !")
    print(
        "Le modèle final est entraîné et prêt à être utilisé par le script de prédiction."
    )
    print("Étape suivante : python 3_predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
