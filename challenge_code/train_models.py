#!/usr/bin/env python3
"""
Script pour l'entraînement et l'évaluation des modèles
Partie 2 du pipeline : entraînement, évaluation, prédictions
"""
import pickle
import os
from src.modeling.train import train_and_save_all_models
from src.modeling.evaluate import compare_models
from src.modeling.predict import predict_with_all_models, predict_with_best_model
from src.data.prepare import prepare_enriched_dataset, prepare_test_dataset
from src.utils.helpers import create_submission_summary
from src.visualization.plots import create_visualization_report


def train_and_predict():
    """Entraîne les modèles et génère les prédictions"""
    print("=== ENTRAÎNEMENT ET PRÉDICTION DES MODÈLES ===")

    # 1. Chargement des données préparées
    print("\n1. Chargement des données préparées...")
    if not os.path.exists("prepared_data/training_data.pkl"):
        print("Erreur: Les données préparées n'existent pas!")
        print("Veuillez d'abord exécuter: python prepare_data.py")
        return

    with open("prepared_data/training_data.pkl", "rb") as f:
        prepared_data = pickle.load(f)

    X_train = prepared_data["X_train"]
    X_test = prepared_data["X_test"]
    y_train = prepared_data["y_train"]
    y_test = prepared_data["y_test"]
    features = prepared_data["features"]
    df_enriched = prepared_data["df_enriched"]
    imputer = prepared_data["imputer"]
    data = prepared_data["data"]

    print(f"Données chargées: {X_train.shape[0]} échantillons d'entraînement")

    # 2. Entraînement des modèles
    print("\n2. Entraînement des modèles...")
    models = train_and_save_all_models(X_train, y_train)

    # 3. Évaluation des modèles
    print("\n3. Évaluation des modèles...")
    results, best_model_name = compare_models(models, X_train, y_train, X_test, y_test)

    # 4. Préparer les données de test
    print("\n4. Préparation des données de test...")
    df_test_enriched = prepare_enriched_dataset(
        data["clinical_test"],
        data["molecular_test"],
        target_df=None,
        imputer=imputer,
        is_training=False,
    )

    # Obtenir les colonnes center du train
    center_columns_train = [
        col for col in df_enriched.columns if col.startswith("center_")
    ]

    X_eval = prepare_test_dataset(df_test_enriched, features, center_columns_train)
    print(f"Dataset de test préparé: {X_eval.shape}")

    # 5. Prédictions avec tous les modèles
    print("\n5. Génération des prédictions...")
    submissions = predict_with_all_models(models, X_eval, df_test_enriched)

    # Prédiction avec le meilleur modèle
    best_filepath, best_submission = predict_with_best_model(
        models, best_model_name, X_eval, df_test_enriched
    )

    # 6. Résumé final
    print("\n6. Résumé final...")
    create_submission_summary(submissions)

    # 7. Rapport de visualisation
    print("\n7. Génération du rapport de visualisation...")
    create_visualization_report(models, results, X_train, features)

    print(f"\n=== PIPELINE TERMINÉ ===")
    print(f"Meilleur modèle: {best_model_name}")
    print(f"Fichier de soumission principal: {best_filepath}")

    return models, results, submissions, best_model_name


if __name__ == "__main__":
    train_and_predict()
