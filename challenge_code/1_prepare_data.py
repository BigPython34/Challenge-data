#!/usr/bin/env python3
"""
Script 1/3 : Préparation des données
Charge, nettoie et prépare les données d'entraînement et de test
Sauvegarde un dataset prêt pour l'entraînement
"""
import pickle
import os
import pandas as pd
from src.config import SEED
from src.data.load import load_all_data
from src.data.prepare import (
    clean_target_data,
    prepare_enriched_dataset,
    prepare_features_and_target,
)
from src.utils.helpers import set_seed, print_dataset_info
from src.visualization.plots import plot_correlation_matrix


def prepare_and_save_dataset():
    """Prépare et sauvegarde le dataset complet pour l'entraînement"""
    print("=== SCRIPT 1/3 : PRÉPARATION DES DONNÉES ===")
    print("Objectif : Créer un dataset prêt pour l'entraînement")
    print("=" * 60)

    # 1. Configuration initiale
    set_seed()

    # 2. Chargement des données brutes
    print("\n📂 1. Chargement des données brutes...")
    data = load_all_data()
    print_dataset_info(data)

    # 3. Nettoyage et préparation
    print("\n🧹 2. Nettoyage des données...")

    # Nettoyer les données target
    target_clean = clean_target_data(data["target_train"])
    print(f"   ✅ Target nettoyée : {len(target_clean)} échantillons")

    # Préparer le dataset d'entraînement enrichi
    df_enriched, imputer = prepare_enriched_dataset(
        data["clinical_train"], data["molecular_train"], target_clean
    )
    print(f"   ✅ Dataset enrichi créé : {df_enriched.shape}")

    # Préparer les features et target pour l'entraînement
    X_train, X_test, y_train, y_test, features = prepare_features_and_target(
        df_enriched, target_clean
    )

    print(f"\n📊 3. Statistiques du dataset préparé :")
    print(f"   • Features d'entraînement: {X_train.shape}")
    print(f"   • Features de validation: {X_test.shape}")
    print(f"   • Nombre de features: {len(features)}")
    print(f"   • Target d'entraînement: {y_train.shape}")
    print(f"   • Target de validation: {y_test.shape}")

    # 4. Visualisations exploratoires
    print("\n📈 4. Visualisations exploratoires...")
    plot_correlation_matrix(df_enriched)

    # 5. Sauvegarde du dataset complet
    print("\n💾 5. Sauvegarde du dataset...")

    # Créer le répertoire de données préparées
    os.makedirs("datasets", exist_ok=True)

    # Dataset d'entraînement
    training_dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features,
        "df_enriched": df_enriched,
        "imputer": imputer,
        "raw_data": data,  # données brutes pour les prédictions finales
        "target_clean": target_clean,
        "metadata": {
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(features),
            "feature_names": features,
            "preparation_date": pd.Timestamp.now().isoformat(),
        },
    }

    # Sauvegarder le dataset
    dataset_path = "datasets/training_dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(training_dataset, f)

    print(f"   ✅ Dataset sauvegardé : {dataset_path}")
    print(
        f"   📁 Taille du fichier : {os.path.getsize(dataset_path) / 1024 / 1024:.2f} MB"
    )

    # Sauvegarder aussi un résumé lisible
    summary_path = "datasets/dataset_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== RÉSUMÉ DU DATASET PRÉPARÉ ===\n")
        f.write(
            f"Date de création : {training_dataset['metadata']['preparation_date']}\n"
        )
        f.write(
            f"Échantillons d'entraînement : {training_dataset['metadata']['n_samples_train']}\n"
        )
        f.write(
            f"Échantillons de validation : {training_dataset['metadata']['n_samples_test']}\n"
        )
        f.write(f"Nombre de features : {training_dataset['metadata']['n_features']}\n")
        f.write(f"\nFeatures disponibles :\n")
        for i, feature in enumerate(features, 1):
            f.write(f"{i:3d}. {feature}\n")

    print(f"   ✅ Résumé sauvegardé : {summary_path}")

    # Sauvegarde du dataset enrichi au format CSV
    enriched_train_csv = "datasets/enriched_train.csv"
    # On ajoute la target au DataFrame enrichi
    df_enriched_with_target = df_enriched.copy()
    if "ID" in df_enriched_with_target.columns and "ID" in target_clean.columns:
        df_enriched_with_target = df_enriched_with_target.merge(
            target_clean[["ID", "OS_STATUS", "OS_YEARS"]], on="ID", how="left"
        )
    df_enriched_with_target.to_csv(enriched_train_csv, index=False)
    print(f"   ✅ Dataset enrichi d'entraînement exporté : {enriched_train_csv}")

    print("\n" + "=" * 60)
    print("🎉 SCRIPT 1/3 TERMINÉ AVEC SUCCÈS !")
    print("✅ Dataset prêt pour l'entraînement")
    print("➡️  Prochaine étape : python 2_train_models.py")
    print("=" * 60)

    return training_dataset


if __name__ == "__main__":
    prepare_and_save_dataset()
