#!/usr/bin/env python3
"""
Script 1/3 : Preparation des donnees pour la modelisation de survie

Ce script nettoie, enrichit et prepare les donnees cliniques et moleculaires
pour l'entrainement de modeles de survie.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings

# Ajouter le repertoire racine au path pour les imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.helpers import set_seed, print_dataset_info
from src.data.load import load_all_data
from src.data.prepare import (
    clean_target_data,
    prepare_enriched_dataset,
    prepare_features_and_target,
)
from src.visualization.plots import plot_correlation_matrix
import pickle

# Ignorer les warnings pendant le preprocessing
warnings.filterwarnings("ignore")


def prepare_and_save_dataset():
    """
    Preparer le dataset d'entrainement enrichi avec imputation avancee

    Charge, nettoie et prepare les donnees d'entrainement et de test
    Sauvegarde un dataset pret pour l'entrainement
    """
    print("=== SCRIPT 1/3 : PREPARATION DES DONNEES ===")
    print("Objectif : Creer un dataset pret pour l'entrainement")
    print("=" * 60)

    # 1. Configuration initiale
    set_seed()

    # 2. Chargement des donnees brutes
    print("\n 1. Chargement des donnees brutes...")
    data = load_all_data()
    print_dataset_info(data)

    # 3. Nettoyage et preparation
    print("\n 2. Nettoyage des donnees...")

    # Nettoyer les donnees target
    target_clean = clean_target_data(data["target_train"])
    print(f"  Target nettoyee : {len(target_clean)} echantillons")

    # Preparer le dataset d'entrainement enrichi avec imputation avancee
    df_enriched, imputer = prepare_enriched_dataset(
        data["clinical_train"],
        data["molecular_train"],
        target_clean,
        advanced_imputation_method="medical",
    )

    # CORRECTION: Eliminer toutes les valeurs manquantes restantes
    print(f"   Dataset enrichi cree : {df_enriched.shape}")
    nan_count_before = df_enriched.isnull().sum().sum()
    if nan_count_before > 0:
        print(f"   Correction de {nan_count_before} valeurs NaN restantes...")
        numeric_cols = df_enriched.select_dtypes(include=[np.number]).columns
        df_enriched[numeric_cols] = df_enriched[numeric_cols].fillna(0)
        df_enriched = df_enriched.fillna(0)  # Pour toutes les autres colonnes
        nan_count_after = df_enriched.isnull().sum().sum()
        print(f"   Imputation finale: {nan_count_after} NaN restant")
    else:
        print("   Aucune valeur manquante detectee")

    # Preparer les features et targets pour l'entrainement
    X_train, X_test, y_train, y_test, features = prepare_features_and_target(
        df_enriched, target_clean
    )

    print("\n 3. Statistiques du dataset prepare :")
    print(f"   • Features d'entrainement: {X_train.shape}")
    print(f"   • Features de validation: {X_test.shape}")
    print(f"   • Nombre de features: {len(features)}")
    print(f"   • Target d'entrainement: {y_train.shape}")
    print(f"   • Target de validation: {y_test.shape}")

    # 4. Visualisations exploratoires
    print("\n 4. Visualisations exploratoires...")
    plot_correlation_matrix(df_enriched)

    # 5. Sauvegarde du dataset complet
    print("\n 5. Sauvegarde du dataset...")

    # Creer le repertoire de donnees preparees
    os.makedirs("datasets", exist_ok=True)

    # Dataset d'entrainement
    training_dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features,
        "metadata": {
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(features),
            "imputer": imputer,
            "feature_names": list(features),
            "target_info": {
                "train_events": y_train["OS_STATUS"].sum(),
                "test_events": y_test["OS_STATUS"].sum(),
                "train_event_rate": y_train["OS_STATUS"].mean(),
                "test_event_rate": y_test["OS_STATUS"].mean(),
            },
        },
    }

    # Sauvegarde du dataset
    dataset_path = "datasets/training_dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(training_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculer la taille du fichier
    file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
    print(f"   Dataset sauvegarde : {dataset_path}")
    print(f"   Taille du fichier : {file_size_mb:.2f} MB")

    # Sauvegarde du resume
    summary_path = "datasets/dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== RESUME DU DATASET D'ENTRAINEMENT ===\n\n")
        f.write(
            f"Date de creation : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            f"Echantillons d'entrainement : {training_dataset['metadata']['n_samples_train']}\n"
        )
        f.write(
            f"Echantillons de validation : {training_dataset['metadata']['n_samples_test']}\n"
        )
        f.write(f"Nombre de features : {training_dataset['metadata']['n_features']}\n")
        f.write("\nFeatures disponibles :\n")
        for i, feature in enumerate(features, 1):
            f.write(f"{i:3d}. {feature}\n")

    print(f"   Resume sauvegarde : {summary_path}")

    # Sauvegarde du dataset enrichi au format CSV
    enriched_train_csv = "datasets/enriched_train.csv"
    # On ajoute la target au DataFrame enrichi
    df_enriched_with_target = df_enriched.copy()
    if "ID" in df_enriched_with_target.columns and "ID" in target_clean.columns:
        df_enriched_with_target = df_enriched_with_target.merge(
            target_clean[["ID", "OS_STATUS", "OS_YEARS"]], on="ID", how="left"
        )
    df_enriched_with_target.to_csv(enriched_train_csv, index=False)
    print(f"   Dataset enrichi d'entrainement exporte : {enriched_train_csv}")

    print("\n" + "=" * 60)
    print("SCRIPT 1/3 TERMINE AVEC SUCCES !")
    print("Dataset pret pour l'entrainement")
    print("Prochaine etape : python 2_train_models.py")
    print("=" * 60)

    return training_dataset


if __name__ == "__main__":
    prepare_and_save_dataset()
