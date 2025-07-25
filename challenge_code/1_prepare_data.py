#!/usr/bin/env python3
"""
Script 1/3 : Preparation avancee des donnees avec preprocess.py

Ce script utilise le module preprocess.py pour effectuer une preparation
complete et avancee des donnees cliniques et moleculaires.
Il integre:
- Toutes les fonctionnalites de preprocessing disponibles
- Le feature engineering avance
- L'ajout des donnees MyVariant
- Le preprocessing personnalise pour chaque type de donnees
"""
from src.utils.helpers import set_seed, print_dataset_info
from src.data.load import load_all_data
from src.utils.preprocess_safe import (
    safe_add_myvariant_data,
    safe_create_one_hot,
    safe_count_bases_per_id,
    safe_parse_cytogenetics_v3,
    safe_log_transform,
    safe_process_outliers,
)
from src.data.prepare import (
    clean_target_data,
    prepare_features_and_target,
)
from src.visualization.plots import plot_correlation_matrix
from typing import Dict, Any
import os
import sys
import warnings
import pickle
import pandas as pd
import numpy as np


# Ajouter le repertoire racine au path pour les imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Ignorer les warnings pendant le preprocessing
warnings.filterwarnings("ignore")


def create_preprocessing_config() -> Dict[str, Any]:
    """
    Configuration des parametres de preprocessing

    Returns:
        Dict: Configuration complete du preprocessing
    """
    return {
        "clinical": ["CYTOGENETICSv3"],  # Parser cytogenetique avance
        "molecular": [
            "GENE",  # Feature engineering sur les genes
        ],
        "additional": [
            # Champs MyVariant a ajouter (deja integres)
            ["cadd", "phred"],
            ["dbnsfp", "polyphen2", "hdiv", "score"],
            ["clinvar", "clinical_significance"],
            ["gnomad_exome", "af", "af"],
        ],
        "merge": ["featuretools"],  # Utiliser featuretools pour le merge avance
        "outliers": {
            "threshold": 0.05,  # Threshold pour le traitement des outliers
            "multiplier": 1.5,
        },
        "feature_engineering": {
            "ratios": [
                # Format: "numerator/denominator"
                # Ajouter des ratios pertinents si necessaire
            ],
            "interactions": [
                # Format: "col1*col2"
                # Ajouter des interactions si necessaire
            ],
            "logs": [
                # Colonnes a transformer en log(x+1)
                "DEPTH",
                "VAF",
            ],
        },
        "aggregations": {
            "molecular": {
                "GENE": {
                    "method": "one_hot",
                    "min_count": 5,
                    "rare_label": "gene_other",
                },
                "REF": {"method": "count_bases"},
                "ALT": {"method": "count_bases"},
            }
        },
    }


def load_and_validate_data() -> Dict[str, pd.DataFrame]:
    """
    Charge et valide les donnees brutes

    Returns:
        Dict: Dictionnaire contenant tous les datasets
    """
    print("\n=== CHARGEMENT ET VALIDATION DES DONNEES ===")

    # Chargement des donnees
    data = load_all_data()
    print_dataset_info(data)

    # Validation basique
    required_files = ["clinical_train", "molecular_train", "target_train"]
    for file_key in required_files:
        if file_key not in data or data[file_key].empty:
            raise ValueError(f"Fichier requis manquant ou vide: {file_key}")

    print("✓ Tous les fichiers requis sont presents et non vides")
    return data


def preprocess_clinical_data(
    clinical_df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Preprocessing avance des donnees cliniques

    Args:
        clinical_df: DataFrame des donnees cliniques
        config: Configuration du preprocessing

    Returns:
        DataFrame preprocess
    """
    print("\n=== PREPROCESSING DES DONNEES CLINIQUES ===")

    df_processed = clinical_df.copy()

    # Parser la cytogenetique si demande
    if "CYTOGENETICSv3" in config.get("clinical", []):
        print("• Parsing cytogenetique avance (version 3)...")
        if "CYTOGENETICS" in df_processed.columns:
            df_processed = safe_parse_cytogenetics_v3(df_processed, "CYTOGENETICS")
            print(
                f"  → {len([c for c in df_processed.columns if 'cytogenetics' in c.lower()])} nouvelles features cytogenetiques"
            )
        else:
            print(" Colonne CYTOGENETICS non trouvee")

    # Nettoyage des valeurs manquantes dans les donnees cliniques
    print("• Nettoyage des valeurs manquantes...")
    initial_shape = df_processed.shape

    # Remplir les valeurs numeriques manquantes par la mediane
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isna().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)

    # Remplir les valeurs categorielles par la mode ou "unknown"
    categorical_cols = df_processed.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df_processed[col].isna().sum() > 0:
            mode_val = df_processed[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "unknown"
            df_processed[col].fillna(fill_val, inplace=True)

    print(f"  → Forme: {initial_shape} → {df_processed.shape}")
    print(f"  → Valeurs manquantes restantes: {df_processed.isna().sum().sum()}")

    return df_processed


def preprocess_molecular_data(
    molecular_df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Preprocessing avance des donnees moleculaires

    Args:
        molecular_df: DataFrame des donnees moleculaires
        config: Configuration du preprocessing

    Returns:
        DataFrame preprocess
    """
    print("\n=== PREPROCESSING DES DONNEES MOLECULAIRES ===")

    df_processed = molecular_df.copy()
    initial_shape = df_processed.shape

    # Ajouter les donnees MyVariant si demande
    additional_fields = config.get("additional", [])
    if additional_fields:
        print("• Ajout des donnees MyVariant...")
        print(f"  → Champs a ajouter: {len(additional_fields)}")
        df_processed = safe_add_myvariant_data(df_processed, additional_fields)
        myvariant_cols = [
            c
            for c in df_processed.columns
            if any(field[0] in c for field in additional_fields)
        ]
        print(f"  → {len(myvariant_cols)} nouvelles colonnes MyVariant ajoutees")

    # Feature engineering sur les genes
    if "GENE" in config.get("molecular", []):
        print("• Feature engineering sur les genes...")
        if "GENE" in df_processed.columns and "ID" in df_processed.columns:
            gene_config = (
                config.get("aggregations", {}).get("molecular", {}).get("GENE", {})
            )
            if gene_config.get("method") == "one_hot":
                gene_features = safe_create_one_hot(
                    df_processed,
                    id_col="ID",
                    ref_col="GENE",
                    min_count=gene_config.get("min_count", 5),
                    rare_label=gene_config.get("rare_label", "gene_other"),
                )
                print(f"  → {gene_features.shape[1]-1} features gene crees (one-hot)")
            else:
                gene_features = None
        else:
            print(" Colonnes GENE ou ID non trouvees")
            gene_features = None
    else:
        gene_features = None

    # Comptage des bases pour REF et ALT
    ref_features = None
    alt_features = None

    if "REF" in df_processed.columns and "ID" in df_processed.columns:
        print("• Comptage des bases REF...")
        ref_features = safe_count_bases_per_id(df_processed, id_col="ID", ref_col="REF")
        print(f"  → {ref_features.shape[1]-1} features REF crees")

    if "ALT" in df_processed.columns and "ID" in df_processed.columns:
        print("• Comptage des bases ALT...")
        alt_features = safe_count_bases_per_id(df_processed, id_col="ID", ref_col="ALT")
        print(f"  → {alt_features.shape[1]-1} features ALT crees")

    # Transformations logarithmiques
    log_cols = config.get("feature_engineering", {}).get("logs", [])
    for col in log_cols:
        if col in df_processed.columns:
            print(f"• Transformation log(x+1) pour {col}...")
            df_processed[f"{col}_log"] = safe_log_transform(df_processed, col)

    # Traitement des outliers
    if "outliers" in config:
        print("• Traitement des outliers...")
        outlier_params = config["outliers"]
        df_processed = safe_process_outliers(df_processed, **outlier_params)

    # Agregation au niveau patient
    print("• Agregation au niveau patient...")

    # Aggreger les donnees numeriques par ID (moyenne)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    # Exclure ID de l'agregation
    numeric_cols = [c for c in numeric_cols if c != "ID"]

    if len(numeric_cols) > 0:
        df_agg = (
            df_processed.groupby("ID")[numeric_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        # Aplatir les noms de colonnes
        df_agg.columns = ["ID"] + [f"{col}_{stat}" for col, stat in df_agg.columns[1:]]

        # Remplir les NaN par 0 (notamment pour std quand count=1)
        df_agg = df_agg.fillna(0)
    else:
        # Si pas de colonnes numeriques, creer un DataFrame minimal
        df_agg = df_processed[["ID"]].drop_duplicates().reset_index(drop=True)

    # Merger avec les features additionnelles si disponibles
    if gene_features is not None:
        df_agg = df_agg.merge(gene_features, on="ID", how="left")

    if ref_features is not None:
        df_agg = df_agg.merge(ref_features, on="ID", how="left")

    if alt_features is not None:
        df_agg = df_agg.merge(alt_features, on="ID", how="left")

    # Remplir les valeurs manquantes finales
    df_agg = df_agg.fillna(0)

    print(f"  → Forme finale: {initial_shape} → {df_agg.shape}")
    print(f"  → Patients uniques: {df_agg['ID'].nunique()}")

    return df_agg


def merge_and_finalize_dataset(
    clinical_processed: pd.DataFrame,
    molecular_processed: pd.DataFrame,
    target_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Merge final et finalisation du dataset

    Args:
        clinical_processed: Donnees cliniques preprocessees
        molecular_processed: Donnees moleculaires preprocessees
        target_df: Donnees target
        config: Configuration

    Returns:
        DataFrame final merge
    """
    print("\n=== MERGE ET FINALISATION ===")

    # Nettoyer les donnees target
    target_clean = clean_target_data(target_df)
    print(f"• Target nettoyee: {len(target_clean)} echantillons")

    # Merge des donnees cliniques et moleculaires
    print("• Merge clinical + molecular...")
    df_merged = clinical_processed.merge(molecular_processed, on="ID", how="inner")
    print(f"  → Forme apres merge: {df_merged.shape}")

    # Merge avec les targets
    print("• Merge avec target...")
    df_final = df_merged.merge(target_clean, on="ID", how="inner")
    print(f"  → Forme finale: {df_final.shape}")

    # Verification finale des valeurs manquantes
    nan_count = df_final.isna().sum().sum()
    if nan_count > 0:
        print(f"• Nettoyage final: {nan_count} valeurs manquantes...")
        # Remplir les valeurs numeriques par 0
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns
        df_final[numeric_cols] = df_final[numeric_cols].fillna(0)
        # Remplir les autres par "unknown"
        df_final = df_final.fillna("unknown")
        print(f"  → Valeurs manquantes restantes: {df_final.isna().sum().sum()}")

    print(f"✓ Dataset final pret: {df_final.shape}")
    return df_final


def prepare_and_save_dataset():
    """
    Pipeline principal de preparation des donnees avec preprocess.py
    """
    print("=== SCRIPT 1/3 : PREPARATION AVANCEE DES DONNEES ===")
    print("Utilisation du module preprocess.py pour un preprocessing complet")
    print("=" * 70)

    # Configuration
    set_seed()
    config = create_preprocessing_config()

    # 1. Chargement et validation
    data = load_and_validate_data()

    # 2. Preprocessing clinique
    clinical_processed = preprocess_clinical_data(data["clinical_train"], config)

    # 3. Preprocessing moleculaire
    molecular_processed = preprocess_molecular_data(data["molecular_train"], config)

    # 4. Merge et finalisation
    df_final = merge_and_finalize_dataset(
        clinical_processed, molecular_processed, data["target_train"], config
    )

    # 5. Preparation pour l'entrainement
    print("\n=== PREPARATION POUR L'ENTRAINEMENT ===")
    X_train, X_test, y_train, y_test, features = prepare_features_and_target(
        df_final, data["target_train"]
    )

    print(f"• Features d'entrainement: {X_train.shape}")
    print(f"• Features de validation: {X_test.shape}")
    print(f"• Nombre de features: {len(features)}")
    print(f"• Target d'entrainement: {y_train.shape}")
    print(f"• Target de validation: {y_test.shape}")

    # 6. Visualisations
    print("\n=== VISUALISATIONS ===")
    # Limiter aux premieres colonnes numeriques pour la matrice de correlation
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns[
        :50
    ]  # Limiter pour eviter les problemes de memoire
    if len(numeric_cols) > 0:
        plot_correlation_matrix(df_final[numeric_cols])

    # 7. Sauvegarde
    print("\n=== SAUVEGARDE ===")
    os.makedirs("datasets", exist_ok=True)

    # Dataset d'entrainement
    training_dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features,
        "config": config,  # Sauvegarder la config utilisee
        "metadata": {
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(features),
            "feature_names": list(features),
            "preprocessing_version": "preprocess_v2",
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

    file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
    print(f"• Dataset sauvegarde: {dataset_path}")
    print(f"• Taille: {file_size_mb:.2f} MB")

    # Sauvegarde du dataset enrichi
    enriched_train_csv = "datasets/enriched_train.csv"
    df_final.to_csv(enriched_train_csv, index=False)
    print(f"• Dataset enrichi: {enriched_train_csv}")

    # Resume
    summary_path = "datasets/dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== RESUME DU DATASET D'ENTRAINEMENT (PREPROCESS V2) ===\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Version preprocessing: preprocess_v2\n")
        f.write(
            f"Echantillons entrainement: {training_dataset['metadata']['n_samples_train']}\n"
        )
        f.write(
            f"Echantillons validation: {training_dataset['metadata']['n_samples_test']}\n"
        )
        f.write(f"Nombre de features: {training_dataset['metadata']['n_features']}\n\n")
        f.write("Configuration preprocessing:\n")
        for key, value in config.items():
            f.write(f"- {key}: {value}\n")
        f.write("\nFeatures disponibles:\n")
        for i, feature in enumerate(features, 1):
            f.write(f"{i:3d}. {feature}\n")

    print(f"• Resume: {summary_path}")

    # 8. Preparation des donnees de test
    print("\n=== PREPARATION DES DONNEES DE TEST ===")
    try:
        test_clinical = data.get("clinical_test")
        test_molecular = data.get("molecular_test")

        if test_clinical is not None and test_molecular is not None:
            print(f"• Donnees cliniques test: {test_clinical.shape}")
            print(f"• Donnees moleculaires test: {test_molecular.shape}")

            # Appliquer le meme preprocessing
            test_clinical_processed = preprocess_clinical_data(test_clinical, config)
            test_molecular_processed = preprocess_molecular_data(test_molecular, config)

            # Merge
            df_test_final = test_clinical_processed.merge(
                test_molecular_processed, on="ID", how="inner"
            )

            # S'assurer que les colonnes correspondent a celles d'entrainement
            train_features = [
                c for c in df_final.columns if c not in ["ID", "OS_STATUS", "OS_YEARS"]
            ]
            missing_cols = set(train_features) - set(df_test_final.columns)
            for col in missing_cols:
                df_test_final[col] = 0  # Ajouter les colonnes manquantes avec des zeros

            # Reordonner les colonnes pour correspondre
            df_test_final = df_test_final[["ID"] + train_features]

            # Sauvegarde
            enriched_test_csv = "datasets/enriched_test.csv"
            df_test_final.to_csv(enriched_test_csv, index=False)
            print(f"• Dataset test enrichi: {enriched_test_csv}")
            print(f"• Forme: {df_test_final.shape}")
        else:
            print("• Donnees de test non trouvees")

    except Exception as e:
        print(f"• Erreur lors de la preparation du test: {e}")
        print("• Les donnees de test seront preparees lors de la prediction")

    print("\n" + "=" * 70)
    print("✓ SCRIPT 1/3 TERMINE AVEC SUCCES !")
    print("Dataset preprocess avec le module preprocess.py")
    print("Prochaine etape: python 2_train_models.py")
    print("=" * 70)

    return training_dataset


if __name__ == "__main__":
    prepare_and_save_dataset()
