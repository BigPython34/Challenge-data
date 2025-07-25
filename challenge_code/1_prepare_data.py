#!/usr/bin/env python3
"""
Script 1/3 : Préparation avancée des données avec architecture modulaire

Ce script utilise notre nouvelle architecture modulaire pour effectuer une préparation
complète et avancée des données cliniques et moléculaires pour l'analyse de survie AML.

Architecture modulaire :
- src.data.load : Chargement des données
- src.data.data_cleaning : Nettoyage et imputation intelligente
- src.data.features : Feature engineering clinique et moléculaire
- src.data.prepare : Orchestration et intégration finale

Fonctionnalités :
- Nettoyage intelligent avec stratégies d'imputation médicales
- Feature engineering basé sur les guidelines ELN 2022
- Extraction de features cytogénétiques avancées
- Features moléculaires avec patterns de co-mutations
- Scores de risque intégrés selon ELN 2022
"""

import os
import sys
import warnings
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

# Ajouter le répertoire racine au path pour les imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.load import load_all_data
from src.data.prepare import prepare_survival_dataset
from src.data.data_cleaning import ImputationStrategy
from src.visualization.plots import plot_correlation_matrix
from src.config import SEED

# Ignorer les warnings pendant le preprocessing
warnings.filterwarnings("ignore")


def set_seed(seed: int = SEED) -> None:
    """Définit la graine aléatoire pour la reproductibilité."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def print_dataset_info(data: Dict[str, pd.DataFrame]) -> None:
    """Affiche les informations sur les datasets chargés."""
    print("\n=== INFORMATIONS SUR LES DATASETS ===")
    for name, df in data.items():
        if df is not None and not df.empty:
            print(f"• {name}: {df.shape}")
            if "ID" in df.columns:
                print(f"  - IDs uniques: {df['ID'].nunique()}")
            print(
                f"  - Colonnes: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
            )
        else:
            print(f"• {name}: Non disponible ou vide")


def create_data_preparation_config() -> Dict[str, Any]:
    """
    Configuration avancée pour la préparation des données.

    Returns:
        Dict: Configuration complète du pipeline de données
    """
    return {
        "pipeline": {
            "test_size": 0.3,  # Plus grande validation pour plus de robustesse
            "use_advanced_features": True,
            "include_molecular_burden": True,
            "include_cytogenetic_features": True,
            "include_interaction_features": True,
        },
        "imputation": {
            "strategy": "medical_informed",  # medical_informed, median, mean, knn, iterative, regression
            "fill_missing_with_zero": ["mutations", "molecular"],
            "fill_missing_with_median": ["clinical_numeric"],
            "fill_missing_with_mode": ["clinical_categorical"],
        },
        "feature_engineering": {
            "clinical": {
                "create_ratios": True,
                "create_thresholds": True,
                "create_composite_scores": True,
                "create_log_transforms": True,
            },
            "molecular": {
                "extract_binary_mutations": True,
                "extract_vaf_features": True,
                "extract_mutation_types": True,
                "extract_comutation_patterns": True,
                "extract_pathway_alterations": True,
            },
            "cytogenetic": {
                "extract_eln2022_abnormalities": True,
                "calculate_complexity": True,
                "extract_chromosome_features": True,
                "calculate_risk_scores": True,
            },
            "integrated": {
                "create_eln2022_risk_scores": True,
                "create_interaction_features": True,
                "create_comprehensive_scores": True,
            },
        },
        "quality_control": {
            "remove_low_variance_features": True,
            "variance_threshold": 0.01,
            "remove_highly_correlated": True,
            "correlation_threshold": 0.95,
            "handle_outliers": True,
            "outlier_method": "clip",  # clip, remove, transform
        },
        "output": {
            "save_datasets": True,
            "save_metadata": True,
            "save_feature_importance": True,
            "create_visualizations": True,
            "datasets_dir": "datasets",
            "models_dir": "models",
        },
    }


def validate_input_data(data: Dict[str, pd.DataFrame]) -> None:
    """
    Valide les données d'entrée.

    Args:
        data: Dictionnaire contenant tous les datasets

    Raises:
        ValueError: Si les données ne respectent pas les prérequis
    """
    print("\n=== VALIDATION DES DONNÉES D'ENTRÉE ===")

    # Vérifier la présence des fichiers requis
    required_files = ["clinical_train", "molecular_train", "target_train"]
    for file_key in required_files:
        if file_key not in data or data[file_key].empty:
            raise ValueError(f"Fichier requis manquant ou vide: {file_key}")

    # Vérifier les colonnes essentielles
    clinical_required_cols = ["ID"]
    molecular_required_cols = ["ID", "GENE"]
    target_required_cols = ["ID", "OS_STATUS", "OS_YEARS"]

    for col in clinical_required_cols:
        if col not in data["clinical_train"].columns:
            raise ValueError(f"Colonne manquante dans clinical_train: {col}")

    for col in molecular_required_cols:
        if col not in data["molecular_train"].columns:
            raise ValueError(f"Colonne manquante dans molecular_train: {col}")

    for col in target_required_cols:
        if col not in data["target_train"].columns:
            raise ValueError(f"Colonne manquante dans target_train: {col}")

    # Vérifier la cohérence des IDs
    clinical_ids = set(data["clinical_train"]["ID"].unique())
    molecular_ids = set(data["molecular_train"]["ID"].unique())
    target_ids = set(data["target_train"]["ID"].unique())

    common_ids = clinical_ids & molecular_ids & target_ids
    if len(common_ids) == 0:
        raise ValueError("Aucun ID commun entre clinical, molecular et target")

    print(f"✓ Fichiers requis présents et valides")
    print(f"✓ Colonnes essentielles présentes")
    print(f"✓ IDs cohérents: {len(common_ids)} patients communs")
    print(f"  - Clinical: {len(clinical_ids)} patients")
    print(f"  - Molecular: {len(molecular_ids)} patients")
    print(f"  - Target: {len(target_ids)} patients")


def prepare_training_data(
    data: Dict[str, pd.DataFrame], config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Prépare les données d'entraînement avec notre pipeline modulaire.

    Args:
        data: Données brutes chargées
        config: Configuration du pipeline

    Returns:
        Tuple: (dataset_train, dataset_test, metadata)
    """
    print("\n=== PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT ===")

    # Conversion de la stratégie d'imputation
    strategy_map = {
        "medical_informed": ImputationStrategy.MEDICAL_INFORMED,
        "median": ImputationStrategy.MEDIAN,
        "mean": ImputationStrategy.MEAN,
        "knn": ImputationStrategy.KNN,
        "iterative": ImputationStrategy.ITERATIVE,
        "regression": ImputationStrategy.REGRESSION,
    }

    imputation_strategy = strategy_map.get(
        config["imputation"]["strategy"], ImputationStrategy.MEDICAL_INFORMED
    )

    print(f"• Stratégie d'imputation: {config['imputation']['strategy']}")
    print(f"• Features avancées: {config['pipeline']['use_advanced_features']}")
    print(f"• Taille du test: {config['pipeline']['test_size']}")

    # Utilisation du pipeline principal
    dataset_train, dataset_test, pipeline_metadata = prepare_survival_dataset(
        clinical_df=data["clinical_train"],
        molecular_df=data["molecular_train"],
        target_df=data["target_train"],
        test_size=config["pipeline"]["test_size"],
        use_advanced_features=config["pipeline"]["use_advanced_features"],
        imputation_strategy=imputation_strategy,
    )

    print(f"✓ Dataset d'entraînement préparé: {dataset_train.shape}")
    print(f"✓ Dataset de validation préparé: {dataset_test.shape}")
    print(f"✓ Features créées: {len(pipeline_metadata.get('feature_names', []))}")

    return dataset_train, dataset_test, pipeline_metadata


def prepare_test_data(
    data: Dict[str, pd.DataFrame], pipeline_metadata: Dict, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Prépare les données de test en appliquant les mêmes transformations SANS supprimer de patients.

    Args:
        data: Données brutes incluant les données de test
        pipeline_metadata: Métadonnées du pipeline d'entraînement
        config: Configuration du pipeline

    Returns:
        DataFrame: Données de test préparées
    """
    print("\n=== PRÉPARATION DES DONNÉES DE TEST ===")

    if "clinical_test" not in data or "molecular_test" not in data:
        print("⚠ Données de test non disponibles")
        return None

    try:
        print(f"• Données cliniques test: {data['clinical_test'].shape}")
        print(f"• Données moleculaires test: {data['molecular_test'].shape}")

        # IMPORTANT: Les données de test n'ont PAS de target car c'est ce qu'on veut prédire
        # Utilisation directe des modules de feature engineering SANS target
        from src.data.features import (
            create_clinical_features,
            extract_cytogenetic_risk_features,
            extract_molecular_risk_features,
            create_molecular_burden_features,
            combine_all_features,
        )
        from src.data.data_cleaning import intelligent_clinical_imputation

        # 1. Feature engineering clinique
        print("• Feature engineering clinique...")
        clinical_features = create_clinical_features(data["clinical_test"])

        # 2. Feature engineering cytogénétique
        print("• Feature engineering cytogénétique...")
        cyto_features = extract_cytogenetic_risk_features(data["clinical_test"])

        # 3. Feature engineering moléculaire
        print("• Feature engineering moléculaire...")
        molecular_features = extract_molecular_risk_features(
            data["clinical_test"], data["molecular_test"]
        )

        # 4. Features de mutation burden
        print("• Features mutation burden...")
        burden_features = create_molecular_burden_features(data["molecular_test"])

        # 5. Combinaison de toutes les features
        print("• Combinaison des features...")
        dataset_test = combine_all_features(
            clinical_features, molecular_features, burden_features, cyto_features
        )

        # 6. Imputation intelligente
        print("• Imputation intelligente...")
        dataset_test, imputation_metadata = intelligent_clinical_imputation(
            dataset_test, strategy=ImputationStrategy.MEDICAL_INFORMED
        )

        print(f"✓ Données de test préparées: {dataset_test.shape}")
        print(
            f"✓ Aucun patient supprimé (conservation de tous les {data['clinical_test']['ID'].nunique()} patients)"
        )
        print("✓ Aucun target créé - données prêtes pour prédiction")

        return dataset_test

    except Exception as e:
        print(f"⚠ Erreur lors de la préparation des données de test: {e}")
        import traceback

        traceback.print_exc()
        return None


def apply_quality_control(
    dataset_train: pd.DataFrame, dataset_test: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Applique le contrôle qualité sur les datasets.

    Args:
        dataset_train: Dataset d'entraînement
        dataset_test: Dataset de test/validation
        config: Configuration QC

    Returns:
        Tuple: (dataset_train_qc, dataset_test_qc, qc_metadata)
    """
    print("\n=== CONTRÔLE QUALITÉ ===")

    qc_config = config.get("quality_control", {})
    qc_metadata = {"removed_features": [], "transformations": []}

    # Séparer features et target
    feature_cols = [
        col
        for col in dataset_train.columns
        if col
        not in ["ID", "OS_STATUS", "OS_YEARS", "y_survival"]  # Exclure y_survival !
    ]

    # Sélectionner seulement les features numériques pour le contrôle qualité
    numeric_feature_cols = []
    for col in feature_cols:
        if dataset_train[col].dtype in ["int64", "float64", "int32", "float32"]:
            numeric_feature_cols.append(col)

    train_features = dataset_train[numeric_feature_cols]
    train_target = dataset_train[["ID", "OS_STATUS", "OS_YEARS"]]

    if dataset_test is not None:
        # Vérifier si les données de test ont des colonnes target ou pas
        if "OS_STATUS" in dataset_test.columns and "OS_YEARS" in dataset_test.columns:
            # Dataset de validation (avec target)
            test_features = dataset_test[numeric_feature_cols]
            test_target = dataset_test[["ID", "OS_STATUS", "OS_YEARS"]]
        else:
            # Dataset de test final (sans target, pour prédiction)
            test_numeric_cols = [
                col for col in numeric_feature_cols if col in dataset_test.columns
            ]
            test_features = dataset_test[test_numeric_cols]
            test_target = dataset_test[["ID"]]  # Seulement l'ID
    else:
        test_features = None
        test_target = None

    initial_features = len(numeric_feature_cols)
    print(
        f"• Features numériques sélectionnées: {initial_features}/{len(feature_cols)}"
    )

    # 1. Supprimer les features à faible variance
    if qc_config.get("remove_low_variance_features", False):
        from sklearn.feature_selection import VarianceThreshold

        variance_threshold = qc_config.get("variance_threshold", 0.01)
        selector = VarianceThreshold(threshold=variance_threshold)

        train_features_filtered = pd.DataFrame(
            selector.fit_transform(train_features),
            columns=train_features.columns[selector.get_support()],
            index=train_features.index,
        )

        removed_features = list(
            set(feature_cols) - set(train_features_filtered.columns)
        )
        qc_metadata["removed_features"].extend(removed_features)

        if test_features is not None:
            test_features_filtered = pd.DataFrame(
                selector.transform(test_features),
                columns=train_features_filtered.columns,
                index=test_features.index,
            )
        else:
            test_features_filtered = None

        train_features = train_features_filtered
        test_features = test_features_filtered

        print(
            f"• Variance: supprimé {len(removed_features)} features (seuil: {variance_threshold})"
        )

    # 2. Gérer les features hautement corrélées
    if qc_config.get("remove_highly_correlated", False):
        correlation_threshold = qc_config.get("correlation_threshold", 0.95)

        # Calculer la matrice de corrélation
        corr_matrix = train_features.corr().abs()

        # Trouver les paires hautement corrélées
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Sélectionner les features à supprimer
        highly_corr_features = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > correlation_threshold)
        ]

        # Supprimer les features corrélées
        remaining_features = [
            col for col in train_features.columns if col not in highly_corr_features
        ]

        train_features = train_features[remaining_features]
        if test_features is not None:
            test_features = test_features[remaining_features]

        qc_metadata["removed_features"].extend(highly_corr_features)

        print(
            f"• Corrélation: supprimé {len(highly_corr_features)} features (seuil: {correlation_threshold})"
        )

    # 3. Traitement des outliers
    if qc_config.get("handle_outliers", False):
        outlier_method = qc_config.get("outlier_method", "clip")

        if outlier_method == "clip":
            # Clipping à 1er et 99ème percentiles
            numeric_cols = train_features.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                q01 = train_features[col].quantile(0.01)
                q99 = train_features[col].quantile(0.99)

                train_features[col] = train_features[col].clip(lower=q01, upper=q99)
                if test_features is not None:
                    test_features[col] = test_features[col].clip(lower=q01, upper=q99)

            qc_metadata["transformations"].append(
                f"outlier_clipping_{len(numeric_cols)}_features"
            )
            print(
                f"• Outliers: clipping appliqué à {len(numeric_cols)} features numériques"
            )

    # Reconstruction des datasets avec toutes les features originales
    # Combiner les features numériques traitées avec les features non-numériques
    non_numeric_cols = [col for col in feature_cols if col not in numeric_feature_cols]

    if non_numeric_cols:
        train_non_numeric = dataset_train[non_numeric_cols]

        # Pour les données de test, vérifier quelles colonnes existent
        if test_features is not None and dataset_test is not None:
            available_non_numeric_cols = [
                col for col in non_numeric_cols if col in dataset_test.columns
            ]
            test_non_numeric = (
                dataset_test[available_non_numeric_cols]
                if available_non_numeric_cols
                else None
            )
        else:
            test_non_numeric = None

        # Combiner toutes les features
        all_train_features = pd.concat([train_features, train_non_numeric], axis=1)

        if test_features is not None and test_non_numeric is not None:
            all_test_features = pd.concat([test_features, test_non_numeric], axis=1)
        elif test_features is not None:
            all_test_features = test_features
        else:
            all_test_features = None
    else:
        all_train_features = train_features
        all_test_features = test_features

    dataset_train_qc = pd.concat([train_target, all_train_features], axis=1)
    dataset_test_qc = (
        pd.concat([test_target, all_test_features], axis=1)
        if all_test_features is not None and test_target is not None
        else None
    )

    final_features = len(all_train_features.columns)
    removed_total = len(feature_cols) - final_features

    print(f"✓ Contrôle qualité terminé")
    print(f"  - Features totales initiales: {len(feature_cols)}")
    print(f"  - Features numériques traitées: {len(train_features.columns)}")
    print(
        f"  - Features non-numériques conservées: {len(non_numeric_cols) if non_numeric_cols else 0}"
    )
    print(f"  - Features finales: {final_features}")

    return dataset_train_qc, dataset_test_qc, qc_metadata


def create_data_visualizations(
    dataset_train: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """
    Crée les visualisations des données.

    Args:
        dataset_train: Dataset d'entraînement
        config: Configuration
    """
    print("\n=== CRÉATION DES VISUALISATIONS ===")

    if not config.get("output", {}).get("create_visualizations", True):
        print("• Visualisations désactivées")
        return

    try:
        # Sélectionner les features numériques pour la matrice de corrélation
        numeric_cols = dataset_train.select_dtypes(include=[np.number]).columns
        # Exclure les colonnes target et limiter le nombre pour éviter les problèmes de mémoire
        viz_cols = [
            col for col in numeric_cols if col not in ["OS_STATUS", "OS_YEARS"]
        ][:50]

        if len(viz_cols) > 0:
            print(f"• Matrice de corrélation: {len(viz_cols)} features")
            plot_correlation_matrix(dataset_train[viz_cols])
            print("✓ Matrice de corrélation sauvegardée")
        else:
            print("• Aucune feature numérique pour la visualisation")

    except Exception as e:
        print(f"⚠ Erreur lors de la création des visualisations: {e}")


def save_datasets_and_metadata(
    dataset_train: pd.DataFrame,
    dataset_test: pd.DataFrame,
    dataset_test_final: pd.DataFrame,
    pipeline_metadata: Dict,
    qc_metadata: Dict,
    config: Dict[str, Any],
) -> None:
    """
    Sauvegarde les datasets et métadonnées.

    Args:
        dataset_train: Dataset d'entraînement
        dataset_test: Dataset de validation
        dataset_test_final: Dataset de test final
        pipeline_metadata: Métadonnées du pipeline
        qc_metadata: Métadonnées du contrôle qualité
        config: Configuration
    """
    print("\n=== SAUVEGARDE ===")

    # Créer les répertoires
    datasets_dir = Path(config["output"]["datasets_dir"])
    datasets_dir.mkdir(exist_ok=True)

    # Préparer les données pour la sauvegarde
    feature_cols = [
        col
        for col in dataset_train.columns
        if col not in ["ID", "OS_STATUS", "OS_YEARS"]
    ]

    # Séparer les features numériques et non-numériques pour éviter les erreurs
    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if dataset_train[col].dtype in ["int64", "float64", "int32", "float32", "bool"]:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    print(f"• Features numériques: {len(numeric_features)}")
    print(f"• Features catégorielles: {len(categorical_features)}")

    # Encoder les features catégorielles si nécessaire (exclure les colonnes target)
    if categorical_features:
        print(f"• Encodage des features catégorielles: {categorical_features}")
        from sklearn.preprocessing import LabelEncoder

        dataset_train_encoded = dataset_train.copy()
        dataset_test_encoded = dataset_test.copy()

        for col in categorical_features:
            # Ne pas encoder les colonnes target
            if col in ["OS_STATUS", "OS_YEARS", "ID"]:
                continue

            le = LabelEncoder()

            # Combiner les valeurs train et test pour un encodage cohérent
            all_values = pd.concat([dataset_train[col], dataset_test[col]]).astype(str)
            le.fit(all_values)

            dataset_train_encoded[col] = le.transform(dataset_train[col].astype(str))
            dataset_test_encoded[col] = le.transform(dataset_test[col].astype(str))

        # Utiliser les datasets encodés
        dataset_train = dataset_train_encoded
        dataset_test = dataset_test_encoded

        # Mettre à jour les types de features
        feature_cols = numeric_features + categorical_features

    # Séparer features et targets
    X_train = dataset_train[feature_cols]

    # Créer le structured array pour la survie
    from sksurv.util import Surv

    # Assurer que OS_STATUS est bien en 0/1 (int) et non True/False (bool)
    survival_train = dataset_train[["OS_STATUS", "OS_YEARS"]].copy()
    survival_train["OS_STATUS"] = survival_train["OS_STATUS"].astype(int)

    y_train = Surv.from_dataframe("OS_STATUS", "OS_YEARS", survival_train)

    X_test = dataset_test[feature_cols]

    # Pour les données de test, vérifier si on a les colonnes target
    if "OS_STATUS" in dataset_test.columns and "OS_YEARS" in dataset_test.columns:
        # Dataset de validation avec target
        survival_test = dataset_test[["OS_STATUS", "OS_YEARS"]].copy()
        survival_test["OS_STATUS"] = survival_test["OS_STATUS"].astype(int)
        y_test = Surv.from_dataframe("OS_STATUS", "OS_YEARS", survival_test)
    else:
        # Dataset de test sans target - créer un placeholder
        y_test = None

    # Dataset complet d'entraînement
    training_dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": feature_cols,
        "config": config,
        "pipeline_metadata": pipeline_metadata,
        "qc_metadata": qc_metadata,
        "metadata": {
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "preprocessing_version": "modular_v3",
            "creation_date": pd.Timestamp.now().isoformat(),
            "target_info": {
                "train_events": (
                    int(np.sum(y_train["OS_STATUS"])) if y_train is not None else 0
                ),
                "test_events": (
                    int(np.sum(y_test["OS_STATUS"])) if y_test is not None else 0
                ),
                "train_event_rate": (
                    float(np.mean(y_train["OS_STATUS"])) if y_train is not None else 0.0
                ),
                "test_event_rate": (
                    float(np.mean(y_test["OS_STATUS"])) if y_test is not None else 0.0
                ),
                "train_median_time": (
                    float(np.median(y_train["OS_YEARS"]))
                    if y_train is not None
                    else 0.0
                ),
                "test_median_time": (
                    float(np.median(y_test["OS_YEARS"])) if y_test is not None else 0.0
                ),
            },
        },
    }

    # Sauvegarde du dataset principal
    dataset_path = datasets_dir / "training_dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(training_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
    print(f"• Dataset principal: {dataset_path} ({file_size_mb:.2f} MB)")

    # Sauvegarde des datasets CSV
    enriched_train_csv = datasets_dir / "enriched_train.csv"
    dataset_train.to_csv(enriched_train_csv, index=False)
    print(f"• Dataset train CSV: {enriched_train_csv}")

    # Sauvegarde du dataset de test final si disponible
    if dataset_test_final is not None:
        # Encoder les features catégorielles du dataset de test final si nécessaire
        dataset_test_final_encoded = dataset_test_final.copy()

        if categorical_features:
            print(f"• Encodage des features catégorielles pour le test final...")
            from sklearn.preprocessing import LabelEncoder

            for col in categorical_features:
                if col in dataset_test_final_encoded.columns:
                    le = LabelEncoder()

                    # Utiliser les mêmes valeurs de référence que pour l'entraînement
                    all_train_values = dataset_train[col].astype(str).unique()
                    le.fit(all_train_values)

                    # Traiter les valeurs inconnues
                    test_values = dataset_test_final_encoded[col].astype(str)
                    known_mask = test_values.isin(all_train_values)

                    dataset_test_final_encoded[col] = 0  # Valeur par défaut
                    dataset_test_final_encoded.loc[known_mask, col] = le.transform(
                        test_values[known_mask]
                    )

        # Sauvegarder le dataset de test final encodé
        enriched_test_csv = datasets_dir / "enriched_test.csv"
        dataset_test_final_encoded.to_csv(enriched_test_csv, index=False)
        print(f"• Dataset test CSV: {enriched_test_csv}")

        # Sauvegarder aussi pour les prédictions
        test_features_path = datasets_dir / "test_features.pkl"

        # Sélectionner seulement les features qui existent dans le dataset de test
        available_features = [
            col for col in feature_cols if col in dataset_test_final_encoded.columns
        ]

        with open(test_features_path, "wb") as f:
            pickle.dump(
                {
                    "X_test": dataset_test_final_encoded[available_features],
                    "feature_names": available_features,
                    "ids": dataset_test_final_encoded["ID"],
                },
                f,
            )
        print(f"• Features test: {test_features_path}")
        print(
            f"• Features disponibles pour test: {len(available_features)}/{len(feature_cols)}"
        )

    # Sauvegarde du résumé détaillé
    summary_path = datasets_dir / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== RÉSUMÉ DATASET MODULAIRE V3 ===\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Version: modular_v3\n\n")

        f.write("=== DONNÉES ===\n")
        f.write(
            f"Échantillons entraînement: {training_dataset['metadata']['n_samples_train']}\n"
        )
        f.write(
            f"Échantillons validation: {training_dataset['metadata']['n_samples_test']}\n"
        )
        f.write(f"Features finales: {training_dataset['metadata']['n_features']}\n")
        f.write(
            f"Événements train: {training_dataset['metadata']['target_info']['train_events']}\n"
        )
        f.write(
            f"Taux événements train: {training_dataset['metadata']['target_info']['train_event_rate']:.3f}\n"
        )
        if training_dataset["metadata"]["target_info"]["test_events"] > 0:
            f.write(
                f"Événements test: {training_dataset['metadata']['target_info']['test_events']}\n"
            )
            f.write(
                f"Taux événements test: {training_dataset['metadata']['target_info']['test_event_rate']:.3f}\n"
            )
        else:
            f.write("Test: Pas de target (données pour prédiction)\n")
        f.write("\n")

        f.write("=== PIPELINE ===\n")
        f.write(f"Imputation: {config['imputation']['strategy']}\n")
        f.write(f"Features avancées: {config['pipeline']['use_advanced_features']}\n")
        f.write(f"Test size: {config['pipeline']['test_size']}\n\n")

        f.write("=== CONTRÔLE QUALITÉ ===\n")
        if qc_metadata.get("removed_features"):
            f.write(f"Features supprimées: {len(qc_metadata['removed_features'])}\n")
        if qc_metadata.get("transformations"):
            f.write(f"Transformations: {', '.join(qc_metadata['transformations'])}\n")
        f.write("\n")

        f.write("=== FEATURES FINALES ===\n")
        for i, feature in enumerate(feature_cols, 1):
            f.write(f"{i:3d}. {feature}\n")

    print(f"• Résumé détaillé: {summary_path}")
    print(f"✓ Tous les fichiers sauvegardés dans {datasets_dir}")


def main():
    """
    Pipeline principal de préparation des données avec architecture modulaire.
    """
    print("=== SCRIPT 1/3 : PRÉPARATION DONNÉES ARCHITECTURE MODULAIRE ===")
    print("Utilisation de l'architecture modulaire src.data.*")
    print("=" * 70)

    # Configuration
    set_seed()
    config = create_data_preparation_config()

    try:
        # 1. Chargement et validation des données
        print("\n=== ÉTAPE 1/6 : CHARGEMENT DES DONNÉES ===")
        data = load_all_data()
        print_dataset_info(data)
        validate_input_data(data)

        # 2. Préparation des données d'entraînement
        print("\n=== ÉTAPE 2/6 : PRÉPARATION ENTRAÎNEMENT ===")
        dataset_train, dataset_test, pipeline_metadata = prepare_training_data(
            data, config
        )

        # 3. Préparation des données de test
        print("\n=== ÉTAPE 3/6 : PRÉPARATION TEST ===")
        dataset_test_final = prepare_test_data(data, pipeline_metadata, config)

        # 4. Contrôle qualité
        print("\n=== ÉTAPE 4/6 : CONTRÔLE QUALITÉ ===")
        dataset_train_qc, dataset_test_qc, qc_metadata = apply_quality_control(
            dataset_train, dataset_test, config
        )

        # 5. Visualisations
        print("\n=== ÉTAPE 5/6 : VISUALISATIONS ===")
        create_data_visualizations(dataset_train_qc, config)

        # 6. Sauvegarde
        print("\n=== ÉTAPE 6/6 : SAUVEGARDE ===")
        save_datasets_and_metadata(
            dataset_train_qc,
            dataset_test_qc,
            dataset_test_final,
            pipeline_metadata,
            qc_metadata,
            config,
        )

        # Résumé final
        print("\n" + "=" * 70)
        print("✓ PIPELINE DE PRÉPARATION TERMINÉ AVEC SUCCÈS !")
        print("Architecture modulaire utilisée avec succès")
        print(f"Dataset final: {dataset_train_qc.shape}")
        print(
            f"Features: {len([c for c in dataset_train_qc.columns if c not in ['ID', 'OS_STATUS', 'OS_YEARS']])}"
        )
        print("Prochaine étape: python 2_train_models.py")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ ERREUR DANS LE PIPELINE: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
