# Nettoyage, split, imputation, etc.
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

from .. import config
from ..config import SEED
from . import features


def clean_target_data(target_df):
    """Nettoie les données target"""
    # Drop rows where 'OS_YEARS' is NaN
    target_df.dropna(subset=["OS_YEARS", "OS_STATUS"], inplace=True)

    # Convert 'OS_YEARS' to numeric
    target_df["OS_YEARS"] = pd.to_numeric(target_df["OS_YEARS"], errors="coerce")

    # Ensure 'OS_STATUS' is boolean
    target_df["OS_STATUS"] = target_df["OS_STATUS"].astype(bool)

    return target_df


def prepare_enriched_dataset(
    clinical_df, molecular_df, target_df=None, imputer=None, is_training=True
):
    """Prépare un dataset enrichi avec toutes les features"""
    # 1. Création de features basées sur les gènes
    gene_features = pd.DataFrame(index=clinical_df["ID"].unique())

    for gene in config.IMPORTANT_GENES:
        mutated_patients = molecular_df[molecular_df["GENE"] == gene]["ID"].unique()
        gene_features[f"has_{gene}_mutation"] = gene_features.index.isin(
            mutated_patients
        ).astype(int)

    # 2. Statistiques moléculaires
    mutation_counts, vaf_stats, effect_counts = (
        features.create_molecular_stats_features(molecular_df)
    )

    # 3. Fusion des features moléculaires
    mol_features = gene_features.reset_index().rename(columns={"index": "ID"})
    mol_features = pd.merge(mol_features, mutation_counts, on="ID", how="outer")
    mol_features = pd.merge(mol_features, vaf_stats, on="ID", how="outer")
    mol_features = pd.merge(mol_features, effect_counts, on="ID", how="outer")

    # 4. Fusion avec les données cliniques
    df_enriched = clinical_df.merge(mol_features, on="ID", how="left")

    # 5. Features cytogénétiques
    cyto_features = features.extract_advanced_cytogenetic_features(clinical_df)
    df_enriched = pd.concat([df_enriched, cyto_features], axis=1)

    # 6. Features cliniques avancées
    df_enriched = features.create_advanced_clinical_features(df_enriched)

    # 7. Encoder le centre médical
    center_dummies = pd.get_dummies(df_enriched["CENTER"], prefix="center")
    df_enriched = pd.concat([df_enriched, center_dummies], axis=1)

    # 8. Gestion des valeurs manquantes
    for col in df_enriched.columns:
        if col not in [
            "ID",
            "CENTER",
            "BM_BLAST",
            "WBC",
            "ANC",
            "MONOCYTES",
            "HB",
            "PLT",
            "CYTOGENETICS",
        ]:
            df_enriched[col] = df_enriched[col].fillna(0)

    # 9. Imputation des données cliniques
    clinical_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

    if is_training:
        imputer_clinical = SimpleImputer(strategy="median")
        df_enriched[clinical_cols] = imputer_clinical.fit_transform(
            df_enriched[clinical_cols]
        )
        return df_enriched, imputer_clinical
    else:
        if imputer is None:
            raise ValueError("Imputer must be provided for test data")
        df_enriched[clinical_cols] = imputer.transform(df_enriched[clinical_cols])
        return df_enriched


def prepare_features_and_target(df_enriched, target_df, test_size=0.2):
    """Prépare les features et le target pour l'entraînement"""
    # Définir les features
    feature_lists = features.get_feature_lists()

    # Obtenir les colonnes center
    center_features = [col for col in df_enriched.columns if col.startswith("center_")]
    effect_features = [
        col
        for col in df_enriched.columns
        if col
        in [
            "frameshift_variant",
            "missense_variant",
            "nonsense",
            "splice_acceptor_variant",
            "splice_donor_variant",
        ]
    ]

    # Construire la liste finale des features (éviter le conflit de nom avec le module features)
    final_features = (
        feature_lists["clinical"]
        + feature_lists["gene_mutations"]
        + feature_lists["statistics"]
        + effect_features
        + feature_lists["cytogenetic"]
        + feature_lists["ratios"]
        + feature_lists["clinical_scores"]
        + center_features
    )

    # Créer X et y
    X = df_enriched.loc[df_enriched["ID"].isin(target_df["ID"]), final_features]
    y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", target_df)

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    # Gérer les valeurs manquantes
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    return X_train, X_test, y_train, y_test, final_features


def prepare_test_dataset(df_test_enriched, features, center_columns_train):
    """Prépare le dataset de test pour les prédictions"""
    # Assurer la cohérence des colonnes center
    for col in center_columns_train:
        if col not in df_test_enriched.columns:
            df_test_enriched[col] = 0

    # Vérifier que toutes les features sont présentes
    for feature in features:
        if feature not in df_test_enriched.columns:
            df_test_enriched[feature] = 0

    # S'assurer que les features sont dans le même ordre
    X_test = df_test_enriched.loc[:, features]
    X_test.fillna(0, inplace=True)

    return X_test
