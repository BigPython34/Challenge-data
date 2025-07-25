"""
Preprocessing médical intelligent pour la modélisation de survie en LMA

Ce module gère le preprocessing complet des données de leucémie myéloïde aiguë
avec une approche médicalement informée :

1. Nettoyage et validation des données
2. Imputation intelligente basée sur le contexte médical
3. Feature engineering cliniquement pertinent
4. Préparation pour les modèles de survie

Principes directeurs:
- Préserver l'information temporelle (survie)
- Utiliser des méthodes d'imputation adaptées au domaine médical
- Maintenir l'interprétabilité clinique
- Robustesse aux données manquantes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sksurv.util import Surv

from .. import config
from ..config import SEED
from . import features


def clean_and_validate_data(
    clinical_df: pd.DataFrame, molecular_df: pd.DataFrame, target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nettoyage et validation initiale des données

    Parameters:
    -----------
    clinical_df, molecular_df, target_df : DataFrames brutes

    Returns:
    --------
    Tuple : DataFrames nettoyées
    """
    print("=== DATA CLEANING AND VALIDATION ===")

    # ===== TARGET CLEANING (crucial for survival) =====

    # Remove patients without survival data
    target_clean = target_df.dropna(subset=["OS_YEARS", "OS_STATUS"]).copy()

    # Validate survival data
    target_clean["OS_YEARS"] = pd.to_numeric(target_clean["OS_YEARS"], errors="coerce")
    target_clean["OS_STATUS"] = target_clean["OS_STATUS"].astype(bool)

    # Remove negative or zero survival times
    invalid_survival = (target_clean["OS_YEARS"] <= 0) | target_clean["OS_YEARS"].isna()
    if invalid_survival.any():
        print(
            f"Suppression de {invalid_survival.sum()} patients avec temps de survie invalide"
        )
        target_clean = target_clean[~invalid_survival]

    print(f"Target nettoye: {len(target_clean)} patients")
    print(f"   - Death rate: {target_clean['OS_STATUS'].mean():.1%}")
    print(f"   - Median survival: {target_clean['OS_YEARS'].median():.2f} years")

    # ===== CLINICAL CLEANING =====

    # Keep only patients with survival data
    clinical_clean = clinical_df[clinical_df["ID"].isin(target_clean["ID"])].copy()

    # Validate clinical measurements
    numeric_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    for col in numeric_cols:
        if col in clinical_clean.columns:
            # Convertir en numérique
            clinical_clean[col] = pd.to_numeric(clinical_clean[col], errors="coerce")

            # Détecter les valeurs aberrantes biologiquement impossibles
            if col == "BM_BLAST":
                # Blastes ne peuvent pas dépasser 100%
                clinical_clean.loc[clinical_clean[col] > 100, col] = np.nan
            elif col in ["WBC", "ANC", "MONOCYTES", "PLT"]:
                # Valeurs négatives impossibles
                clinical_clean.loc[clinical_clean[col] < 0, col] = np.nan
            elif col == "HB":
                # Hémoglobine doit être dans une plage raisonnable
                clinical_clean.loc[
                    (clinical_clean[col] < 3) | (clinical_clean[col] > 20), col
                ] = np.nan

    print(f"Clinique nettoye: {len(clinical_clean)} patients")

    # ===== MOLECULAR CLEANING =====

    # Garder seulement les patients avec données de survie
    molecular_clean = molecular_df[molecular_df["ID"].isin(target_clean["ID"])].copy()

    # Valider les données de mutations
    molecular_clean["VAF"] = pd.to_numeric(molecular_clean["VAF"], errors="coerce")
    molecular_clean["DEPTH"] = pd.to_numeric(molecular_clean["DEPTH"], errors="coerce")

    # Filtrer les mutations de qualité douteuse
    # VAF doit être entre 0 et 1, depth > 10 pour être fiable
    valid_mutations = (
        (molecular_clean["VAF"] >= 0)
        & (molecular_clean["VAF"] <= 1)
        & (molecular_clean["DEPTH"] >= 10)
    )

    invalid_count = (~valid_mutations).sum()
    if invalid_count > 0:
        print(f"Suppression de {invalid_count} mutations de qualite douteuse")
        molecular_clean = molecular_clean[valid_mutations]

    print(
        f"Moleculaire nettoye: {len(molecular_clean)} mutations sur {molecular_clean['ID'].nunique()} patients"
    )

    return clinical_clean, molecular_clean, target_clean


def medical_imputation_strategy(
    df: pd.DataFrame, column: str, patient_context: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Imputation médicalement informée pour une colonne spécifique

    Utilise la connaissance du domaine médical pour choisir la meilleure
    stratégie d'imputation selon le type de mesure clinique.

    Parameters:
    -----------
    df : pd.DataFrame avec les données
    column : str, nom de la colonne à imputer
    patient_context : pd.DataFrame optionnel avec contexte additionnel

    Returns:
    --------
    pd.Series : Valeurs imputées
    """
    values = df[column].copy()
    missing_mask = values.isna()

    if not missing_mask.any():
        return values

    print(
        f"   🏥 Imputation médicale pour {column} ({missing_mask.sum()} valeurs manquantes)"
    )

    # ===== SPECIFIC STRATEGIES BY CLINICAL MEASURE =====

    if column == "BM_BLAST":
        # Blastes médullaires: utiliser la médiane, car distribution asymétrique
        # Si WBC très élevé, supposer blastose élevée (corrélation connue)
        if "WBC" in df.columns:
            high_wbc_mask = (df["WBC"] > 25) & missing_mask
            normal_wbc_mask = (df["WBC"] <= 25) & missing_mask

            median_val = values.median()
            high_wbc_median = (
                values[df["WBC"] > 25].median()
                if (df["WBC"] > 25).any()
                else median_val
            )

            values.loc[high_wbc_mask] = high_wbc_median
            values.loc[normal_wbc_mask] = median_val
        else:
            values.fillna(values.median(), inplace=True)

    elif column in ["WBC", "ANC", "MONOCYTES", "PLT"]:
        # Comptages cellulaires: distribution log-normale
        # Utiliser imputation par regression si d'autres comptages disponibles
        other_counts = ["WBC", "ANC", "MONOCYTES", "PLT"]
        other_counts = [c for c in other_counts if c in df.columns and c != column]

        if len(other_counts) >= 2:
            # Imputation par régression (relation entre les comptages)
            from sklearn.linear_model import LinearRegression

            # Préparer les données pour la régression
            X = df[other_counts].fillna(df[other_counts].median())
            y = values.dropna()
            X_train = X.loc[y.index]

            if len(y) > 10:  # Assez de données pour la régression
                reg = LinearRegression()
                reg.fit(X_train, y)

                # Prédire les valeurs manquantes
                X_missing = X.loc[missing_mask]
                predictions = reg.predict(X_missing)

                # Éviter les valeurs négatives
                predictions = np.maximum(predictions, 0.1)
                values.loc[missing_mask] = predictions
            else:
                # Fallback: médiane
                values.fillna(values.median(), inplace=True)
        else:
            # Fallback: médiane
            values.fillna(values.median(), inplace=True)

    elif column == "HB":
        # Hémoglobine: corrélée avec l'âge et le sexe
        # Utiliser des valeurs normales par tranche d'âge si possible
        median_val = values.median()

        # Distinguer anémie selon le contexte (si d'autres cytopénies présentes)
        if (
            patient_context is not None
            and "cytopenia_context" in patient_context.columns
        ):
            # Si autres cytopénies, HB probablement plus basse
            cytopenia_mask = patient_context["cytopenia_context"] & missing_mask
            normal_mask = ~patient_context["cytopenia_context"] & missing_mask

            anemic_median = values[patient_context["cytopenia_context"]].median()
            normal_median = values[~patient_context["cytopenia_context"]].median()

            if not np.isnan(anemic_median):
                values.loc[cytopenia_mask] = anemic_median
            if not np.isnan(normal_median):
                values.loc[normal_mask] = normal_median
        else:
            values.fillna(median_val, inplace=True)

    else:
        # Stratégie par défaut: médiane pour les variables continues
        values.fillna(values.median(), inplace=True)

    print(f"      {missing_mask.sum()} valeurs imputees")
    return values


def intelligent_clinical_imputation(
    clinical_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Imputation intelligente des données cliniques avec contexte médical

    Parameters:
    -----------
    clinical_df : pd.DataFrame avec données cliniques

    Returns:
    --------
    Tuple : (DataFrame imputé, métadonnées d'imputation)
    """
    print("=== IMPUTATION CLINIQUE INTELLIGENTE ===")

    df_imputed = clinical_df.copy()
    imputation_metadata = {"method": "medical_informed", "columns_imputed": []}

    # Créer un contexte de cytopénies pour informer l'imputation de l'HB
    cytopenia_context = pd.DataFrame(index=df_imputed.index)
    cytopenia_context["cytopenia_context"] = (
        (df_imputed["PLT"] < 100) | (df_imputed["ANC"] < 1.5)
    ).fillna(False)

    # Ordre d'imputation basé sur les dépendances médicales
    # 1. D'abord les comptages principaux (WBC)
    # 2. Puis les sous-populations (ANC, MONOCYTES)
    # 3. Puis les autres lignées (PLT, HB)
    # 4. Enfin les mesures dérivées (BM_BLAST)

    imputation_order = ["WBC", "ANC", "MONOCYTES", "PLT", "HB", "BM_BLAST"]

    for column in imputation_order:
        if column in df_imputed.columns and df_imputed[column].isna().any():
            df_imputed[column] = medical_imputation_strategy(
                df_imputed, column, cytopenia_context
            )
            imputation_metadata["columns_imputed"].append(column)

    # Gestion des données catégorielles
    if "CENTER" in df_imputed.columns:
        df_imputed["CENTER"] = df_imputed["CENTER"].fillna("Unknown")

    if "CYTOGENETICS" in df_imputed.columns:
        # Cytogénétique manquante = normale (hypothèse médicale standard)
        df_imputed["CYTOGENETICS"] = df_imputed["CYTOGENETICS"].fillna("46,XX")

    print("Imputation clinique terminee")
    print(f"   Colonnes imputées: {imputation_metadata['columns_imputed']}")
    print(f"   Valeurs manquantes restantes: {df_imputed.isna().sum().sum()}")

    return df_imputed, imputation_metadata


def prepare_survival_dataset(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    target_df: pd.DataFrame,
    test_size: float = 0.2,
    use_advanced_features: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Préparation complète du dataset pour la modélisation de survie

    Pipeline complet:
    1. Nettoyage et validation
    2. Feature engineering médical
    3. Imputation intelligente
    4. Split train/test
    5. Préparation des cibles de survie

    Parameters:
    -----------
    clinical_df, molecular_df, target_df : DataFrames bruts
    test_size : proportion pour le test set
    use_advanced_features : utiliser le feature engineering avancé

    Returns:
    --------
    Tuple : (train_df, test_df, metadata)
    """
    print("🏥 === PRÉPARATION DATASET DE SURVIE EN LMA ===")

    # ===== 1. NETTOYAGE ET VALIDATION =====
    clinical_clean, molecular_clean, target_clean = clean_and_validate_data(
        clinical_df, molecular_df, target_df
    )

    # ===== 2. FEATURE ENGINEERING MÉDICAL =====
    if use_advanced_features:
        print("\n=== FEATURE ENGINEERING MÉDICAL ===")

        # Features cliniques
        clinical_features = features.create_clinical_features(clinical_clean)
        print(f"Features cliniques: {len(clinical_features.columns)} variables")

        # Features cytogénétiques
        cyto_features = features.extract_cytogenetic_risk_features(clinical_features)
        print(f"Features cytogenetiques: {len(cyto_features.columns)} variables")

        # Features moléculaires
        molecular_features = features.extract_molecular_risk_features(
            clinical_features, molecular_clean
        )
        burden_features = features.create_molecular_burden_features(molecular_clean)
        print(
            f"Features moleculaires: {len(molecular_features.columns)} + {len(burden_features.columns)} variables"
        )

        # Combinaison de toutes les features
        enriched_df = features.combine_all_features(
            clinical_features, molecular_features, burden_features, cyto_features
        )
        print(f"Dataset enrichi: {enriched_df.shape}")

    else:
        # Version simple
        enriched_df = clinical_clean.copy()

    # ===== 3. IMPUTATION INTELLIGENTE =====
    enriched_df, imputation_metadata = intelligent_clinical_imputation(enriched_df)

    # ===== 4. MERGER AVEC LES TARGETS =====
    final_df = enriched_df.merge(target_clean, on="ID", how="inner")
    print(f"Dataset final: {final_df.shape}")

    # ===== 5. SPLIT TRAIN/TEST STRATIFIÉ =====
    # Stratification sur le status de survie pour équilibrer les événements
    train_df, test_df = train_test_split(
        final_df, test_size=test_size, random_state=SEED, stratify=final_df["OS_STATUS"]
    )

    print("Split termine:")
    print(
        f"   Train: {len(train_df)} patients ({train_df['OS_STATUS'].mean():.1%} événements)"
    )
    print(
        f"   Test:  {len(test_df)} patients ({test_df['OS_STATUS'].mean():.1%} événements)"
    )

    # ===== 6. PRÉPARER LES STRUCTURES DE SURVIE =====

    # Créer les arrays structurés pour scikit-survival
    train_y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", train_df)
    test_y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", test_df)

    # Ajouter aux DataFrames pour compatibilité
    train_df["y_survival"] = [train_y[i] for i in range(len(train_y))]
    test_df["y_survival"] = [test_y[i] for i in range(len(test_y))]

    # ===== 7. MÉTADONNÉES =====
    metadata = {
        "n_patients_initial": len(clinical_df),
        "n_patients_final": len(final_df),
        "n_mutations": len(molecular_clean),
        "n_features": final_df.shape[1]
        - 4,  # Moins ID, OS_YEARS, OS_STATUS, y_survival
        "train_size": len(train_df),
        "test_size": len(test_df),
        "event_rate_train": train_df["OS_STATUS"].mean(),
        "event_rate_test": test_df["OS_STATUS"].mean(),
        "median_followup": final_df["OS_YEARS"].median(),
        "imputation_metadata": imputation_metadata,
        "feature_engineering": use_advanced_features,
    }

    print("\nPREPARATION TERMINEE")
    print(
        f"   {metadata['n_patients_final']} patients, {metadata['n_features']} features"
    )
    print(f"   {metadata['n_mutations']} mutations analysees")
    print(f"   Suivi median: {metadata['median_followup']:.1f} ans")

    return train_df, test_df, metadata


def get_survival_feature_sets() -> Dict[str, List[str]]:
    """
    Définir des ensembles de features pour différents niveaux d'analyse

    Returns:
    --------
    Dict : Ensembles de features organisés par complexité/usage clinique
    """
    clean_lists = features.get_clean_feature_lists()

    return {
        # Ensemble minimal - pour usage clinique de routine
        "clinical_minimal": [
            "BM_BLAST",
            "WBC",
            "HB",
            "PLT",
            "anemia_severe",
            "thrombocytopenia_severe",
            "high_blast_count",
            "cytopenia_score",
        ],
        # Ensemble ELN 2017 - standard pour pronostic LMA
        "eln_standard": (
            clean_lists["clinical_base"]
            + clean_lists["cytogenetic"]
            + [
                "mut_NPM1",
                "mut_FLT3",
                "mut_TP53",
                "mut_CEBPA",
                "mut_ASXL1",
                "mut_RUNX1",
            ]
            + ["eln_integrated_risk"]
        ),
        # Ensemble complet - recherche/développement
        "research_complete": (
            clean_lists["clinical_base"]
            + clean_lists["clinical_ratios"]
            + clean_lists["clinical_binary"]
            + clean_lists["clinical_scores"]
            + clean_lists["cytogenetic"]
            + clean_lists["molecular_mutations"]
            + clean_lists["molecular_derived"]
            + clean_lists["mutation_burden"]
            + clean_lists["integrated_scores"]
        ),
        # Ensemble robuste - équilibre performance/interprétabilité
        "robust_balanced": (
            [
                "BM_BLAST",
                "WBC",
                "ANC",
                "HB",
                "PLT",
                "neutrophil_ratio",
                "monocyte_ratio",
            ]
            + [
                "anemia_severe",
                "thrombocytopenia_severe",
                "neutropenia_severe",
                "high_blast_count",
            ]
            + ["cytopenia_score", "proliferation_score"]
            + ["normal_karyotype", "complex_karyotype", "eln_cyto_risk"]
            + [
                "mut_NPM1",
                "mut_FLT3",
                "mut_TP53",
                "mut_ASXL1",
                "mut_DNMT3A",
                "mut_TET2",
            ]
            + ["eln_molecular_risk", "total_mutations", "vaf_mean"]
            + ["eln_integrated_risk", "clinical_risk_score"]
        ),
    }


# ===== FONCTIONS DE COMPATIBILITÉ =====


def prepare_enriched_dataset(
    clinical_df,
    molecular_df,
    target_df=None,
    imputer=None,
    advanced_imputation_method="medical",
    is_training=True,
    save_to_file=None,
):
    """
    Version de compatibilité avec l'ancien pipeline

    Utilise le nouveau système d'imputation médicale par défaut,
    mais peut revenir à l'ancien système si nécessaire.

    Parameters:
    -----------
    clinical_df : DataFrame des données cliniques
    molecular_df : DataFrame des données moléculaires
    target_df : DataFrame des targets (optionnel pour test)
    imputer : Imputer pré-entraîné (pour test)
    advanced_imputation_method : Méthode d'imputation
    is_training : Mode entraînement ou test
    save_to_file : Chemin pour sauvegarder le dataset préparé (optionnel)
    """
    if is_training:
        # Nouveau pipeline pour l'entraînement
        if target_df is not None:
            # Si on a des targets, utiliser le pipeline complet mais sans split
            clinical_clean, molecular_clean, target_clean = clean_and_validate_data(
                clinical_df, molecular_df, target_df
            )

            # Feature engineering
            clinical_features = features.create_clinical_features(clinical_clean)
            cyto_features = features.extract_cytogenetic_risk_features(
                clinical_features
            )
            molecular_features = features.extract_molecular_risk_features(
                clinical_features, molecular_clean
            )
            burden_features = features.create_molecular_burden_features(molecular_clean)

            enriched_df = features.combine_all_features(
                clinical_features, molecular_features, burden_features, cyto_features
            )

            # Imputation
            enriched_df, imputation_metadata = intelligent_clinical_imputation(
                enriched_df
            )

            # Imputation finale pour éliminer tous les NaN
            numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns
            enriched_df[numeric_cols] = enriched_df[numeric_cols].fillna(0)
            nan_count = enriched_df.isnull().sum().sum()
            if nan_count > 0:
                enriched_df = enriched_df.fillna(0)

            # Merger avec les targets
            final_df = enriched_df.merge(target_clean, on="ID", how="inner")

            # Sauvegarder si demandé
            if save_to_file:
                print(f"   Sauvegarde du dataset enrichi vers : {save_to_file}")
                final_df.to_csv(save_to_file, index=False)

            return final_df, imputation_metadata
        else:
            # Version sans target (pour compatibilité)
            clinical_clean, molecular_clean, _ = clean_and_validate_data(
                clinical_df,
                molecular_df,
                pd.DataFrame(
                    {"ID": clinical_df["ID"], "OS_YEARS": 1.0, "OS_STATUS": True}
                ),
            )

            clinical_features = features.create_clinical_features(clinical_clean)
            cyto_features = features.extract_cytogenetic_risk_features(
                clinical_features
            )
            molecular_features = features.extract_molecular_risk_features(
                clinical_features, molecular_clean
            )
            burden_features = features.create_molecular_burden_features(molecular_clean)

            enriched_df = features.combine_all_features(
                clinical_features, molecular_features, burden_features, cyto_features
            )

            enriched_df, imputation_metadata = intelligent_clinical_imputation(
                enriched_df
            )

            # Imputation finale pour éliminer tous les NaN
            numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns
            enriched_df[numeric_cols] = enriched_df[numeric_cols].fillna(0)
            nan_count = enriched_df.isnull().sum().sum()
            if nan_count > 0:
                enriched_df = enriched_df.fillna(0)

            # Sauvegarder si demandé
            if save_to_file:
                print(f"   Sauvegarde du dataset enrichi vers : {save_to_file}")
                enriched_df.to_csv(save_to_file, index=False)

            return enriched_df, imputation_metadata
    else:
        # Mode test - utiliser l'imputer fourni (compatibilité)
        if imputer is None:
            raise ValueError("Imputer requis pour les données de test")

        # Pipeline simplifié pour le test
        enriched_df = clinical_df.copy()

        # Appliquer les mêmes transformations qu'en training
        clinical_features = features.create_clinical_features(enriched_df)
        cyto_features = features.extract_cytogenetic_risk_features(clinical_features)
        molecular_features = features.extract_molecular_risk_features(
            clinical_features, molecular_df
        )
        burden_features = features.create_molecular_burden_features(molecular_df)

        enriched_df = features.combine_all_features(
            clinical_features, molecular_features, burden_features, cyto_features
        )

        # Imputation finale pour éliminer tous les NaN
        numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns
        enriched_df[numeric_cols] = enriched_df[numeric_cols].fillna(0)
        nan_count = enriched_df.isnull().sum().sum()
        if nan_count > 0:
            enriched_df = enriched_df.fillna(0)

        # Imputation simple avec les métadonnées sauvegardées
        for col in imputer.get("columns_imputed", []):
            if col in enriched_df.columns and enriched_df[col].isna().any():
                enriched_df[col] = enriched_df[col].fillna(enriched_df[col].median())

        # Sauvegarder si demandé
        if save_to_file:
            print(f"   Sauvegarde du dataset de test enrichi vers : {save_to_file}")
            enriched_df.to_csv(save_to_file, index=False)

        return enriched_df


def clean_target_data(target_df):
    """Version de compatibilité"""
    target_clean = target_df.copy()
    target_clean.dropna(subset=["OS_YEARS", "OS_STATUS"], inplace=True)
    target_clean["OS_YEARS"] = pd.to_numeric(target_clean["OS_YEARS"], errors="coerce")
    target_clean["OS_STATUS"] = target_clean["OS_STATUS"].astype(bool)
    return target_clean


def prepare_features_and_target(df_enriched, target_df, test_size=0.2):
    """Version de compatibilité avec l'ancien format"""
    # Merger avec les targets
    final_df = df_enriched.merge(target_df, on="ID", how="inner")

    # Nettoyer les colonnes dupliquées
    if "OS_YEARS_y" in final_df.columns:
        final_df = final_df.drop(columns=["OS_YEARS_x", "OS_STATUS_x"])
        final_df = final_df.rename(
            columns={"OS_YEARS_y": "OS_YEARS", "OS_STATUS_y": "OS_STATUS"}
        )

    # Split - utiliser la colonne de statut appropriée
    status_col = None
    for col in ["STATUS", "OS_STATUS", "Event", "event"]:
        if col in final_df.columns:
            status_col = col
            break

    if status_col is None:
        raise ValueError(
            f"Aucune colonne de statut trouvée. Colonnes disponibles: {list(final_df.columns)}"
        )

    train_df, test_df = train_test_split(
        final_df, test_size=test_size, random_state=SEED, stratify=final_df[status_col]
    )

    # Préparer les features (exclure les colonnes de métadonnées)
    exclude_cols = ["ID", "OS_YEARS", "OS_STATUS", "CENTER", "CYTOGENETICS"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Préparer les targets de survie
    y_train = Surv.from_dataframe("OS_STATUS", "OS_YEARS", train_df)
    y_test = Surv.from_dataframe("OS_STATUS", "OS_YEARS", test_df)

    return X_train, X_test, y_train, y_test, feature_cols
