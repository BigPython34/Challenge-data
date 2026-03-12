"""
Imputation Module - Centralized imputation logic for AML pipeline.

Ce module centralise toute la logique d'imputation:
- Early imputation (AVANT feature engineering) avec colonnes auxiliaires optionnelles
- Pipeline imputation (APRÈS feature engineering) différenciée par type de feature
- Monocyte imputation supervisée (optionnelle)

Flow recommandé:
1. Charger données brutes
2. Créer colonnes auxiliaires basiques (via auxiliary_features.py)
3. apply_early_imputation() - impute colonnes cliniques avec aide des auxiliaires
4. Feature Engineering complet
5. Pipeline finale (scaling, encoding) - impute seulement si single_imputation_mode=False
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import joblib

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from .data_cleaning.imputer import AdvancedImputer, supervised_monocyte_imputation
from ..config import (
    PREPROCESSING,
    TARGET_COLUMNS,
    ID_COLUMNS,
    CLINICAL_RANGES,
    MODEL_DIR,
)


def apply_early_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    extra_fit_df: Optional[pd.DataFrame] = None,
    auxiliary_columns: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputation précoce des colonnes cliniques AVANT feature engineering.
    
    Cette fonction impute les colonnes continues (WBC, HB, PLT, etc.) pour permettre
    le calcul des ratios et features dérivées dans le Feature Engineering.
    
    Si use_auxiliary_columns=True et que des colonnes auxiliaires sont présentes,
    elles sont utilisées pour enrichir le fit de l'imputer sans être modifiées.
    
    Args:
        train_df: DataFrame d'entraînement
        test_df: DataFrame de test
        extra_fit_df: DataFrame optionnel pour enrichir le fit (ex: Beat AML)
        auxiliary_columns: Liste des colonnes auxiliaires à utiliser pour le fit
    
    Returns:
        Tuple (train_df imputed, test_df imputed)
    """
    early_cfg = PREPROCESSING.get("early_imputation", {})
    if not early_cfg.get("enabled", False):
        print("[IMPUTE] Early imputation disabled.")
        return train_df, test_df
    
    # Colonnes cibles à imputer
    candidate_columns = early_cfg.get("columns") or PREPROCESSING.get("continuous_features", [])
    protected = {TARGET_COLUMNS["status"], TARGET_COLUMNS["time"]}
    target_columns = [
        col for col in candidate_columns
        if col in train_df.columns and col in test_df.columns and col not in protected
    ]
    
    if not target_columns:
        print("[IMPUTE] No continuous columns available for early imputation.")
        return train_df, test_df
    
    # Configuration
    strategy = early_cfg.get("strategy", PREPROCESSING.get("imputer", "iterative"))
    n_neighbors = early_cfg.get("n_neighbors")
    use_aux = early_cfg.get("use_auxiliary_columns", False)
    
    print(f"[IMPUTE] Early imputation on {len(target_columns)} features using '{strategy}'.")
    before_train = train_df[target_columns].isna().mean().mean()
    before_test = test_df[target_columns].isna().mean().mean()
    
    # --- Déterminer les colonnes auxiliaires disponibles ---
    aux_cols = []
    if use_aux and auxiliary_columns:
        aux_cols = [
            col for col in auxiliary_columns
            if col in train_df.columns and col in test_df.columns and col not in target_columns
        ]
        if aux_cols:
            print(f"[IMPUTE] Using {len(aux_cols)} auxiliary columns: {aux_cols}")
    
    # --- Construire la matrice de fit ---
    fit_columns = target_columns + aux_cols
    fit_parts = [train_df[fit_columns].copy()]
    
    include_test = PREPROCESSING.get("imputer_fit_scope", {}).get("include_test_rows", False)
    if include_test:
        fit_parts.append(test_df[fit_columns].copy())
        print("[IMPUTE] Including test rows in imputer fit.")
    
    if extra_fit_df is not None:
        extra_subset = extra_fit_df.reindex(columns=fit_columns)
        if len(extra_subset) > 0:
            fit_parts.append(extra_subset)
            print(f"[IMPUTE] Including {len(extra_subset)} external rows in fit.")
    
    fit_matrix = pd.concat(fit_parts, ignore_index=True)
    
    # --- Fit de l'imputer ---
    imputer = AdvancedImputer(strategy=strategy, n_neighbors=n_neighbors)
    imputer.fit(fit_matrix)
    
    # --- Transform ---
    if aux_cols:
        # Transformer avec les colonnes auxiliaires, puis extraire seulement les cibles
        train_full = imputer.transform(train_df[fit_columns])
        test_full = imputer.transform(test_df[fit_columns])
        
        # Extraire seulement les colonnes cibles (les premières)
        train_imputed = train_full.iloc[:, :len(target_columns)]
        test_imputed = test_full.iloc[:, :len(target_columns)]
    else:
        train_imputed = imputer.transform(train_df[target_columns])
        test_imputed = imputer.transform(test_df[target_columns])
    
    # --- Appliquer les valeurs imputées ---
    for i, col in enumerate(target_columns):
        train_df.loc[:, col] = train_imputed.iloc[:, i].astype("float32").values
        test_df.loc[:, col] = test_imputed.iloc[:, i].astype("float32").values
    
    # --- Respecter les bornes cliniques ---
    if early_cfg.get("respect_ranges", False):
        range_map = early_cfg.get("range_map") or CLINICAL_RANGES
        for col in target_columns:
            bounds = range_map.get(col)
            if bounds:
                train_df[col] = train_df[col].clip(*bounds)
                test_df[col] = test_df[col].clip(*bounds)
    
    # --- Sauvegarder l'imputer ---
    artifact_path = early_cfg.get("artifact_path")
    if artifact_path:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(imputer, artifact_path)
        print(f"[IMPUTE] Saved early imputer to {artifact_path}")
    
    # --- Stats ---
    after_train = train_df[target_columns].isna().mean().mean()
    after_test = test_df[target_columns].isna().mean().mean()
    print(
        f"[IMPUTE] Train: {before_train:.2%} -> {after_train:.2%} | "
        f"Test: {before_test:.2%} -> {after_test:.2%}"
    )
    
    return train_df, test_df


def apply_monocyte_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    extra_fit_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputation supervisée dédiée pour MONOCYTES.
    
    Utilise un modèle HistGradientBoosting entraîné sur les autres colonnes cliniques
    pour prédire les valeurs manquantes de MONOCYTES.
    
    Args:
        train_df: DataFrame d'entraînement
        test_df: DataFrame de test
        extra_fit_df: DataFrame optionnel pour enrichir le fit
    
    Returns:
        Tuple (train_df, test_df) avec MONOCYTES imputé
    """
    mono_mode = PREPROCESSING.get("monocyte_mode", "joint")
    
    if mono_mode != "separate":
        print(f"[MONOCYTE] Mode '{mono_mode}': MONOCYTES will be imputed via global pipeline.")
        return train_df, test_df
    
    mono_cfg = PREPROCESSING.get("monocyte_imputer", {})
    
    print("[MONOCYTE] Supervised imputation for MONOCYTES...")
    
    # Combiner les données pour le fit si demandé
    combined = train_df.copy()
    if extra_fit_df is not None and not extra_fit_df.empty:
        combined = pd.concat([combined, extra_fit_df], ignore_index=True)
    
    train_df, test_df, meta = supervised_monocyte_imputation(
        train_df, test_df, combined_for_fit=combined
    )
    
    # Sauvegarder le modèle
    model_path = mono_cfg.get("model_path")
    if model_path and meta.get("model"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(meta["model"], model_path)
        
        meta_path = model_path.replace(".joblib", "_meta.json")
        import json
        with open(meta_path, "w") as f:
            safe_meta = {k: v for k, v in meta.items() if k != "model"}
            json.dump(safe_meta, f, indent=2)
        print(f"[MONOCYTE] Model saved to {model_path}")
    
    return train_df, test_df


def create_missingness_indicators(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Crée des indicateurs binaires de données manquantes.
    
    Args:
        df: DataFrame source
        columns: Colonnes pour lesquelles créer des indicateurs (défaut: config)
    
    Returns:
        DataFrame avec colonnes <col>_missing ajoutées
    """
    from ..config import MISSINGNESS_POLICY
    
    if not MISSINGNESS_POLICY.get("create_indicators", False):
        return df
    
    if columns is None:
        columns = MISSINGNESS_POLICY.get("keep_columns", [])
    
    for col in columns:
        if col in df.columns:
            indicator_col = f"{col}_missing"
            df[indicator_col] = df[col].isna().astype(int)
    
    return df


def get_imputation_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Génère un résumé des valeurs manquantes avant/après imputation.
    
    Returns:
        Dict avec statistiques de missingness par colonne
    """
    summary = {
        "train": {
            "total_rows": len(train_df),
            "total_missing": train_df.isna().sum().sum(),
            "missing_by_col": train_df.isna().sum().to_dict(),
        },
        "test": {
            "total_rows": len(test_df),
            "total_missing": test_df.isna().sum().sum(),
            "missing_by_col": test_df.isna().sum().to_dict(),
        },
    }
    return summary
