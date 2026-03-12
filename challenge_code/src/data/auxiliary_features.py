"""
Auxiliary Features for Imputation Guidance.

Ce module crée les colonnes auxiliaires MINIMALES nécessaires pour guider l'imputation
AVANT le Feature Engineering complet.

L'idée est d'extraire rapidement des indicateurs moléculaires/cytogénétiques simples
(mut_TP53, mut_FLT3, eln_cyto_*) pour que l'IterativeImputer puisse apprendre des
corrélations comme "mut_FLT3 → WBC élevé" lors de l'imputation des colonnes cliniques.

Ces features sont :
- Rapides à calculer (pas de parsing complexe)
- Binaires ou catégorielles simples
- Corrélées aux valeurs cliniques manquantes

Utilisé par: apply_early_continuous_imputation() dans 1_prepare_data.py
"""

import pandas as pd
import numpy as np
from typing import Optional

from ..config import (
    PREPROCESSING,
    ADVERSE_GENES,
    FAVORABLE_GENES,
    INTERMEDIATE_GENES,
    GENE_PATHWAYS,
    ID_COLUMNS,
)


def create_basic_mutation_flags(
    molecular_df: pd.DataFrame,
    patient_ids: pd.Series,
    genes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Crée des flags binaires de présence de mutations pour une liste de gènes.
    
    Args:
        molecular_df: DataFrame avec colonnes GENE, ID (format mutation-level)
        patient_ids: Series des IDs patients pour lesquels créer les flags
        genes: Liste de gènes à flaguer (défaut: gènes auxiliaires de la config)
    
    Returns:
        DataFrame avec colonnes mut_<GENE> (1 si mutation présente, 0 sinon)
    """
    if genes is None:
        aux_cfg = PREPROCESSING.get("imputer_auxiliary_features", {})
        aux_cols = aux_cfg.get("columns", [])
        # Extraire les gènes des colonnes mut_*
        genes = [col.replace("mut_", "") for col in aux_cols if col.startswith("mut_")]
    
    if not genes or molecular_df.empty:
        return pd.DataFrame(index=patient_ids)
    
    # Créer un DataFrame patient-level
    result = pd.DataFrame(index=patient_ids)
    result.index.name = ID_COLUMNS["patient"]
    
    # Identifier la colonne du gène
    gene_col = "GENE" if "GENE" in molecular_df.columns else "Hugo_Symbol"
    id_col = ID_COLUMNS["patient"]
    
    if gene_col not in molecular_df.columns or id_col not in molecular_df.columns:
        return result
    
    # Pour chaque gène, créer le flag
    for gene in genes:
        col_name = f"mut_{gene}"
        patients_with_mutation = set(
            molecular_df.loc[molecular_df[gene_col] == gene, id_col].astype(str)
        )
        result[col_name] = result.index.astype(str).isin(patients_with_mutation).astype(int)
    
    return result.reset_index()


def create_basic_cyto_risk(
    clinical_df: pd.DataFrame,
    cyto_column: str = "CYTOGENETICS",
) -> pd.DataFrame:
    """
    Crée des flags ELN cytogénétiques simplifiés.
    
    Version rapide sans parsing complexe - utilise des patterns regex basiques.
    
    Args:
        clinical_df: DataFrame clinique avec colonne CYTOGENETICS
        cyto_column: Nom de la colonne cytogénétique
    
    Returns:
        DataFrame avec colonnes eln_cyto_favorable, eln_cyto_intermediate, eln_cyto_adverse
    """
    result = pd.DataFrame(index=clinical_df.index)
    
    if cyto_column not in clinical_df.columns:
        result["eln_cyto_favorable"] = 0
        result["eln_cyto_intermediate"] = 0
        result["eln_cyto_adverse"] = 0
        return result
    
    cyto = clinical_df[cyto_column].fillna("").astype(str).str.lower()
    
    # Patterns simplifiés pour classification rapide
    # Favorable: t(8;21), inv(16), t(15;17), normal karyotype
    favorable_patterns = [
        r"t\s*\(\s*8\s*;\s*21\s*\)",
        r"inv\s*\(\s*16\s*\)",
        r"t\s*\(\s*15\s*;\s*17\s*\)",
        r"46\s*,\s*x[xy]\s*\[",  # Normal karyotype
        r"\bnormal\b",
    ]
    
    # Adverse: complex (≥3), -5, -7, del(17p), etc.
    adverse_patterns = [
        r"complex",
        r"-\s*5\b",
        r"-\s*7\b",
        r"del\s*\(\s*5\s*\)",
        r"del\s*\(\s*7\s*\)",
        r"del\s*\(\s*17\s*\)\s*\(p",
        r"inv\s*\(\s*3\s*\)",
        r"t\s*\(\s*3\s*;\s*3\s*\)",
        r"t\s*\(\s*6\s*;\s*9\s*\)",
        r"t\s*\(\s*9\s*;\s*22\s*\)",
    ]
    
    # Calculer les flags
    is_favorable = cyto.str.contains("|".join(favorable_patterns), regex=True, na=False)
    is_adverse = cyto.str.contains("|".join(adverse_patterns), regex=True, na=False)
    
    # Adverse prime sur favorable si conflit
    is_favorable = is_favorable & ~is_adverse
    is_intermediate = ~is_favorable & ~is_adverse & (cyto != "") & (cyto != "nan")
    
    result["eln_cyto_favorable"] = is_favorable.astype(int)
    result["eln_cyto_intermediate"] = is_intermediate.astype(int)
    result["eln_cyto_adverse"] = is_adverse.astype(int)
    
    return result


def create_pathway_flags(
    molecular_df: pd.DataFrame,
    patient_ids: pd.Series,
    pathways: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Crée des flags binaires de présence de mutations dans des pathways.
    
    Args:
        molecular_df: DataFrame avec colonnes GENE, ID
        patient_ids: Series des IDs patients
        pathways: Dict pathway_name -> list[genes] (défaut: GENE_PATHWAYS)
    
    Returns:
        DataFrame avec colonnes <pathway>_altered (1 si au moins une mutation)
    """
    if pathways is None:
        pathways = GENE_PATHWAYS
    
    result = pd.DataFrame(index=patient_ids)
    result.index.name = ID_COLUMNS["patient"]
    
    if molecular_df.empty:
        for pathway in pathways:
            result[f"{pathway}_altered"] = 0
        return result.reset_index()
    
    gene_col = "GENE" if "GENE" in molecular_df.columns else "Hugo_Symbol"
    id_col = ID_COLUMNS["patient"]
    
    if gene_col not in molecular_df.columns:
        for pathway in pathways:
            result[f"{pathway}_altered"] = 0
        return result.reset_index()
    
    for pathway_name, genes in pathways.items():
        patients_with_pathway_mut = set(
            molecular_df.loc[molecular_df[gene_col].isin(genes), id_col].astype(str)
        )
        result[f"{pathway_name}_altered"] = (
            result.index.astype(str).isin(patients_with_pathway_mut).astype(int)
        )
    
    return result.reset_index()


def create_auxiliary_features_for_imputation(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    aux_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Point d'entrée principal: crée toutes les features auxiliaires pour l'imputation.
    
    Cette fonction est appelée AVANT le Feature Engineering complet pour fournir
    des colonnes auxiliaires à l'IterativeImputer.
    
    Args:
        clinical_df: DataFrame clinique (avec ID, CYTOGENETICS, etc.)
        molecular_df: DataFrame moléculaire (format mutation-level)
        aux_columns: Liste explicite des colonnes à créer (défaut: config)
    
    Returns:
        DataFrame avec colonnes auxiliaires, indexé par patient ID
    """
    if aux_columns is None:
        aux_cfg = PREPROCESSING.get("imputer_auxiliary_features", {})
        aux_columns = aux_cfg.get("columns", [])
    
    if not aux_columns:
        return pd.DataFrame()
    
    patient_ids = clinical_df[ID_COLUMNS["patient"]].astype(str)
    result = pd.DataFrame({ID_COLUMNS["patient"]: patient_ids})
    
    # Identifier les types de colonnes demandées
    mut_genes = [col.replace("mut_", "") for col in aux_columns if col.startswith("mut_")]
    pathway_cols = [col for col in aux_columns if col.endswith("_altered")]
    cyto_cols = [col for col in aux_columns if col.startswith("eln_cyto_")]
    
    # Créer les flags de mutations
    if mut_genes:
        mut_flags = create_basic_mutation_flags(molecular_df, patient_ids, mut_genes)
        if not mut_flags.empty and ID_COLUMNS["patient"] in mut_flags.columns:
            result = result.merge(mut_flags, on=ID_COLUMNS["patient"], how="left")
    
    # Créer les flags cytogénétiques
    if cyto_cols:
        cyto_flags = create_basic_cyto_risk(clinical_df)
        for col in cyto_cols:
            if col in cyto_flags.columns:
                result[col] = cyto_flags[col].values
    
    # Créer les flags de pathways
    if pathway_cols:
        pathway_names = [col.replace("_altered", "") for col in pathway_cols]
        pathways_to_create = {name: GENE_PATHWAYS.get(name, []) for name in pathway_names if name in GENE_PATHWAYS}
        if pathways_to_create:
            pathway_flags = create_pathway_flags(molecular_df, patient_ids, pathways_to_create)
            if not pathway_flags.empty and ID_COLUMNS["patient"] in pathway_flags.columns:
                result = result.merge(pathway_flags, on=ID_COLUMNS["patient"], how="left")
    
    # Remplir les NaN par 0 pour les colonnes binaires
    for col in result.columns:
        if col != ID_COLUMNS["patient"]:
            result[col] = result[col].fillna(0).astype(int)
    
    created_cols = [col for col in result.columns if col != ID_COLUMNS["patient"]]
    if created_cols:
        print(f"[AUX] {len(created_cols)} auxiliary features created for imputation: {created_cols}")
    
    return result


def merge_auxiliary_features(
    base_df: pd.DataFrame,
    aux_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusionne les features auxiliaires dans le DataFrame de base.
    
    Args:
        base_df: DataFrame principal (clinical + molecular merged)
        aux_df: DataFrame des features auxiliaires
    
    Returns:
        DataFrame avec les colonnes auxiliaires ajoutées
    """
    if aux_df.empty:
        return base_df
    
    id_col = ID_COLUMNS["patient"]
    if id_col not in aux_df.columns:
        return base_df
    
    # S'assurer que les types d'ID correspondent
    base_df[id_col] = base_df[id_col].astype(str)
    aux_df[id_col] = aux_df[id_col].astype(str)
    
    # Ne pas écraser les colonnes existantes
    existing_cols = set(base_df.columns)
    new_cols = [col for col in aux_df.columns if col not in existing_cols or col == id_col]
    
    if len(new_cols) <= 1:  # Seulement l'ID
        return base_df
    
    return base_df.merge(aux_df[new_cols], on=id_col, how="left")


def inject_auxiliary_features_for_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Injecte les features auxiliaires pour guider l'imputation de la pipeline finale.
    
    Cette fonction est appelée APRÈS le Feature Engineering, quand early_imputation est
    désactivé mais que imputer_auxiliary_features.enabled=True.
    
    Elle copie les colonnes auxiliaires existantes (déjà créées par le FE) avec le préfixe
    __aux_impute__ pour que la pipeline les utilise dans l'imputation.
    
    Args:
        X_train: DataFrame d'entraînement après FE
        X_test: DataFrame de test après FE
    
    Returns:
        Tuple (X_train_with_aux, X_test_with_aux, list_of_injected_columns)
    """
    aux_cfg = PREPROCESSING.get("imputer_auxiliary_features", {})
    if not aux_cfg.get("enabled", False):
        return X_train, X_test, []
    
    prefix = aux_cfg.get("prefix", "__aux_impute__")
    source_columns = aux_cfg.get("columns", [])
    
    if not source_columns:
        return X_train, X_test, []
    
    injected_cols = []
    
    for col in source_columns:
        # Check if the source column exists in the data (created by FE)
        if col in X_train.columns and col in X_test.columns:
            new_col_name = f"{prefix}{col}"
            X_train[new_col_name] = X_train[col].fillna(0).astype("float32")
            X_test[new_col_name] = X_test[col].fillna(0).astype("float32")
            injected_cols.append(new_col_name)
    
    return X_train, X_test, injected_cols
