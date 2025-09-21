"""
Data cleaning module for AML survival analysis.

This module handles the initial cleaning and validation of clinical,
molecular, and survival data for AML patients.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from ...config import CLINICAL_RANGES, PREPROCESSING
from sklearn.base import BaseEstimator, TransformerMixin


class ClipQuantiles(BaseEstimator, TransformerMixin):
    """
    Clippe les colonnes et est maintenant 100% compatible avec l'API set_output.
    """

    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        # S'assurer que X est un DataFrame pour utiliser .quantile
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        clip_quantiles_config = PREPROCESSING.get("clip_quantiles", {})
        lower = self.lower if self.lower is not None else clip_quantiles_config.get("lower", 0.01)
        upper = self.upper if self.upper is not None else clip_quantiles_config.get("upper", 0.99)

        self.lower_bounds_ = X.quantile(lower)
        self.upper_bounds_ = X.quantile(upper)
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X) -> pd.DataFrame:
        """Applique le clipping tout en garantissant la préservation de l'index."""
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

    # --- AJOUT CRUCIAL POUR LA COMPATIBILITÉ ---
    def set_output(self, transform=None):
        """
        Méthode requise par scikit-learn pour gérer la configuration de la sortie.
        """
        return self

    def get_feature_names_out(self, input_features=None):
        """
        Méthode requise pour propager les noms de colonnes dans la pipeline.
        """
        return self.feature_names_in_


def clean_and_validate_data(
    clinical_df: pd.DataFrame, molecular_df: pd.DataFrame, target_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean and validate raw data from all sources.

    This function performs comprehensive data cleaning including:
    - Survival data validation (removing invalid times/events)
    - Clinical measurements validation (biologically plausible ranges)
    - Molecular data quality filtering
    - Cross-dataset consistency checks

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Raw clinical data with patient measurements
    molecular_df : pd.DataFrame
        Raw molecular data with mutation information
    target_df : pd.DataFrame
        Survival outcome data (OS_YEARS, OS_STATUS)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Cleaned clinical, molecular, and target dataframes
    """
    print("=== DATA CLEANING AND VALIDATION ===")

    # ===== TARGET CLEANING (crucial for survival analysis) =====
    target_clean = _clean_survival_data(target_df)

    # ===== CLINICAL CLEANING =====
    clinical_clean = _clean_clinical_data(clinical_df, target_clean)

    # ===== MOLECULAR CLEANING =====
    molecular_clean = _clean_molecular_data(molecular_df, target_clean)

    return clinical_clean, molecular_clean, target_clean


def _clean_survival_data(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate survival outcome data.

    Parameters
    ----------
    target_df : pd.DataFrame
        Raw survival data

    Returns
    -------
    pd.DataFrame
        Cleaned survival data
    """
    print("Cleaning survival data...")

    # Remove patients without survival data
    target_clean = target_df.dropna(subset=["OS_YEARS", "OS_STATUS"]).copy()

    # Validate survival data types
    target_clean["OS_YEARS"] = pd.to_numeric(target_clean["OS_YEARS"], errors="coerce")
    target_clean["OS_STATUS"] = pd.to_numeric(
        target_clean["OS_STATUS"], errors="coerce"
    ).astype(int)

    # Remove negative or zero survival times
    invalid_survival = (target_clean["OS_YEARS"] <= 0) | target_clean["OS_YEARS"].isna()
    if invalid_survival.any():
        print(
            f"   Removing {invalid_survival.sum()} patients with invalid survival times"
        )
        target_clean = target_clean[~invalid_survival]

    print(f"   Clean survival data: {len(target_clean)} patients")
    print(f"   Event rate: {target_clean['OS_STATUS'].mean():.1%}")
    print(f"   Median survival: {target_clean['OS_YEARS'].median():.2f} years")

    return target_clean


def _clean_clinical_data(
    clinical_df: pd.DataFrame, target_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean and validate clinical measurements.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Raw clinical data
    target_clean : pd.DataFrame
        Cleaned survival data for patient filtering

    Returns
    -------
    pd.DataFrame
        Cleaned clinical data
    """
    print("Cleaning clinical data...")

    # Keep only patients with survival data
    clinical_clean = clinical_df[clinical_df["ID"].isin(target_clean["ID"])].copy()

    # Validate clinical measurements
    numeric_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

    for col in numeric_cols:
        if col in clinical_clean.columns:
            # Convert to numeric and coerce errors to NaN
            clinical_clean[col] = pd.to_numeric(clinical_clean[col], errors="coerce")
            # Apply biologically plausible ranges
            clinical_clean = _apply_clinical_ranges(clinical_clean, col)

    print(f"   Clean clinical data: {len(clinical_clean)} patients")
    return clinical_clean


def _clean_molecular_data(
    molecular_df: pd.DataFrame, target_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie et valide les données de mutation, avec une gestion spéciale
    pour les événements cliniques importants comme FLT3_ITD.
    """
    print("Cleaning molecular data...")

    molecular_clean = molecular_df[molecular_df["ID"].isin(target_clean["ID"])].copy()

    molecular_clean["VAF"] = pd.to_numeric(molecular_clean["VAF"], errors="coerce")
    molecular_clean["DEPTH"] = pd.to_numeric(molecular_clean["DEPTH"], errors="coerce")

    # --- LOGIQUE DE FILTRAGE AMÉLIORÉE ---

    # Critère 1: Mutations de haute qualité (avec VAF et DEPTH valides)
    high_quality_mutations = (
        (molecular_clean["VAF"] >= 0)
        & (molecular_clean["VAF"] <= 1)
        & (molecular_clean["DEPTH"] >= 10)
    )

    # Critère 2: Mutations cliniquement cruciales qui doivent être conservées
    # même si certaines métriques de qualité sont manquantes.
    # On identifie les FLT3_ITD par la colonne 'EFFECT' ou 'PROTEIN_CHANGE'.
    is_flt3_itd = (molecular_clean["EFFECT"].astype(str).str.upper() == "ITD") | (
        molecular_clean["PROTEIN_CHANGE"]
        .astype(str)
        .str.contains("ITD", na=False, case=False)
    )

    # Préserver également les événements MLL/KMT2A de type PTD (Partial Tandem Duplication),
    # souvent rapportés sans DEPTH complet mais cliniquement importants.
    gene_str = molecular_clean["GENE"].astype(str).str.upper()
    effect_str = molecular_clean["EFFECT"].astype(str).str.upper()
    protein_str = molecular_clean["PROTEIN_CHANGE"].astype(str).str.upper()
    is_mll_gene = gene_str.isin(["MLL", "KMT2A"])  # synonymes
    has_ptd = effect_str.str.contains("PTD", na=False) | protein_str.str.contains(
        "PTD|MLL_PTD", na=False
    )
    is_mll_ptd = is_mll_gene & has_ptd

    # Une mutation est valide si elle est de haute qualité OU si c'est un FLT3_ITD
    valid_mutations = high_quality_mutations | is_flt3_itd | is_mll_ptd

    # Compter uniquement les mutations supprimées qui n'étaient PAS des FLT3_ITD
    removed_count = (~high_quality_mutations & ~is_flt3_itd & ~is_mll_ptd).sum()

    if removed_count > 0:
        print(
            f"   Removing {removed_count} low-quality mutations (FLT3_ITD's are preserved)"
        )
        molecular_clean = molecular_clean[
            valid_mutations
        ].copy()  # Utiliser .copy() ici
    else:
        print("   -> No low-quality mutations removed.")

    print(
        f"   Clean molecular data: {len(molecular_clean)} mutations across "
        f"{molecular_clean['ID'].nunique()} patients"
    )

    return molecular_clean


def _apply_clinical_ranges(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Applique des plages biologiquement plausibles aux mesures cliniques."""
    if column not in CLINICAL_RANGES:
        return df

    original_count = df[column].notna().sum()
    min_val, max_val = CLINICAL_RANGES[column]

    # Appliquer les bornes
    df.loc[(df[column] < min_val) | (df[column] > max_val), column] = np.nan

    invalid_count = original_count - df[column].notna().sum()
    if invalid_count > 0:
        print(f"   -> {invalid_count} valeurs invalides mises à NaN dans '{column}'")

    return df


def clean_target_data(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy compatibility function for target data cleaning.

    Parameters
    ----------
    target_df : pd.DataFrame
        Raw target data

    Returns
    -------
    pd.DataFrame
        Cleaned target data
    """
    return _clean_survival_data(target_df)
