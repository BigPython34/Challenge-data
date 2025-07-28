"""
Data preparation orchestrator for AML survival analysis.

This module serves as the main orchestrator for data preparation, coordinating
between specialized modules for cleaning, imputation, and feature engineering.

Main functions:
- prepare_survival_dataset: Complete pipeline for survival analysis
- prepare_enriched_dataset: Legacy compatibility function
- get_survival_feature_sets: Predefined feature sets for different use cases

Architecture:
- Data cleaning: src.data.data_cleaning
- Feature engineering: src.data.features
- Coordination and integration: this module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

from ..config import SEED, FEATURE_LIST

from .data_cleaning import (
    clean_and_validate_data,
    intelligent_clinical_imputation,
    ImputationStrategy,
)
from .features import (
    create_clinical_features,
    extract_cytogenetic_risk_features,
    extract_molecular_risk_features,
    create_molecular_burden_features,
    combine_all_features,
)


def prepare_survival_dataset(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    target_df: pd.DataFrame,
    test_size: float = 0.2,
    use_advanced_features: bool = True,
    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDICAL_INFORMED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete pipeline for survival dataset preparation.

    This function orchestrates the entire data preparation process:
    1. Data cleaning and validation
    2. Medical feature engineering
    3. Intelligent imputation
    4. Train/test split
    5. Survival target preparation

    Parameters
    ----------
    clinical_df, molecular_df, target_df : pd.DataFrame
        Raw input dataframes
    test_size : float
        Proportion for test set
    use_advanced_features : bool
        Whether to use advanced feature engineering
    imputation_strategy : ImputationStrategy
        Strategy for handling missing values

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        train_df, test_df, metadata
    """
    print("🏥 === AML SURVIVAL DATASET PREPARATION ===")

    # ===== 1. DATA CLEANING AND VALIDATION =====
    clinical_clean, molecular_clean, target_clean = clean_and_validate_data(
        clinical_df, molecular_df, target_df
    )

    # ===== 2. MEDICAL FEATURE ENGINEERING =====
    if use_advanced_features:
        print("\n=== MEDICAL FEATURE ENGINEERING ===")

        # Clinical features
        clinical_features = create_clinical_features(clinical_clean)
        print(f"Clinical features: {len(clinical_features.columns)} variables")

        # Cytogenetic features
        cyto_features = extract_cytogenetic_risk_features(clinical_features)
        print(f"Cytogenetic features: {len(cyto_features.columns)} variables")

        # Molecular features
        molecular_features = extract_molecular_risk_features(
            clinical_features, molecular_clean
        )
        burden_features = create_molecular_burden_features(molecular_clean)
        print(
            f"Molecular features: {len(molecular_features.columns)} + {len(burden_features.columns)} variables"
        )

        # Combine all features
        enriched_df = combine_all_features(
            clinical_features, molecular_features, burden_features, cyto_features
        )
        print(f"Enriched dataset: {enriched_df.shape}")

    else:
        # Simple version
        enriched_df = clinical_clean.copy()

    # ===== 3. INTELLIGENT IMPUTATION =====
    enriched_df, imputation_metadata = intelligent_clinical_imputation(
        enriched_df, strategy=imputation_strategy
    )

    # ===== 4. MERGE WITH TARGETS =====
    final_df = enriched_df.merge(target_clean, on="ID", how="inner")
    print(f"Final dataset: {final_df.shape}")

    # ===== 5. STRATIFIED TRAIN/TEST SPLIT =====
    # Stratify on survival status to balance events
    train_df, test_df = train_test_split(
        final_df, test_size=test_size, random_state=SEED, stratify=final_df["OS_STATUS"]
    )

    print("Split completed:")
    print(
        f"   Train: {len(train_df)} patients ({train_df['OS_STATUS'].mean():.1%} events)"
    )
    print(
        f"   Test:  {len(test_df)} patients ({test_df['OS_STATUS'].mean():.1%} events)"
    )

    # ===== 6. PREPARE SURVIVAL STRUCTURES =====
    # Create structured arrays for scikit-survival
    train_y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", train_df)
    test_y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", test_df)

    # Add to DataFrames for compatibility
    train_df["y_survival"] = [train_y[i] for i in range(len(train_y))]
    test_df["y_survival"] = [test_y[i] for i in range(len(test_y))]

    # ===== 7. METADATA COMPILATION =====
    metadata = {
        "n_patients_initial": len(clinical_df),
        "n_patients_final": len(final_df),
        "n_mutations": len(molecular_clean),
        "n_features": final_df.shape[1]
        - 4,  # Minus ID, OS_YEARS, OS_STATUS, y_survival
        "train_size": len(train_df),
        "test_size": len(test_df),
        "event_rate_train": train_df["OS_STATUS"].mean(),
        "event_rate_test": test_df["OS_STATUS"].mean(),
        "median_followup": final_df["OS_YEARS"].median(),
        "imputation_metadata": imputation_metadata,
        "feature_engineering": use_advanced_features,
        "imputation_strategy": imputation_strategy.value,
    }

    print("\nPREPARATION COMPLETED")
    print(
        f"   {metadata['n_patients_final']} patients, {metadata['n_features']} features"
    )
    print(f"   {metadata['n_mutations']} mutations analyzed")
    print(f"   Median follow-up: {metadata['median_followup']:.1f} years")

    return train_df, test_df, metadata


def get_survival_feature_sets() -> Dict[str, List[str]]:
    """
    Define feature sets for different levels of analysis.

    Returns
    -------
    Dict[str, List[str]]
        Feature sets organized by complexity/clinical usage
    """
    clean_lists = FEATURE_LIST

    return {
        # Minimal set - for routine clinical use
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
        # ELN 2017 standard - standard for AML prognosis
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
        # Complete set - research/development
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
        # Balanced set - performance/interpretability balance
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


# ===== LEGACY COMPATIBILITY FUNCTIONS =====


def prepare_enriched_dataset(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    target_df: Optional[pd.DataFrame] = None,
    imputer=None,
    advanced_imputation_method: str = "medical",
    is_training: bool = True,
    save_to_file: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, Dict], pd.DataFrame]:
    """
    Legacy compatibility function with the old pipeline.

    Uses the new medical imputation system by default,
    but can fall back to the old system if necessary.

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Clinical data
    molecular_df : pd.DataFrame
        Molecular data
    target_df : pd.DataFrame, optional
        Target data (optional for test)
    imputer : optional
        Pre-trained imputer (for test)
    advanced_imputation_method : str
        Imputation method
    is_training : bool
        Training or test mode
    save_to_file : str, optional
        Path to save prepared dataset

    Returns
    -------
    Union[Tuple[pd.DataFrame, Dict], pd.DataFrame]
        Prepared dataset and metadata (training) or just dataset (test)
    """
    if is_training:
        # New pipeline for training
        if target_df is not None:
            # With targets, use complete pipeline but without split
            clinical_clean, molecular_clean, target_clean = clean_and_validate_data(
                clinical_df, molecular_df, target_df
            )

            # Feature engineering
            clinical_features = create_clinical_features(clinical_clean)
            cyto_features = extract_cytogenetic_risk_features(clinical_features)
            molecular_features = extract_molecular_risk_features(
                clinical_features, molecular_clean
            )
            burden_features = create_molecular_burden_features(molecular_clean)

            enriched_df = combine_all_features(
                clinical_features, molecular_features, burden_features, cyto_features
            )

            # Imputation
            strategy = (
                ImputationStrategy.MEDICAL_INFORMED
                if advanced_imputation_method == "medical"
                else ImputationStrategy.MEDIAN
            )
            enriched_df, imputation_metadata = intelligent_clinical_imputation(
                enriched_df, strategy
            )

            # Final imputation to eliminate all NaN
            numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns
            enriched_df[numeric_cols] = enriched_df[numeric_cols].fillna(0)
            if enriched_df.isnull().sum().sum() > 0:
                enriched_df = enriched_df.fillna(0)

            # Merge with targets
            final_df = enriched_df.merge(target_clean, on="ID", how="inner")

            # Save if requested
            if save_to_file:
                print(f"   Saving enriched dataset to: {save_to_file}")
                final_df.to_csv(save_to_file, index=False)

            return final_df, imputation_metadata
        else:
            # Version without target (for compatibility)
            clinical_clean, molecular_clean, _ = clean_and_validate_data(
                clinical_df,
                molecular_df,
                pd.DataFrame(
                    {"ID": clinical_df["ID"], "OS_YEARS": 1.0, "OS_STATUS": True}
                ),
            )

            clinical_features = create_clinical_features(clinical_clean)
            cyto_features = extract_cytogenetic_risk_features(clinical_features)
            molecular_features = extract_molecular_risk_features(
                clinical_features, molecular_clean
            )
            burden_features = create_molecular_burden_features(molecular_clean)

            enriched_df = combine_all_features(
                clinical_features, molecular_features, burden_features, cyto_features
            )

            strategy = (
                ImputationStrategy.MEDICAL_INFORMED
                if advanced_imputation_method == "medical"
                else ImputationStrategy.MEDIAN
            )
            enriched_df, imputation_metadata = intelligent_clinical_imputation(
                enriched_df, strategy
            )

            # Final imputation to eliminate all NaN
            numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns
            enriched_df[numeric_cols] = enriched_df[numeric_cols].fillna(0)
            if enriched_df.isnull().sum().sum() > 0:
                enriched_df = enriched_df.fillna(0)

            # Save if requested
            if save_to_file:
                print(f"   Saving enriched dataset to: {save_to_file}")
                enriched_df.to_csv(save_to_file, index=False)

            return enriched_df, imputation_metadata
    else:
        # Test mode - use provided imputer (compatibility)
        if imputer is None:
            raise ValueError("Imputer required for test data")

        # Simplified pipeline for test
        enriched_df = clinical_df.copy()

        # Apply same transformations as in training
        clinical_features = create_clinical_features(enriched_df)
        cyto_features = extract_cytogenetic_risk_features(clinical_features)
        molecular_features = extract_molecular_risk_features(
            clinical_features, molecular_df
        )
        burden_features = create_molecular_burden_features(molecular_df)

        enriched_df = combine_all_features(
            clinical_features, molecular_features, burden_features, cyto_features
        )

        # Final imputation to eliminate all NaN
        numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns
        enriched_df[numeric_cols] = enriched_df[numeric_cols].fillna(0)
        if enriched_df.isnull().sum().sum() > 0:
            enriched_df = enriched_df.fillna(0)

        # Simple imputation with saved metadata
        for col in imputer.get("columns_imputed", []):
            if col in enriched_df.columns and enriched_df[col].isna().any():
                enriched_df[col] = enriched_df[col].fillna(enriched_df[col].median())

        # Save if requested
        if save_to_file:
            print(f"   Saving test enriched dataset to: {save_to_file}")
            enriched_df.to_csv(save_to_file, index=False)

        return enriched_df


def clean_target_data(target_df: pd.DataFrame) -> pd.DataFrame:
    """Legacy compatibility function for target data cleaning."""
    from .data_cleaning.cleaner import _clean_survival_data

    return _clean_survival_data(target_df)


def prepare_features_and_target(
    df_enriched: pd.DataFrame, target_df: pd.DataFrame, test_size: float = 0.2
):
    """Legacy compatibility function with the old format."""
    # Merge with targets
    final_df = df_enriched.merge(target_df, on="ID", how="inner")

    # Clean duplicate columns
    if "OS_YEARS_y" in final_df.columns:
        final_df = final_df.drop(columns=["OS_YEARS_x", "OS_STATUS_x"])
        final_df = final_df.rename(
            columns={"OS_YEARS_y": "OS_YEARS", "OS_STATUS_y": "OS_STATUS"}
        )

    # Split - use appropriate status column
    status_col = None
    for col in ["STATUS", "OS_STATUS", "Event", "event"]:
        if col in final_df.columns:
            status_col = col
            break

    if status_col is None:
        raise ValueError(
            f"No status column found. Available columns: {list(final_df.columns)}"
        )

    train_df, test_df = train_test_split(
        final_df, test_size=test_size, random_state=SEED, stratify=final_df[status_col]
    )

    # Prepare features (exclude metadata columns)
    exclude_cols = ["ID", "OS_YEARS", "OS_STATUS", "CENTER", "CYTOGENETICS"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Prepare survival targets
    y_train = Surv.from_dataframe("OS_STATUS", "OS_YEARS", train_df)
    y_test = Surv.from_dataframe("OS_STATUS", "OS_YEARS", test_df)

    return X_train, X_test, y_train, y_test, feature_cols
