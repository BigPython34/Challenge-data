"""
Data processing module for AML survival analysis.

This module provides a clean separation between:
- Data cleaning and validation (data_cleaning/)
- Feature engineering (features/)
- Pipeline orchestration (prepare.py)

Main functions:
- prepare_survival_dataset: Complete survival analysis pipeline
- prepare_enriched_dataset: Legacy compatibility function
"""

# Main pipeline functions
from .prepare import (
    prepare_survival_dataset,
    prepare_enriched_dataset,
    get_survival_feature_sets,
    prepare_features_and_target,
    clean_target_data,
)

# Data cleaning functions
from .data_cleaning import (
    clean_and_validate_data,
    intelligent_clinical_imputation,
    medical_imputation_strategy,
    ImputationStrategy,
)

# Feature engineering functions
from .features import (
    create_clinical_features,
    extract_cytogenetic_risk_features,
    extract_molecular_risk_features,
    create_molecular_burden_features,
    combine_all_features,
    get_clean_feature_lists,
)

# Data loading (existing)
from .load import load_all_data

__all__ = [
    # Main pipeline
    "prepare_survival_dataset",
    "prepare_enriched_dataset",
    "get_survival_feature_sets",
    "prepare_features_and_target",
    "clean_target_data",
    # Data cleaning
    "clean_and_validate_data",
    "intelligent_clinical_imputation",
    "medical_imputation_strategy",
    "ImputationStrategy",
    # Feature engineering
    "create_clinical_features",
    "extract_cytogenetic_risk_features",
    "extract_molecular_risk_features",
    "create_molecular_burden_features",
    "combine_all_features",
    "get_clean_feature_lists",
    # Data loading
    "load_all_data",
]
