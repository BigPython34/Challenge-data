import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

from ..config import SEED, FEATURE_LIST

from .data_cleaning.cleaner import clean_and_validate_data
from .data_cleaning.imputer import ClinicalImputer, ImputationStrategy

from .features.feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
    MolecularFeatureExtraction,
    IntegratedFeatureEngineering,
)


class DataPreparationPipeline:
    """
    Modular pipeline for AML survival dataset preparation.
    """

    def __init__(self, test_size: float = 0.2, seed: int = SEED):
        self.test_size = test_size
        self.seed = seed

    def clean_data(
        self,
        clinical_df: pd.DataFrame,
        molecular_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Clean and validate clinical, molecular, and target data.
        """
        return clean_and_validate_data(clinical_df, molecular_df, target_df)

    def feature_engineering(
        self, clinical_clean: pd.DataFrame, molecular_clean: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform feature engineering using updated feature engineering modules.
        """
        clinical_features = ClinicalFeatureEngineering.create_clinical_features(
            clinical_clean
        )
        cyto_features = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
            clinical_features
        )
        molecular_features = MolecularFeatureExtraction.extract_molecular_risk_features(
            clinical_features, molecular_clean
        )
        burden_features = MolecularFeatureExtraction.create_molecular_burden_features(
            molecular_clean
        )

        enriched_df = IntegratedFeatureEngineering.combine_all_features(
            clinical_features, molecular_features, burden_features, cyto_features
        )

        return enriched_df

    def impute_data(self, enriched_df: pd.DataFrame, strategy: ImputationStrategy):
        """
        Impute missing data using intelligent clinical imputation.
        """
        return ClinicalImputer.intelligent_clinical_imputation(enriched_df, strategy)

    def split_data(self, final_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        """
        train_df, test_df = train_test_split(
            final_df,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=final_df["OS_STATUS"],
        )
        return train_df, test_df

    def prepare_survival_targets(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare survival targets for training and test sets.
        """
        train_y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", train_df)
        test_y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", test_df)
        train_df["y_survival"] = [train_y[i] for i in range(len(train_y))]
        test_df["y_survival"] = [test_y[i] for i in range(len(test_y))]
        return train_df, test_df

    def prepare_survival_dataset(
        self,
        clinical_df: pd.DataFrame,
        molecular_df: pd.DataFrame,
        target_df: pd.DataFrame,
        use_advanced_features: bool = True,
        imputation_strategy: ImputationStrategy = ImputationStrategy.MEDICAL_INFORMED,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Complete pipeline for preparing AML survival dataset.
        """
        print("🏥 === AML SURVIVAL DATASET PREPARATION ===")

        clinical_clean, molecular_clean, target_clean = self.clean_data(
            clinical_df, molecular_df, target_df
        )

        if use_advanced_features:
            enriched_df = self.feature_engineering(clinical_clean, molecular_clean)
        else:
            enriched_df = clinical_clean.copy()

        enriched_df, imputation_metadata = self.impute_data(
            enriched_df, imputation_strategy
        )

        final_df = enriched_df.merge(target_clean, on="ID", how="inner")

        train_df, test_df = self.split_data(final_df)

        train_df, test_df = self.prepare_survival_targets(train_df, test_df)

        metadata = {
            "n_patients_initial": len(clinical_df),
            "n_patients_final": len(final_df),
            "n_mutations": len(molecular_clean),
            "n_features": final_df.shape[1] - 4,
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
            clinical_features = ClinicalFeatureEngineering.create_clinical_features(
                clinical_clean
            )
            cyto_features = (
                CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
                    clinical_features
                )
            )
            molecular_features = (
                MolecularFeatureExtraction.extract_molecular_risk_features(
                    clinical_features, molecular_clean
                )
            )
            burden_features = (
                MolecularFeatureExtraction.create_molecular_burden_features(
                    molecular_clean
                )
            )

            enriched_df = IntegratedFeatureEngineering.combine_all_features(
                clinical_features, molecular_features, burden_features, cyto_features
            )

            # Imputation
            strategy = ImputationStrategy[advanced_imputation_method.upper()]
            enriched_df, imputation_metadata = (
                ClinicalImputer.intelligent_clinical_imputation(enriched_df, strategy)
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

            clinical_features = ClinicalFeatureEngineering.create_clinical_features(
                clinical_clean
            )
            cyto_features = (
                CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
                    clinical_features
                )
            )
            molecular_features = (
                MolecularFeatureExtraction.extract_molecular_risk_features(
                    clinical_features, molecular_clean
                )
            )
            burden_features = (
                MolecularFeatureExtraction.create_molecular_burden_features(
                    molecular_clean
                )
            )

            enriched_df = IntegratedFeatureEngineering.combine_all_features(
                clinical_features, molecular_features, burden_features, cyto_features
            )

            strategy = ImputationStrategy[advanced_imputation_method.upper()]
            enriched_df, imputation_metadata = (
                ClinicalImputer.intelligent_clinical_imputation(enriched_df, strategy)
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
        clinical_features = ClinicalFeatureEngineering.create_clinical_features(
            enriched_df
        )
        cyto_features = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
            clinical_features
        )
        molecular_features = MolecularFeatureExtraction.extract_molecular_risk_features(
            clinical_features, molecular_df
        )
        burden_features = MolecularFeatureExtraction.create_molecular_burden_features(
            molecular_df
        )

        enriched_df = IntegratedFeatureEngineering.combine_all_features(
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


def prepare_test_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Prepares test data by applying transformations without removing patients.

    Args:
        data: Raw test data.

    Returns:
        DataFrame: Prepared test data.
    """
    print("\n=== PREPARING TEST DATA ===")

    clinical_features = ClinicalFeatureEngineering.create_clinical_features(
        data["clinical_test"]
    )
    cyto_features = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
        data["clinical_test"]
    )
    molecular_features = MolecularFeatureExtraction.extract_molecular_risk_features(
        data["clinical_test"], data["molecular_test"]
    )
    burden_features = MolecularFeatureExtraction.create_molecular_burden_features(
        data["molecular_test"]
    )

    dataset_test = IntegratedFeatureEngineering.combine_all_features(
        clinical_features, molecular_features, burden_features, cyto_features
    )

    dataset_test, _ = ClinicalImputer.intelligent_clinical_imputation(
        dataset_test, strategy=ImputationStrategy.MEDICAL_INFORMED
    )

    return dataset_test
