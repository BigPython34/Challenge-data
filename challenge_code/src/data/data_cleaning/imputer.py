"""
Advanced imputation strategies for AML clinical data.

This module provides medical domain-informed imputation methods
that consider the biological relationships between clinical measurements.
Refactored for modularity and maintainability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.experimental import enable_iterative_imputer
from enum import Enum
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge


class ImputationStrategy(Enum):
    """Available imputation strategies for different data types."""

    MEDIAN = "median"
    MEAN = "mean"
    KNN = "knn"
    ITERATIVE = "iterative"
    MEDICAL_INFORMED = "medical_informed"
    REGRESSION = "regression"


class ClinicalImputer:
    """Handles intelligent clinical imputation."""

    @staticmethod
    def intelligent_clinical_imputation(
        clinical_df: pd.DataFrame,
        strategy: ImputationStrategy = ImputationStrategy.MEDICAL_INFORMED,
    ) -> tuple[pd.DataFrame, Dict]:
        """
        Perform intelligent imputation of clinical data using medical knowledge.

        Parameters
        ----------
        clinical_df : pd.DataFrame
            Clinical data with missing values
        strategy : ImputationStrategy
            The imputation strategy to use

        Returns
        -------
        tuple[pd.DataFrame, Dict]
            Imputed dataframe and imputation metadata
        """
        print("=== INTELLIGENT CLINICAL IMPUTATION ===")

        df_imputed = clinical_df.copy()
        imputation_metadata = {
            "method": strategy.value,
            "columns_imputed": [],
            "strategy_details": {},
        }

        if strategy == ImputationStrategy.MEDICAL_INFORMED:
            df_imputed, metadata = MedicalImputer.medical_informed_imputation(
                df_imputed
            )
            imputation_metadata.update(metadata)
        else:
            df_imputed, metadata = StandardImputer.standard_imputation(
                df_imputed, strategy
            )
            imputation_metadata.update(metadata)

        # Final cleanup - ensure no NaN values remain
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        remaining_nans = df_imputed[numeric_cols].isna().sum().sum()

        if remaining_nans > 0:
            print(
                f"   Final cleanup: {remaining_nans} remaining NaN values filled with 0"
            )
            df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(0)

        print("Imputation completed")
        print(f"   Columns imputed: {imputation_metadata['columns_imputed']}")
        print(f"   Remaining missing values: {df_imputed.isna().sum().sum()}")

        return df_imputed, imputation_metadata


class MedicalImputer:
    """Handles medical domain-informed imputation."""

    @staticmethod
    def medical_informed_imputation(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
        """
        Apply medical domain-informed imputation strategies.

        Parameters
        ----------
        df : pd.DataFrame
            Clinical data to impute

        Returns
        -------
        tuple[pd.DataFrame, Dict]
            Imputed data and metadata
        """
        df_imputed = df.copy()
        metadata = {
            "method": "medical_informed",
            "columns_imputed": [],
            "strategy_details": {},
        }

        # Create cytopenia context for informed imputation
        cytopenia_context = MedicalImputer._create_cytopenia_context(df_imputed)

        # Medical imputation order based on clinical dependencies
        imputation_order = ["WBC", "ANC", "MONOCYTES", "PLT", "HB", "BM_BLAST"]

        for column in imputation_order:
            if column in df_imputed.columns and df_imputed[column].isna().any():
                df_imputed[column] = MedicalImputer.medical_imputation_strategy(
                    df_imputed, column, cytopenia_context
                )
                metadata["columns_imputed"].append(column)
                metadata["strategy_details"][column] = "medical_informed"

        # Handle categorical variables
        MedicalImputer._handle_categorical_variables(df_imputed, metadata)

        return df_imputed, metadata

    @staticmethod
    def medical_imputation_strategy(
        df: pd.DataFrame, column: str, patient_context: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Apply medical domain-informed imputation for a specific clinical measurement.

        Parameters
        ----------
        df : pd.DataFrame
            Clinical data
        column : str
            Column name to impute
        patient_context : pd.DataFrame, optional
            Additional patient context for informed imputation

        Returns
        -------
        pd.Series
            Imputed values for the column
        """
        values = df[column].copy()
        missing_mask = values.isna()

        if not missing_mask.any():
            return values

        print(
            f"   🏥 Medical imputation for {column} ({missing_mask.sum()} missing values)"
        )

        if column == "BM_BLAST":
            values = MedicalImputer._impute_bone_marrow_blasts(df, values, missing_mask)
        elif column in ["WBC", "ANC", "MONOCYTES", "PLT"]:
            values = MedicalImputer._impute_cell_counts(
                df, column, values, missing_mask
            )
        elif column == "HB":
            values = MedicalImputer._impute_hemoglobin(
                df, values, missing_mask, patient_context
            )
        else:
            values.fillna(values.median(), inplace=True)

        print(f"      {missing_mask.sum()} values imputed")
        return values

    @staticmethod
    def _handle_categorical_variables(df_imputed: pd.DataFrame, metadata: Dict):
        """Handle categorical variables during medical imputation."""
        if "CENTER" in df_imputed.columns:
            df_imputed["CENTER"] = df_imputed["CENTER"].fillna("Unknown")
            metadata["columns_imputed"].append("CENTER")

        if "CYTOGENETICS" in df_imputed.columns:
            df_imputed["CYTOGENETICS"] = df_imputed["CYTOGENETICS"].fillna("46,XX")
            metadata["columns_imputed"].append("CYTOGENETICS")

    @staticmethod
    def _create_cytopenia_context(df: pd.DataFrame) -> pd.DataFrame:
        """Create cytopenia context for informed imputation."""
        context = pd.DataFrame(index=df.index)
        context["cytopenia_context"] = ((df["PLT"] < 100) | (df["ANC"] < 1.5)).fillna(
            False
        )
        return context

    @staticmethod
    def _impute_bone_marrow_blasts(
        df: pd.DataFrame, values: pd.Series, missing_mask: pd.Series
    ) -> pd.Series:
        """Impute bone marrow blast percentage using medical knowledge."""
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

        return values

    @staticmethod
    def _impute_cell_counts(
        df: pd.DataFrame, column: str, values: pd.Series, missing_mask: pd.Series
    ) -> pd.Series:
        """Impute cell counts using regression if other counts are available."""
        other_counts = ["WBC", "ANC", "MONOCYTES", "PLT"]
        other_counts = [c for c in other_counts if c in df.columns and c != column]

        if len(other_counts) >= 2:
            X = df[other_counts].fillna(df[other_counts].median())
            y = values.dropna()

            if len(y) > 10:
                try:
                    reg = LinearRegression()
                    X_train = X.loc[y.index]
                    reg.fit(X_train, y)

                    X_missing = X.loc[missing_mask]
                    predictions = reg.predict(X_missing)

                    predictions = np.maximum(predictions, 0.1)
                    values.loc[missing_mask] = predictions

                except ValueError as e:
                    print(f"      Regression failed for {column}, using median: {e}")
                    values.fillna(values.median(), inplace=True)
            else:
                values.fillna(values.median(), inplace=True)
        else:
            values.fillna(values.median(), inplace=True)

        return values

    @staticmethod
    def _impute_hemoglobin(
        df: pd.DataFrame,
        values: pd.Series,
        missing_mask: pd.Series,
        patient_context: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Impute hemoglobin considering anemia context."""
        median_val = values.median()

        if (
            patient_context is not None
            and "cytopenia_context" in patient_context.columns
        ):
            cytopenia_mask = patient_context["cytopenia_context"] & missing_mask
            normal_mask = ~patient_context["cytopenia_context"] & missing_mask

            anemic_median = values[
                patient_context["cytopenia_context"].fillna(False)
            ].median()
            normal_median = values[
                ~patient_context["cytopenia_context"].fillna(False)
            ].median()

            if not np.isnan(anemic_median):
                values.loc[cytopenia_mask] = anemic_median
            if not np.isnan(normal_median):
                values.loc[normal_mask] = normal_median
        else:
            values.fillna(median_val, inplace=True)

        return values


class StandardImputer:
    """Handles standard statistical imputation methods."""

    @staticmethod
    def standard_imputation(
        df: pd.DataFrame, strategy: ImputationStrategy
    ) -> tuple[pd.DataFrame, Dict]:
        """
        Apply standard statistical imputation methods.

        Parameters
        ----------
        df : pd.DataFrame
            Data to impute
        strategy : ImputationStrategy
            Imputation method to use

        Returns
        -------
        tuple[pd.DataFrame, Dict]
            Imputed data and metadata
        """
        df_imputed = df.copy()
        metadata = {
            "method": strategy.value,
            "columns_imputed": [],
            "strategy_details": {},
        }

        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(
            include=["object", "category"]
        ).columns

        if len(numeric_cols) > 0:
            imputer = StandardImputer._get_numeric_imputer(strategy)
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
            metadata["columns_imputed"].extend(numeric_cols.tolist())

        for col in categorical_cols:
            if df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna("Unknown")
                metadata["columns_imputed"].append(col)

        return df_imputed, metadata

    @staticmethod
    def _get_numeric_imputer(strategy: ImputationStrategy):
        """Get the appropriate numeric imputer based on strategy."""
        if strategy == ImputationStrategy.MEDIAN:
            return SimpleImputer(strategy="median")
        elif strategy == ImputationStrategy.MEAN:
            return SimpleImputer(strategy="mean")
        elif strategy == ImputationStrategy.KNN:
            return KNNImputer(n_neighbors=5)
        elif strategy == ImputationStrategy.ITERATIVE:
            return IterativeImputer(
                estimator=BayesianRidge(), random_state=42, max_iter=10
            )
        else:
            return SimpleImputer(strategy="median")
