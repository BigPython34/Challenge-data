"""
Advanced imputation module for post-feature-engineered AML data.

This module provides robust, scikit-learn compatible imputation strategies,
designed to be used AFTER comprehensive feature engineering. It ensures no
data leakage between training and testing sets by adhering to the fit/transform
paradigm.
"""

import pandas as pd
import numpy as np
from typing import List
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import (
    RandomForestRegressor,
)  # Un estimateur plus puissant pour l'imputation


class AdvancedImputer:
    """
    A scikit-learn compatible imputer for post-feature-engineered data.

    This class wraps standard but powerful imputation strategies like KNN and
    IterativeImputer, ensuring they are correctly fitted only on training
    data and then applied to any new data.

    Parameters
    ----------
    strategy : str, optional
        The imputation strategy to use. Supported: 'knn', 'iterative'.
        Defaults to 'iterative'.
    n_neighbors : int, optional
        Number of neighbors to use for 'knn' imputation. Defaults to 5.
    """

    def __init__(self, strategy: str = "iterative", n_neighbors: int = 5):
        if strategy not in ["knn", "iterative"]:
            raise ValueError("Strategy must be either 'knn' or 'iterative'")

        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer_ = None  # L'imputeur scikit-learn sera stocké ici
        self.trained_columns_: List[str] = None  # Pour garder l'ordre des colonnes

    def fit(self, X, y=None):
        """
        Fit the imputer on the training data.

        Learns the parameters for imputation (e.g., neighbors for KNN, regression
        models for IterativeImputer) from the training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The training data with missing values.
        """
        print(f"Fitting AdvancedImputer with '{self.strategy}' strategy...")
        if hasattr(X, "columns"):
            self.trained_columns_ = X.columns.tolist()
        else:
            self.trained_columns_ = [f"col_{i}" for i in range(X.shape[1])]

        if self.strategy == "knn":
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)

        elif self.strategy == "iterative":
            estimator = RandomForestRegressor(
                n_estimators=10, random_state=42, n_jobs=-1
            )
            self.imputer_ = IterativeImputer(
                estimator=estimator,
                max_iter=10,
                random_state=42,
                initial_strategy="median",
                imputation_order="ascending",  # ordre d'imputation par nombre de NaN
            )

        self.imputer_.fit(X)
        print("Imputer fitted successfully.")
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Impute missing values using the learned parameters.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The data to transform (can be train or test data).

        Returns
        -------
        pd.DataFrame
            The dataframe with missing values imputed.
        """
        if self.imputer_ is None:
            raise RuntimeError("You must fit the imputer before transforming data.")

        # S'assurer que les colonnes sont dans le même ordre que pendant le fit
        if hasattr(X, "columns"):
            X_reordered = X[self.trained_columns_]
            print(f"Transforming data... ({X.isna().sum().sum()} missing values found)")
            X_imputed_np = self.imputer_.transform(X_reordered)
            # Reconstruire le DataFrame
            X_imputed_df = pd.DataFrame(
                X_imputed_np, columns=self.trained_columns_, index=X.index
            )
        else:
            # X is ndarray, assume columns already match order
            print(f"Transforming ndarray data... (cannot count missing values)")
            X_imputed_np = self.imputer_.transform(X)
            X_imputed_df = pd.DataFrame(X_imputed_np, columns=self.trained_columns_)
        print(
            f"Transformation complete. Remaining missing values: {X_imputed_df.isna().sum().sum()}"
        )
        return X_imputed_df
