# Fichier: src/modeling/pipeline_components.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Définition des classes personnalisées ---
# (Ces classes étaient définies localement dans votre script 1_prepare_data.py)


class AdvancedImputer(BaseEstimator, TransformerMixin):
    """Classe d'imputation avancée (KNN ou Itérative)."""

    def __init__(self, strategy: str = "iterative", n_neighbors: int = 5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        self.columns_ = X.columns
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
            )
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        X_imputed = self.imputer_.transform(X)
        return pd.DataFrame(X_imputed, columns=self.columns_, index=X.index)


class ClipQuantiles(BaseEstimator, TransformerMixin):
    """Clippe les colonnes selon des quantiles appris."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self.lower_bounds_ = X.quantile(self.lower)
        self.upper_bounds_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)


# --- Fonction qui assemble et retourne la pipeline de prétraitement ---


def get_preprocessing_pipeline(X_train_df: pd.DataFrame) -> ColumnTransformer:
    """
    Construit et retourne le ColumnTransformer pour le prétraitement des données.
    """
    # Identifier les types de colonnes à partir du dataframe d'entrée
    numeric_features = X_train_df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(
        f"Preprocessing pipeline created with {len(numeric_features)} numeric and {len(categorical_features)} categorical features."
    )

    # Pipeline pour les données numériques
    numeric_transformer = Pipeline(
        steps=[
            ("clip", ClipQuantiles(lower=0.01, upper=0.99)),
            (
                "imputer",
                AdvancedImputer(strategy="iterative"),
            ),  # Utilise la stratégie la plus puissante
            ("scaler", RobustScaler()),
        ]
    )

    # Pipeline pour les données catégorielles
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Assemblage avec ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Garde les colonnes non spécifiées (au cas où)
    )

    return preprocessor
