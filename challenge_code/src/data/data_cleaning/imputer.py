import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class AdvancedImputer(BaseEstimator, TransformerMixin):
    """
    Imputeur avancé qui est maintenant 100% compatible avec l'API set_output.
    Cette version est corrigée pour être internement cohérente.
    """

    def __init__(self, strategy: str = "iterative", n_neighbors: int = 4):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer_ = None
        self.feature_names_in_: List[str] = None  # Initialisation

    def fit(self, X, y=None):
        """Apprend les paramètres d'imputation et stocke les noms de colonnes."""
        if self.strategy == "knn":
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        elif self.strategy == "iterative":
            estimator = RandomForestRegressor(
                n_estimators=30, random_state=42, n_jobs=-1
            )
            self.imputer_ = IterativeImputer(
                estimator=estimator,
                max_iter=30,
                random_state=42,
                initial_strategy="median",
            )

        self.imputer_.fit(X)

        # Stocker les noms de features vus pendant l'entraînement
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()
        else:  # Gérer le cas où X est un np.array
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        return self

    def transform(self, X) -> pd.DataFrame:
        """Impute les données et garantit la préservation de l'index et des colonnes."""
        if self.imputer_ is None:
            raise RuntimeError(
                "L'imputeur doit être entraîné avant de transformer des données."
            )

        # L'imputeur scikit-learn renvoie un tableau NumPy
        X_imputed_np = self.imputer_.transform(X)

        # Reconstruire le DataFrame en utilisant la bonne variable de classe pour les colonnes
        # et en préservant l'index de l'input X.
        X_imputed_df = pd.DataFrame(
            X_imputed_np,
            columns=self.feature_names_in_,  # <- CORRECTION: Utilisation de la bonne variable
            index=X.index,
        )
        return X_imputed_df

    def set_output(self, transform=None):
        """Méthode requise pour la compatibilité avec set_output("pandas")."""
        return self

    def get_feature_names_out(self, input_features=None):
        """Méthode requise pour propager les noms de colonnes."""
        return self.feature_names_in_
