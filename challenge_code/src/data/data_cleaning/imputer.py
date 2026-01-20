import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import (
    enable_iterative_imputer,
)
from sklearn.linear_model import BayesianRidge
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import os
import joblib
import json
from ...config import PREPROCESSING, SEED


from ...config import PREPROCESSING, SEED

class AdvancedImputer(BaseEstimator, TransformerMixin):
    """
    Imputeur avancé qui est maintenant 100% compatible avec l'API set_output.
    Cette version est corrigée pour être internement cohérente.
    """

    def __init__(self, strategy: str = "iterative", n_neighbors: int = None):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer_ = None
        self.feature_names_in_: List[str] = None  # Initialisation

    def fit(self, X, y=None):
        """Apprend les paramètres d'imputation et stocke les noms de colonnes."""
        if self.strategy == "knn":
            knn_config = PREPROCESSING.get("knn", {})
            n_neighbors = self.n_neighbors if self.n_neighbors is not None else knn_config.get("n_neighbors", 4)
            self.imputer_ = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        elif self.strategy == "iterative":
            iterative_config = PREPROCESSING.get("iterative", {})
            estimator_name = iterative_config.get("estimator", "RandomForest")
            estimator_params = iterative_config.get("estimator_params", {})

            # Supporte plusieurs estimateurs
            if estimator_name == "RandomForest":
                params = {
                    "n_estimators": iterative_config.get("estimator_n_estimators", 70),
                    "random_state": SEED,
                    "n_jobs": iterative_config.get("n_jobs", -1)
                }
                params.update(estimator_params)
                estimator = RandomForestRegressor(**params)
            elif estimator_name == "ExtraTrees":
                params = {
                    "n_estimators": iterative_config.get("estimator_n_estimators", 70),
                    "random_state": SEED,
                    "n_jobs": iterative_config.get("n_jobs", -1)
                }
                params.update(estimator_params)
                estimator = ExtraTreesRegressor(**params)
            elif estimator_name == "BayesianRidge":
                estimator = BayesianRidge()
            elif estimator_name == "HistGradientBoosting":
                
                params = {"random_state": SEED}
                params.update(estimator_params)
                estimator = HistGradientBoostingRegressor(**params)
            else:
                raise ValueError(f"Estimateur non supporté: {estimator_name}")

            self.imputer_ = IterativeImputer(
                estimator=estimator,
                max_iter=iterative_config.get("max_iter", 350),
                random_state=SEED,
                initial_strategy=iterative_config.get("initial_strategy", "median"),
                verbose=iterative_config.get("verbose", 2),
                keep_empty_features=True
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


# --- Supervised imputation for MONOCYTES (train-only model, applied to train/test) ---
def supervised_monocyte_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keep_indicator: bool = False,
    model_path: str = "models/monocyte_imputer.joblib",
    predictors: dict | None = None,
    preprocessing: dict | None = None,
    regressor: dict | None = None,
    clip_to_wbc: bool | None = None,
    winsorize_pct: float | None = None,
    extra_fit_df: pd.DataFrame | None = None,
    include_test_rows: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "MONOCYTES" not in train_df.columns:
        print("[MONO] Colonne MONOCYTES absente. Aucune imputation supervisée.")
        return train_df, test_df

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path) or "models", exist_ok=True)

    # Predictors available in clinical data
    if predictors is None:
        num_candidates = ["WBC", "ANC", "HB", "PLT", "BM_BLAST"]
        cat_candidates = ["CENTER"]
    else:
        num_candidates = predictors.get("num", [])
        cat_candidates = predictors.get("cat", [])

    num_features = [c for c in num_candidates if c in train_df.columns]
    cat_features = [c for c in cat_candidates if c in train_df.columns]

    if not num_features and not cat_features:
        print("[MONO] Aucune feature disponible pour imputer MONOCYTES. Abandon.")
        return train_df, test_df

    feature_cols_for_pool = list(dict.fromkeys(num_features + cat_features + ["MONOCYTES"]))
    fit_pool = train_df.reindex(columns=feature_cols_for_pool).copy()
    extra_rows = 0
    if include_test_rows:
        test_subset = test_df.reindex(columns=feature_cols_for_pool)
        fit_pool = pd.concat([fit_pool, test_subset], ignore_index=True)
        print(f"[MONO] +{len(test_subset)} lignes de test ajoutées pour entraîner l'imputeur.")
    if extra_fit_df is not None:
        extra_subset = extra_fit_df.reindex(columns=feature_cols_for_pool)
        extra_rows = len(extra_subset)
        if extra_rows:
            fit_pool = pd.concat([fit_pool, extra_subset], ignore_index=True)
            print(f"[MONO] +{extra_rows} lignes externes ajoutées pour entraîner l'imputeur.")

    mask_obs = fit_pool["MONOCYTES"].notna()
    if mask_obs.sum() < 50:
        print(
            f"[MONO] Trop peu de valeurs observées ({mask_obs.sum()}) pour imputer. Abandon."
        )
        return train_df, test_df

    X_train = fit_pool.loc[mask_obs, num_features + cat_features].copy()
    y_train = np.log1p(fit_pool.loc[mask_obs, "MONOCYTES"].astype(float).values)

    # Ensure dense output for downstream regressor
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Build numeric preprocessing
    pre_cfg = preprocessing or {"num_imputer": "median", "num_scaler": "standard"}
    num_imputer_strategy = pre_cfg.get("num_imputer", "median")
    num_scaler_kind = pre_cfg.get("num_scaler", "standard")
    scaler = StandardScaler() if num_scaler_kind == "standard" else RobustScaler()
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy=num_imputer_strategy)),
                        ("sc", scaler),
                    ]
                ),
                num_features,
            ),
            ("cat", ohe, cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Build regressor
    reg_cfg = regressor or {
        "type": "HistGradientBoostingRegressor",
        "learning_rate": 0.08,
        "max_depth": None,
        "max_iter": 400,
        "l2_regularization": 0.0,
        "random_state": 42,
    }
    reg_type = reg_cfg.get("type", "HistGradientBoostingRegressor")
    if reg_type != "HistGradientBoostingRegressor":
        raise ValueError(
            f"Unsupported regressor type for MONOCYTES imputer: {reg_type}"
        )
    reg = HistGradientBoostingRegressor(
        learning_rate=reg_cfg.get("learning_rate", 0.08),
        max_depth=reg_cfg.get("max_depth", None),
        max_iter=reg_cfg.get("max_iter", 400),
        l2_regularization=reg_cfg.get("l2_regularization", 0.0),
        random_state=reg_cfg.get("random_state", 42),
    )

    mono_pipe = Pipeline([("pre", pre), ("reg", reg)])
    print(
        f"[MONO] Entraînement de l'imputeur supervisé (n={mask_obs.sum()} lignes observées)..."
    )
    mono_pipe.fit(X_train, y_train)
    joblib.dump(mono_pipe, model_path)

    # Prepare meta for traceability
    meta = {
        "observed_n": int(mask_obs.sum()),
        "predictors": {"num": num_features, "cat": cat_features},
        "preprocessing": {
            "num_imputer": num_imputer_strategy,
            "num_scaler": num_scaler_kind,
        },
        "regressor": reg_cfg,
        "postprocess": {
            "clip_to_wbc": True if clip_to_wbc is None else bool(clip_to_wbc),
            "winsorize_pct": 99.5 if winsorize_pct is None else float(winsorize_pct),
        },
    }

    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        miss_mask = out["MONOCYTES"].isna()
        if miss_mask.any():
            Xp = out.loc[miss_mask, num_features + cat_features].copy()
            y_pred = np.expm1(mono_pipe.predict(Xp))
            # Clip to [0, +inf)
            y_pred = np.clip(y_pred, 0.0, None)
            # Upper bound within WBC if available (physiologic constraint)
            do_clip_to_wbc = True if clip_to_wbc is None else bool(clip_to_wbc)
            if do_clip_to_wbc and "WBC" in out.columns:
                ub = out.loc[miss_mask, "WBC"].astype(float).values
                y_pred = np.minimum(y_pred, np.where(np.isfinite(ub), ub, y_pred))
            # Mild winsorization to limit extremes
            pct = 99.5 if winsorize_pct is None else float(winsorize_pct)
            y_pred = np.clip(y_pred, 0.0, np.nanpercentile(y_pred, pct))
            y_pred = y_pred.astype("float32", copy=False)
            out.loc[miss_mask, "MONOCYTES"] = y_pred
        if keep_indicator:
            out["MONOCYTES_missing"] = df["MONOCYTES"].isna().astype(int)
        else:
            if "MONOCYTES_missing" in out.columns:
                out.drop(columns=["MONOCYTES_missing"], inplace=True, errors="ignore")
        return out

    before_tr = train_df["MONOCYTES"].isna().mean()
    before_te = test_df["MONOCYTES"].isna().mean()
    train_df_imputed = _impute(train_df)
    test_df_imputed = _impute(test_df)
    after_tr = train_df_imputed["MONOCYTES"].isna().mean()
    after_te = test_df_imputed["MONOCYTES"].isna().mean()
    print(f"[MONO] Taux de NA MONOCYTES train: {before_tr:.1%} -> {after_tr:.1%}")
    print(f"[MONO] Taux de NA MONOCYTES test : {before_te:.1%} -> {after_te:.1%}")

    # Save meta sidecar
    try:
        meta.update(
            {
                "na_rates": {
                    "train_before": float(before_tr),
                    "train_after": float(after_tr),
                    "test_before": float(before_te),
                    "test_after": float(after_te),
                }
            }
        )
        meta_path = os.path.splitext(model_path)[0] + "_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[MONO] Impossible d'enregistrer les métadonnées de l'imputeur: {e}")

    return train_df_imputed, test_df_imputed
