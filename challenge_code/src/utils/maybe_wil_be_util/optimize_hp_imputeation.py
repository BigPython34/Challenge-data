import pandas as pd
import numpy as np
import optuna
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

# Ignore warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Global cache to avoid redundant preprocessing
_preprocessing_cache = {}


def get_cache_key(strategy, params):
    """Generate a cache key based on strategy and params."""
    if strategy == "knn":
        return f"knn_neighbors_{params.get('n_neighbors')}"
    elif strategy == "iterative":
        return (
            f"iterative_maxiter_{params.get('max_iter')}"
            f"_rf_est_{params.get('rf_n_estimators')}"
            f"_rf_feat_{params.get('rf_max_features'):.3f}"
        )
    return None


def objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    y_full,
    numeric_features,
    categorical_features,
    phase="knn",
):
    """
    Objective function for Optuna.
    Phase can be "knn" or "iterative".
    """

    print(f"\n--- Trial {trial.number} ({phase}) ---")

    # Restrict the strategy based on the phase
    if phase == "knn":
        strategy = "knn"
    elif phase == "iterative":
        strategy = "iterative"
    else:
        raise ValueError("Phase must be either 'knn' or 'iterative'.")

    params = {"strategy": strategy}
    numeric_imputer = None

    if strategy == "knn":
        n_neighbors = trial.suggest_int("n_neighbors", 2, 20, step=1)
        params["n_neighbors"] = n_neighbors
        numeric_imputer = KNNImputer(n_neighbors=n_neighbors)
        print(f"   KNN with n_neighbors={n_neighbors}")

    elif strategy == "iterative":
        max_iter = trial.suggest_int("max_iter", 10, 50)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 50, step=5)
        rf_max_features = trial.suggest_float("rf_max_features", 0.2, 0.7)

        params.update(
            {
                "max_iter": max_iter,
                "rf_n_estimators": rf_n_estimators,
                "rf_max_features": rf_max_features,
            }
        )

        estimator = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_features=rf_max_features,
            random_state=42,
            n_jobs=-1,
        )
        numeric_imputer = IterativeImputer(
            estimator=estimator, max_iter=max_iter, random_state=42
        )
        print(
            f"   Iterative RF: max_iter={max_iter}, "
            f"rf_n_estimators={rf_n_estimators}, "
            f"rf_max_features={rf_max_features:.3f}"
        )

    # Check cache
    cache_key = get_cache_key(strategy, params)
    if cache_key in _preprocessing_cache:
        print(f"   Cache HIT: {cache_key}")
        X_train_processed, X_val_processed = _preprocessing_cache[cache_key]
    else:
        print(f"   Cache MISS: {cache_key}")

        numeric_pipeline = Pipeline(
            steps=[("imputer", numeric_imputer), ("scaler", StandardScaler())]
        )
        categorical_pipeline = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

        try:
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            _preprocessing_cache[cache_key] = (X_train_processed, X_val_processed)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return 0.0

    # Train Cox model and evaluate
    try:
        cox_model = CoxPHSurvivalAnalysis(alpha=0.1)
        cox_model.fit(X_train_processed, y_train)
        predictions = cox_model.predict(X_val_processed)
        c_index, _, _, _, _ = concordance_index_ipcw(y_full, y_val, predictions)
        print(f"   C-Index: {c_index:.5f}")
        return c_index
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def run_phase(
    X_train,
    y_train,
    X_val,
    y_val,
    y,
    numeric_features,
    categorical_features,
    phase,
    n_trials,
):
    """Run a dedicated Optuna study for a given phase."""
    print(f"\n=== Starting phase: {phase.upper()} ===")

    objective_with_data = lambda trial: objective(
        trial,
        X_train,
        y_train,
        X_val,
        y_val,
        y,
        numeric_features,
        categorical_features,
        phase=phase,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_with_data, n_trials=n_trials)

    print("\n" + "=" * 60)
    print(f"PHASE {phase.upper()} COMPLETED")
    print("=" * 60)

    best_trial = study.best_trial
    print(f"  > Best C-Index: {best_trial.value:.5f}")
    print("  > Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"- {key}: {value}")

    return study


def main():
    print("=== HYPERPARAMETER OPTIMIZATION WITH OPTUNA ===")

    _preprocessing_cache.clear()

    # Load dataset
    print("\n1. Loading feature-engineered dataset...")
    try:
        X = pd.read_csv("datasets_featured/X_train_featured.csv")
        y_df = pd.read_csv("datasets_featured/y_train_featured.csv")
        y = Surv.from_arrays(event=y_df["OS_STATUS"], time=y_df["OS_YEARS"])
    except FileNotFoundError:
        print("ERROR: Missing files in 'datasets_featured/'.")
        return

    print(f"   Dataset loaded. Shape: {X.shape}")

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in ["ID"]:
        numeric_features = [f for f in numeric_features if f != col]
        categorical_features = [f for f in categorical_features if f != col]

    print(
        f"   {len(numeric_features)} numeric and {len(categorical_features)} categorical features detected."
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y_df["OS_STATUS"]
    )
    print(f"\n2. Train ({len(X_train)}) / Validation ({len(X_val)}) split done.")

    # Phase 1: KNN search
    study_knn = run_phase(
        X_train,
        y_train,
        X_val,
        y_val,
        y,
        numeric_features,
        categorical_features,
        phase="knn",
        n_trials=30,
    )

    # Phase 2: Iterative RF search
    study_iterative = run_phase(
        X_train,
        y_train,
        X_val,
        y_val,
        y,
        numeric_features,
        categorical_features,
        phase="iterative",
        n_trials=500,
    )

    print("\n=== ALL PHASES COMPLETED ===")
    print(f"Cache used for {len(_preprocessing_cache)} unique configurations.")


if __name__ == "__main__":
    main()
