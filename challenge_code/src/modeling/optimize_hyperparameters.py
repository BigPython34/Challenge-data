import numpy as np
import os
import joblib
import csv
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import optuna
from sksurv.metrics import concordance_index_ipcw
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
import warnings

# Import local config
try:
    from ..config import SEED, MODEL_DIR
except ImportError:
    # Fallback si import relatif échoue
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.config import SEED, MODEL_DIR

warnings.filterwarnings("ignore")


def find_latest_study(model_type="rsf"):
    """
    Find the most recent study file for a given model type.
    Returns the path to the study file or None if not found.
    """
    if not os.path.exists(MODEL_DIR):
        return None

    pattern = f"{model_type}_study_*.pkl"
    study_files = []

    for file in os.listdir(MODEL_DIR):
        if file.startswith(f"{model_type}_study_") and file.endswith(".pkl"):
            file_path = os.path.join(MODEL_DIR, file)
            study_files.append((file_path, os.path.getctime(file_path)))

    if study_files:
        # Return the most recent file
        latest_file = max(study_files, key=lambda x: x[1])[0]
        return latest_file

    return None


def resume_or_start_rsf_optimization(
    X_train, y_train, n_trials=50, n_splits=5, auto_resume=True
):
    """
    Smart function that automatically resumes the latest RSF optimization or starts a new one.

    Parameters:
    - auto_resume: If True, automatically resume the latest study if found
    """
    latest_study = find_latest_study("rsf") if auto_resume else None

    if latest_study:
        print(f"🔄 Found existing study: {os.path.basename(latest_study)}")

        # Load study to check progress
        study_data = joblib.load(latest_study)

        # Handle both old and new formats
        if isinstance(study_data, dict) and "study" in study_data:
            # New format with CV splits saved
            study = study_data["study"]
            saved_n_splits = study_data.get("metadata", {}).get("n_splits", None)
        else:
            # Old format - just the study
            study = study_data
            saved_n_splits = None

        completed_trials = len(study.trials)

        # Check if CV configuration is compatible
        if saved_n_splits and saved_n_splits != n_splits:
            print(f"⚠️  Incompatible CV configuration:")
            print(f"   Saved study: {saved_n_splits} folds")
            print(f"   Requested: {n_splits} folds")
            print(f"🆕 Starting new optimization with {n_splits} folds")
            return optimize_random_survival_forest_hyperparameters_cv(
                X_train, y_train, n_trials=n_trials, n_splits=n_splits
            )

        if completed_trials >= n_trials:
            print(f"✅ Study already completed ({completed_trials}/{n_trials} trials)")
            return study.best_params, study.best_value, None

        remaining_trials = n_trials - completed_trials
        print(
            f"📊 Resuming optimization: {completed_trials} trials completed, {remaining_trials} remaining"
        )

        return optimize_random_survival_forest_hyperparameters_cv(
            X_train,
            y_train,
            n_trials=remaining_trials,
            n_splits=n_splits,
            resume_study=latest_study,
        )
    else:
        print("🆕 Starting new optimization")
        return optimize_random_survival_forest_hyperparameters_cv(
            X_train, y_train, n_trials=n_trials, n_splits=n_splits
        )


# TAU pour le calcul du C-index - FIXÉ SELON LES CRITÈRES DU CHALLENGE
TAU = 7  # Valeur fixe pour le challenge


def setup_logging(log_path):
    """Setup advanced logging with timestamps"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def create_stratified_survival_cv(y_train, n_splits=5, random_state=None):
    """
    Create stratified cross-validation splits based on event occurrence.
    Ensures balanced distribution of events vs censored observations across folds.
    """
    # Extract event indicator from structured survival array
    if hasattr(y_train, "dtype") and y_train.dtype.names:
        # Standard sksurv format with named fields
        event_field = None
        for field in y_train.dtype.names:
            if any(
                keyword in field.lower() for keyword in ["event", "status", "observed"]
            ):
                event_field = field
                break

        if event_field:
            events = y_train[event_field]
        else:
            # Use first field as event indicator
            events = y_train[y_train.dtype.names[0]]
    else:
        # Fallback: assume boolean array or first column
        events = y_train

    # Create stratified splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(np.arange(len(y_train)), events)


def compute_c_index_score(y_train, y_val, predictions, tau=TAU):
    """
    Compute C-index score for survival data.
    Simple and reliable metric for optimization.
    """
    try:
        c_index = concordance_index_ipcw(y_train, y_val, predictions, tau=tau)[0]
        return c_index
    except Exception as e:
        logging.warning(f"C-index computation failed: {e}. Returning 0.")
        return 0.0


def train_and_evaluate_fold(fold_data):
    """
    Train and evaluate model on a single fold.
    Designed for parallel execution with joblib.
    """
    fold_idx, train_idx, val_idx, X_train, y_train, model_class, params, tau = fold_data

    try:
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train model
        model = model_class(random_state=SEED, **params)
        model.fit(X_tr, y_tr)

        # Get predictions
        predictions = model.predict(X_val)

        # Compute C-index score
        c_index = compute_c_index_score(y_tr, y_val, predictions, tau)

        return {
            "fold_idx": fold_idx,
            "c_index": c_index,
            "success": True,
        }

    except Exception as e:
        logging.error(f"Error in fold {fold_idx}: {e}")
        return {
            "fold_idx": fold_idx,
            "c_index": 0.0,
            "success": False,
        }


def optimize_gradient_boosting_hyperparameters_cv(
    X_train, y_train, n_trials=50, n_splits=5, save_study=True, n_jobs=-1
):
    """
    Optimize Gradient Boosting hyperparameters with enhanced CV and composite metrics.
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(MODEL_DIR, f"gb_optimization_log_{timestamp}.log")
    csv_path = os.path.join(MODEL_DIR, f"gb_optimization_results_{timestamp}.csv")
    study_path = os.path.join(MODEL_DIR, f"gb_study_{timestamp}.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger = setup_logging(log_path)
    logger.info(f"Starting Gradient Boosting optimization with TAU = {TAU}")
    logger.info(f"Dataset shape: {X_train.shape}")

    # Setup CSV logging
    csv_header = [
        "trial",
        "fold",
        "model",
        "c_index_ipcw",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "min_samples_leaf",
        "min_samples_split",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    logger.info(f"CSV file created: {csv_path}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
        }

        logger.info(f"Trial {trial.number}: Testing parameters {params}")

        # Create stratified CV splits
        cv_splits = list(create_stratified_survival_cv(y_train, n_splits, SEED))

        # Prepare data for parallel processing
        fold_data = [
            (
                fold_idx,
                train_idx,
                val_idx,
                X_train,
                y_train,
                GradientBoostingSurvivalAnalysis,
                params,
                TAU,
            )
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits)
        ]

        # Parallel execution of folds
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(train_and_evaluate_fold)(data) for data in fold_data
        )

        # Aggregate results
        c_indices = []

        for result in fold_results:
            if result["success"]:
                c_indices.append(result["c_index"])

                # Log to CSV
                row_data = [
                    trial.number,
                    result["fold_idx"],
                    "GradientBoosting",
                    result["c_index"],
                ] + list(params.values())

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)

        if not c_indices:
            logger.error(f"Trial {trial.number}: All folds failed")
            return 0.0

        mean_c_index = np.mean(c_indices)

        logger.info(f"Trial {trial.number} completed: C-Index={mean_c_index:.4f}")

        return mean_c_index

    # Create study with enhanced configuration
    study = optuna.create_study(
        direction="maximize",
        study_name="GradientBoosting_enhanced_optimization",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,
            n_ei_candidates=24,
            multivariate=True,  # Better capture parameter correlations
        ),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=2, reduction_factor=2  # Minimum number of folds before pruning
        ),
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Save study for later analysis
    if save_study:
        joblib.dump(study, study_path)
        logger.info(f"Study saved to: {study_path}")

    logger.info(f"Optimization completed. Best score: {study.best_value:.5f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study.best_params, study.best_value, csv_path


def optimize_random_survival_forest_hyperparameters_cv(
    X_train,
    y_train,
    n_trials=50,
    n_splits=5,
    save_study=True,
    n_jobs=-1,
    resume_study=None,
):
    """
    Optimize Random Survival Forest hyperparameters with enhanced CV.

    Parameters:
    - resume_study: Path to an existing study file (.pkl) to resume optimization
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if we're resuming an existing study
    if resume_study and os.path.exists(resume_study):
        logger = setup_logging(os.path.join(MODEL_DIR, "temp_log.log"))
        logger.info(f"Resuming optimization from: {resume_study}")

        # Load existing study and its CV splits
        study_data = joblib.load(resume_study)
        if isinstance(study_data, dict) and "study" in study_data:
            # New format with CV splits saved
            study = study_data["study"]
            cv_splits = study_data["cv_splits"]
            logger.info("Loaded existing CV splits for consistency")
        else:
            # Old format - just the study
            study = study_data
            # Recreate splits with same seed for consistency
            cv_splits = list(create_stratified_survival_cv(y_train, n_splits, SEED))
            logger.warning(
                "Old study format - recreating CV splits (may be inconsistent)"
            )

        # Extract timestamp from existing study file to maintain file naming
        # Format: rsf_study_YYYYMMDD_HHMMSS.pkl -> YYYYMMDD_HHMMSS
        study_basename = os.path.basename(resume_study)
        if study_basename.startswith("rsf_study_") and study_basename.endswith(".pkl"):
            study_timestamp = study_basename[10:-4]  # Remove "rsf_study_" and ".pkl"
        else:
            # Fallback to old logic
            study_timestamp = study_basename.split("_")[-1].replace(".pkl", "")

        log_path = os.path.join(
            MODEL_DIR, f"rsf_optimization_log_{study_timestamp}.log"
        )
        csv_path = os.path.join(
            MODEL_DIR, f"rsf_optimization_results_{study_timestamp}.csv"
        )
        study_path = resume_study

        # Setup logging in append mode for existing study
        logger = setup_logging(log_path)
        logger.info(f"RESUMING Random Survival Forest optimization with TAU = {TAU}")
        logger.info(f"Previous trials completed: {len(study.trials)}")
        logger.info(f"Current best score: {study.best_value:.4f}")

    else:
        # Setup new optimization
        log_path = os.path.join(MODEL_DIR, f"rsf_optimization_log_{timestamp}.log")
        csv_path = os.path.join(MODEL_DIR, f"rsf_optimization_results_{timestamp}.csv")
        study_path = os.path.join(MODEL_DIR, f"rsf_study_{timestamp}.pkl")
        os.makedirs(MODEL_DIR, exist_ok=True)

        logger = setup_logging(log_path)
        logger.info(f"Starting Random Survival Forest optimization with TAU = {TAU}")

        # Create CV splits ONCE at the beginning with fixed seed
        cv_splits = list(create_stratified_survival_cv(y_train, n_splits, SEED))
        logger.info(f"Created {n_splits} stratified CV splits with SEED={SEED}")

        # Setup CSV logging for new study
        csv_header = [
            "trial",
            "fold",
            "model",
            "c_index_ipcw",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

        # Create new study
        study = optuna.create_study(
            direction="maximize",
            study_name="RandomSurvivalForest_enhanced_optimization",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=20,
                n_ei_candidates=24,
                multivariate=True,
            ),
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=2, reduction_factor=2
            ),
        )

    logger.info(f"Dataset shape: {X_train.shape}")
    logger.info(f"CV splits: {len(cv_splits)} folds")
    logger.info(f"CSV file: {csv_path}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]
            ),
        }

        logger.info(f"Trial {trial.number}: Testing parameters {params}")

        # Use pre-computed CV splits (consistent across all trials)
        # No need to recreate splits for each trial anymore

        # Prepare data for parallel processing
        fold_data = [
            (
                fold_idx,
                train_idx,
                val_idx,
                X_train,
                y_train,
                RandomSurvivalForest,
                params,
                TAU,
            )
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits)
        ]

        # Parallel execution of folds
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(train_and_evaluate_fold)(data) for data in fold_data
        )

        # Aggregate results
        c_indices = []

        for result in fold_results:
            if result["success"]:
                c_indices.append(result["c_index"])

                # Log to CSV
                row_data = [
                    trial.number,
                    result["fold_idx"],
                    "RandomSurvivalForest",
                    result["c_index"],
                ] + list(params.values())

                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)

        if not c_indices:
            logger.error(f"Trial {trial.number}: All folds failed")
            return 0.0

        mean_c_index = np.mean(c_indices)

        logger.info(f"Trial {trial.number} completed: C-Index={mean_c_index:.4f}")

        return mean_c_index

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Save study AND CV splits for later analysis and resume
    if save_study:
        # Save both study and CV splits in a dictionary for RSF optimization
        study_data = {
            "study": study,
            "cv_splits": cv_splits,
            "metadata": {
                "n_splits": n_splits,
                "seed": SEED,
                "model_type": "RandomSurvivalForest",
                "created_at": timestamp,
            },
        }
        joblib.dump(study_data, study_path)
        logger.info(f"Study and CV splits saved to: {study_path}")

    logger.info(f"Optimization completed. Best score: {study.best_value:.5f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study.best_params, study.best_value, csv_path
