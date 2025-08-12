import os
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_ipcw
from datetime import datetime
import logging


class OptimizationOrchestrator:
    """
    A reusable class to handle the optimization of any scikit-survival model.
    Manages study resuming, logging, parallel cross-validation, and results saving.
    """

    def __init__(
        self, model_name, model_class, search_space, X, y, n_splits=5, n_jobs=-1
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.search_space = search_space
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.SEED = 42
        self.MODEL_DIR = "models/studies"
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # Setup paths
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_path = os.path.join(
            self.MODEL_DIR, f"{self.model_name}_study.joblib"
        )
        self.log_path = os.path.join(
            self.MODEL_DIR, f"{self.model_name}_optimization.log"
        )
        self.csv_path = os.path.join(self.MODEL_DIR, f"{self.model_name}_results.csv")

        self.logger = self._setup_logging()
        self.cv_splits = self._create_stratified_cv_splits()

    def _setup_logging(self):
        """Sets up a logger for the optimization process."""
        logger = logging.getLogger(f"{self.model_name}_Optimizer")
        logger.setLevel(logging.INFO)
        # Avoid adding handlers if they already exist (e.g., in interactive environments)
        if not logger.handlers:
            file_handler = logging.FileHandler(self.log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        return logger

    def _create_stratified_cv_splits(self):
        """Creates stratified cross-validation splits based on event status."""
        event_indicator = self.y[self.y.dtype.names[0]]
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.SEED
        )
        return list(skf.split(np.arange(len(self.y)), event_indicator))

    def _objective(self, trial):
        """The objective function that Optuna will minimize/maximize."""
        params = {}
        for name, suggestion in self.search_space.items():
            param_type, args = suggestion
            if param_type == "int":
                params[name] = trial.suggest_int(name, *args)
            elif param_type == "float":
                params[name] = trial.suggest_float(name, *args)
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, args)

        self.logger.info(f"Trial {trial.number}: Testing params {params}")

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splits):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            try:
                model = self.model_class(random_state=self.SEED, **params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                score = concordance_index_ipcw(y_train, y_val, predictions)[0]
                scores.append(score)
            except Exception as e:
                self.logger.error(
                    f"  Fold {fold_idx+1} failed for trial {trial.number}: {e}"
                )
                scores.append(np.nan)

        mean_score = np.nanmean(scores)
        self.logger.info(f"  -> Trial {trial.number} Mean IPCW: {mean_score:.5f}")
        return mean_score

    def run(self, n_trials=100):
        """Starts or resumes the optimization study."""
        self.logger.info(
            f"--- Starting/Resuming Optimization for {self.model_name} ---"
        )

        # Try to load an existing study to resume
        try:
            study = joblib.load(self.study_path)
            self.logger.info(f"Resuming existing study from: {self.study_path}")
            self.logger.info(f"Completed trials: {len(study.trials)}")
        except FileNotFoundError:
            self.logger.info("No existing study found. Creating a new one.")
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{self.model_name} Optimization",
                sampler=optuna.samplers.TPESampler(seed=self.SEED, multivariate=True),
            )

        # Calculate remaining trials
        completed_trials = len(study.trials)
        if completed_trials >= n_trials:
            self.logger.info(
                f"Study has already completed {completed_trials} trials. Nothing to do."
            )
        else:
            remaining_trials = n_trials - completed_trials
            self.logger.info(
                f"Running for {remaining_trials} more trials (target: {n_trials} total)."
            )
            study.optimize(
                self._objective, n_trials=remaining_trials, n_jobs=self.n_jobs
            )

        # Save the study progress
        joblib.dump(study, self.study_path)
        self.logger.info(f"Study progress saved to: {self.study_path}")

        # Print final results
        self.logger.info("\n--- Optimization Finished ---")
        self.logger.info(f"Best score (Mean IPCW C-index): {study.best_value:.5f}")
        self.logger.info(f"Best hyperparameters: {study.best_params}")

        return study.best_params
