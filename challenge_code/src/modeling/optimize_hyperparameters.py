import os
import time
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
    Reusable orchestrator for survival model optimization using Optuna.
    Clean, safe for low-RAM environments (Azure VM friendly).
    """

    def __init__(
        self,
        model_name,
        model_class,
        search_space,
        X,
        y,
        n_splits=5,
        study_n_jobs=1,
        model_n_jobs=1,
        timeout_per_trial=120,
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.search_space = search_space
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.study_n_jobs = study_n_jobs
        self.model_n_jobs = model_n_jobs
        self.timeout_per_trial = timeout_per_trial  # anti-Killed
        self.SEED = 42

        self.MODEL_DIR = "models/studies"
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # Setup paths
        self.study_path = os.path.join(
            self.MODEL_DIR, f"{self.model_name}_study.joblib"
        )
        self.log_path = os.path.join(
            self.MODEL_DIR, f"{self.model_name}_optimization.log"
        )

        self.logger = self._setup_logging()
        self.cv_splits = self._create_stratified_cv_splits()

    def _setup_logging(self):
        logger = logging.getLogger(f"{self.model_name}_Optimizer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(self.log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger

    def _create_stratified_cv_splits(self):
        """Stratification only on event indicator (ok for survival)."""
        event_indicator = self.y[self.y.dtype.names[0]]
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.SEED,
        )
        return list(skf.split(np.arange(len(self.y)), event_indicator))

    def _objective(self, trial):
        start_time = time.time()

        # Build params safely
        params = {}
        for name, spec in self.search_space.items():
            ptype, args = spec
            if ptype == "int":
                params[name] = trial.suggest_int(name, *args)
            elif ptype == "float":
                params[name] = trial.suggest_float(name, *args)
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, args)

        # Anti-Killed: enforce n_jobs=1
        params_local = params.copy()
        if self.model_n_jobs is not None:
            params_local["n_jobs"] = self.model_n_jobs

        self.logger.info(f"Trial {trial.number} → {params_local}")

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splits):
            if time.time() - start_time > self.timeout_per_trial:
                self.logger.info(f"Trial {trial.number} KILLED by timeout")
                raise optuna.TrialPruned()

            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y[val_idx]

            try:
                model = self.model_class(
                    random_state=self.SEED,
                    **params_local
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = concordance_index_ipcw(y_train, y_val, preds)[0]
                scores.append(score)

            except Exception as e:
                self.logger.info(f"Fold {fold_idx+1} failed → {e}")
                scores.append(np.nan)

        mean_score = np.nanmean(scores)

        self.logger.info(f"Trial {trial.number} → mean IPCW: {mean_score:.5f}")
        return mean_score

    def run(self, n_trials=50):
        """Load existing study or create a new one."""
        if os.path.exists(self.study_path):
            study = joblib.load(self.study_path)
            self.logger.info(f"Resuming existing study with {len(study.trials)} trials.")
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.SEED,
                    multivariate=True,
                ),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
            )

        completed = len(study.trials)
        remaining = max(0, n_trials - completed)

        if remaining == 0:
            self.logger.info("Nothing left to run.")
        else:
            study.optimize(
                self._objective,
                n_trials=remaining,
                n_jobs=self.study_n_jobs,
            )

        joblib.dump(study, self.study_path)
        self.logger.info(f"Study saved → {self.study_path}")

        self.logger.info("BEST PARAMETERS:")
        self.logger.info(str(study.best_params))

        return study.best_params
