import numpy as np
from sklearn.model_selection import KFold
import optuna
from sksurv.metrics import concordance_index_ipcw
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from ..config import SEED, TAU


def optimize_gradient_boosting_hyperparameters_cv(
    X_train, y_train, n_trials=20, n_splits=5
):
    """Optimize Gradient Boosting hyperparameters with Optuna and K-Fold CV"""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        cv_scores = []

        for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = GradientBoostingSurvivalAnalysis(random_state=SEED, **params)
            model.fit(X_tr, y_tr)

            c_index = concordance_index_ipcw(
                y_tr, y_val, model.predict(X_val), tau=TAU
            )[0]
            cv_scores.append(c_index)

        mean_cv_score = np.mean(cv_scores)
        print(f"Trial {trial.number}: Mean CV C-Index: {mean_cv_score:.5f}")
        return mean_cv_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.5f}")

    return study.best_params, study.best_value
