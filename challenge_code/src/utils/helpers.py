# Fonctions utilitaires (plots, seed, etc.)
import numpy as np
import random
import optuna
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw

from ..config import SEED, TAU, IMPORTANT_GENES


def set_seed(seed=None):
    """Définit le seed pour la reproductibilité"""
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed défini à: {seed}")


def optimize_gradient_boosting_hyperparameters(
    X_train, y_train, X_test, y_test, n_trials=20
):
    """Optimise les hyperparamètres du Gradient Boosting avec Optuna"""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

        # Entraînement du modèle
        model = GradientBoostingSurvivalAnalysis(random_state=SEED, **params)
        model.fit(X_train, y_train)

        # Évaluation
        xgb_cindex_train = concordance_index_ipcw(
            y_train, y_train, model.predict(X_train), tau=TAU
        )[0]
        xgb_cindex_test = concordance_index_ipcw(
            y_train, y_test, model.predict(X_test), tau=TAU
        )[0]

        print(
            f"Trial {trial.number}: Train C-Index: {xgb_cindex_train:.5f}, Test C-Index: {xgb_cindex_test:.5f}"
        )

        return xgb_cindex_test

    # Optimisation avec Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Meilleurs hyperparamètres: {study.best_params}")
    print(f"Meilleur score: {study.best_value:.5f}")

    return study.best_params, study.best_value


def print_dataset_info(data_dict):
    """Affiche des informations sur les datasets chargés"""
    print("=== INFORMATIONS SUR LES DATASETS ===")
    for name, df in data_dict.items():
        print(f"{name}: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    print(f"\nGènes importants: {len(IMPORTANT_GENES)} gènes")
    print("=" * 50)


def create_submission_summary(submissions):
    """Crée un résumé des soumissions générées"""
    print("\n=== RÉSUMÉ DES SOUMISSIONS ===")
    for name, info in submissions.items():
        filepath = info["filepath"]
        submission_df = info["submission_df"]

        print(f"\nModèle: {name}")
        print(f"Fichier: {filepath}")
        print(f"Nombre de prédictions: {len(submission_df)}")
        print(f"Score min: {submission_df['risk_score'].min():.5f}")
        print(f"Score max: {submission_df['risk_score'].max():.5f}")
        print(f"Score moyen: {submission_df['risk_score'].mean():.5f}")

    print("=" * 50)
