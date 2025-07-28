import numpy as np
import pandas as pd
import pickle
import os
import joblib
import csv
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
import optuna
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
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

# TAU pour le calcul du C-index
TAU = None  # Will be set automatically based on data


def set_tau_from_data(y_train):
    """Set TAU based on training data (max observed time)"""
    global TAU
    # Les données de survie créées avec Surv.from_dataframe ont des champs différents
    # Essayer différents noms de champs possibles
    try:
        # Essayer d'abord les noms standards de sksurv
        if hasattr(y_train, "dtype") and y_train.dtype.names:
            field_names = y_train.dtype.names
            print(f"Available fields in y_train: {field_names}")

            # Chercher le champ de temps
            time_field = None
            for field in field_names:
                if any(
                    keyword in field.lower()
                    for keyword in ["time", "duration", "years", "months", "days"]
                ):
                    time_field = field
                    break

            if time_field:
                max_time = np.max(y_train[time_field])
            else:
                # Utiliser le deuxième champ par défaut (généralement le temps)
                max_time = np.max(y_train[field_names[1]])
        else:
            # Si c'est un array simple, prendre le maximum
            max_time = np.max(y_train)

    except Exception as e:
        print(f"Error accessing survival time data: {e}")
        # Fallback: utiliser une valeur par défaut
        max_time = 10.0  # années
        print(f"Using default max_time: {max_time}")

    TAU = max_time * 0.95  # 95% du temps max observé
    print(f"TAU set to: {TAU:.2f}")
    return TAU


def optimize_gradient_boosting_hyperparameters_cv(
    X_train, y_train, n_trials=50, n_splits=5, save_study=True
):
    """Optimize Gradient Boosting hyperparameters with Optuna and K-Fold CV"""

    # Set TAU if not already set
    if TAU is None:
        set_tau_from_data(y_train)

    # Préparer le fichier CSV pour écriture au fur et à mesure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(MODEL_DIR, f"gb_optimization_results_{timestamp}.csv")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Créer le header du CSV
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

    # Écrire le header
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    print(f"CSV file created: {csv_path}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 1000, step=50
            ),  # Étendu jusqu'à 1000
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.5, log=True
            ),  # Plus large gamme avec log scale
            "max_depth": trial.suggest_int("max_depth", 1, 12),  # Étendu de 1 à 12
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),  # Plus large gamme
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),  # Étendu
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 200
            ),  # Étendu
        }

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = GradientBoostingSurvivalAnalysis(random_state=SEED, **params)
            model.fit(X_tr, y_tr)

            c_index = concordance_index_ipcw(
                y_tr, y_val, model.predict(X_val), tau=TAU
            )[0]
            cv_scores.append(c_index)

            # Écrire immédiatement chaque résultat dans le CSV
            row_data = [trial.number, fold_idx, "GradientBoosting", c_index] + list(
                params.values()
            )
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        print(
            f"Trial {trial.number}: Mean CV C-Index: {mean_cv_score:.5f} (±{std_cv_score:.5f}) - {n_splits} individual scores saved"
        )
        return mean_cv_score

    study = optuna.create_study(
        direction="maximize",
        study_name="GradientBoosting_optimization",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20, n_ei_candidates=24
        ),  # Optimisation avancée
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=5
        ),  # Élimination précoce
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"\n=== GRADIENT BOOSTING OPTIMIZATION RESULTS ===")
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.5f}")
    print(f"All individual results saved to CSV: {csv_path}")

    return study.best_params, study.best_value, csv_path


def optimize_random_survival_forest_hyperparameters_cv(
    X_train, y_train, n_trials=50, n_splits=5, save_study=True
):
    """Optimize Random Survival Forest hyperparameters with Optuna and K-Fold CV"""

    # Set TAU if not already set
    if TAU is None:
        set_tau_from_data(y_train)

    # Préparer le fichier CSV pour écriture au fur et à mesure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(MODEL_DIR, f"rsf_optimization_results_{timestamp}.csv")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Créer le header du CSV
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

    # Écrire le header
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    print(f"CSV file created: {csv_path}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 1000, step=50
            ),  # Étendu jusqu'à 1000
            "max_depth": trial.suggest_int("max_depth", 2, 20),  # Étendu jusqu'à 20
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 100
            ),  # Étendu
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),  # Étendu
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None, 0.3, 0.5, 0.7],  # Ajouté des valeurs numériques
            ),
        }

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = RandomSurvivalForest(random_state=SEED, **params)
            model.fit(X_tr, y_tr)

            c_index = concordance_index_ipcw(
                y_tr, y_val, model.predict(X_val), tau=TAU
            )[0]
            cv_scores.append(c_index)

            # Écrire immédiatement chaque résultat dans le CSV
            row_data = [trial.number, fold_idx, "RandomSurvivalForest", c_index] + list(
                params.values()
            )
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        print(
            f"Trial {trial.number}: Mean CV C-Index: {mean_cv_score:.5f} (±{std_cv_score:.5f}) - {n_splits} individual scores saved"
        )
        return mean_cv_score

    study = optuna.create_study(
        direction="maximize",
        study_name="RandomSurvivalForest_optimization",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20, n_ei_candidates=24
        ),  # Optimisation avancée
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=5
        ),  # Élimination précoce
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"\n=== RANDOM SURVIVAL FOREST OPTIMIZATION RESULTS ===")
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.5f}")
    print(f"All individual results saved to CSV: {csv_path}")

    return study.best_params, study.best_value, csv_path


def optimize_both_models(X_train, y_train, n_trials=50, n_splits=5):
    """Optimize both Gradient Boosting and Random Survival Forest models"""

    print("=== HYPERPARAMETER OPTIMIZATION FOR SURVIVAL MODELS ===")
    print(f"Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Cross-validation: {n_splits} folds")
    print(f"Trials per model: {n_trials}")
    print(f"🔥 INTENSIVE OPTIMIZATION MODE - ESTIMATED TIME: 5-6 HOURS 🔥")
    print(f"Total evaluations: {n_trials * 2 * n_splits} = {n_trials * 2 * n_splits}")
    print("💡 Tip: Results are saved continuously - safe to interrupt if needed")
    print("=" * 70)

    # Enregistrer l'heure de début
    start_time = datetime.now()
    print(f"⏰ Optimization started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Optimize Gradient Boosting
    print("\n🔧 OPTIMIZING GRADIENT BOOSTING SURVIVAL ANALYSIS...")
    gb_params, gb_score, gb_csv_path = optimize_gradient_boosting_hyperparameters_cv(
        X_train, y_train, n_trials=n_trials, n_splits=n_splits
    )
    results["gradient_boosting"] = {
        "params": gb_params,
        "score": gb_score,
        "csv_path": gb_csv_path,
    }

    # Optimize Random Survival Forest
    print("\n🌲 OPTIMIZING RANDOM SURVIVAL FOREST...")
    rsf_params, rsf_score, rsf_csv_path = (
        optimize_random_survival_forest_hyperparameters_cv(
            X_train, y_train, n_trials=n_trials, n_splits=n_splits
        )
    )
    results["random_survival_forest"] = {
        "params": rsf_params,
        "score": rsf_score,
        "csv_path": rsf_csv_path,
    }

    # Combiner tous les résultats des deux CSV dans un seul fichier
    print("\n📊 COMBINING ALL RESULTS INTO ONE CSV...")
    try:
        # Lire les deux CSV
        gb_df = pd.read_csv(gb_csv_path)
        rsf_df = pd.read_csv(rsf_csv_path)

        # Combiner
        combined_df = pd.concat([gb_df, rsf_df], ignore_index=True)

        # Sauvegarder le fichier combiné
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_csv_path = os.path.join(
            MODEL_DIR, f"optimization_all_results_{timestamp}.csv"
        )
        combined_df.to_csv(combined_csv_path, index=False)

        print(f"All combined results saved to: {combined_csv_path}")
        print(f"Total individual CV results: {len(combined_df)}")
        results["combined_csv_path"] = combined_csv_path

    except Exception as e:
        print(f"Warning: Could not combine CSV files: {e}")

    # Compare results
    elapsed_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("🏆 OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"⏰ Total optimization time: {elapsed_time}")
    print(f"Gradient Boosting - Best C-Index: {gb_score:.5f}")
    print(f"  Best params: {gb_params}")
    print(f"  CSV saved: {gb_csv_path}")
    print(f"\nRandom Survival Forest - Best C-Index: {rsf_score:.5f}")
    print(f"  Best params: {rsf_params}")
    print(f"  CSV saved: {rsf_csv_path}")

    # Determine best model
    if gb_score > rsf_score:
        print(f"\n🥇 Best model: Gradient Boosting (C-Index: {gb_score:.5f})")
        best_model_type = "gradient_boosting"
    else:
        print(f"\n🥇 Best model: Random Survival Forest (C-Index: {rsf_score:.5f})")
        best_model_type = "random_survival_forest"

    results["best_model"] = best_model_type

    # Sauvegarder un résumé des résultats (sans les modèles)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(MODEL_DIR, f"optimization_summary_{timestamp}.pkl")
    joblib.dump(results, summary_path)
    print(f"\nOptimization summary saved to: {summary_path}")

    return results
