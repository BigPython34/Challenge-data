# Entraînement des modèles
import joblib
import os
from datetime import datetime
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

from ..config import SEED, MODEL_DIR


def train_cox_model(X_train, y_train, alpha=1.0):
    """Entraîne un modèle Cox Proportional Hazards"""
    cox = CoxPHSurvivalAnalysis(alpha=alpha)
    cox.fit(X_train, y_train)
    return cox


def train_rsf_model(
    X_train,
    y_train,
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=5,
    max_depth=5,
):
    """Entraîne un modèle Random Survival Forest"""
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=SEED,
    )
    rsf.fit(X_train, y_train)
    return rsf


def train_gradient_boosting_model(
    X_train,
    y_train,
    n_estimators=850,
    learning_rate=0.03160736770883459,
    max_depth=3,
    subsample=0.8118022194771892,
    min_samples_leaf=4,
    min_samples_split=4,
):
    """Entraîne un modèle Gradient Boosting Survival Analysis"""
    xgb_surv = GradientBoostingSurvivalAnalysis(
        random_state=SEED,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
    )
    xgb_surv.fit(X_train, y_train)
    return xgb_surv


def save_model(model, model_name, params=None):
    """Sauvegarde un modèle avec un nom basé sur ses paramètres"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if params:
        param_str = "_".join([f"{k}{v}" for k, v in params.items()])
        filename = f"{model_name}_{param_str}_{timestamp}.pkl"
    else:
        filename = f"{model_name}_{timestamp}.pkl"

    filepath = os.path.join(MODEL_DIR, filename)
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé: {filepath}")
    return filepath


def load_model(filepath):
    """Charge un modèle sauvegardé"""
    return joblib.load(filepath)


def train_and_save_all_models(X_train, y_train):
    """Entraîne et sauvegarde tous les modèles"""
    models = {}

    # Cox
    print("Entraînement du modèle Cox...")
    cox = train_cox_model(X_train, y_train)
    cox_path = save_model(cox, "cox", {"alpha": 1.0})
    models["cox"] = {"model": cox, "path": cox_path}

    # RSF
    print("Entraînement du modèle Random Survival Forest...")
    rsf = train_rsf_model(X_train, y_train)
    rsf_params = {"n_est": 100, "max_depth": 5}
    rsf_path = save_model(rsf, "rsf", rsf_params)
    models["rsf"] = {"model": rsf, "path": rsf_path}

    # Gradient Boosting
    print("Entraînement du modèle Gradient Boosting...")
    xgb = train_gradient_boosting_model(X_train, y_train)
    xgb_params = {"n_est": 850, "lr": 0.032, "depth": 3}
    xgb_path = save_model(xgb, "gradient_boosting", xgb_params)
    models["gradient_boosting"] = {"model": xgb, "path": xgb_path}

    return models
