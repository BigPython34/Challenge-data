# Entraînement des modèles
import joblib
import os
from datetime import datetime
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

from ..config import SEED, MODEL_DIR, COX_PARAMS, RSF_PARAMS, GRADIENT_BOOSTING_PARAMS


def train_cox_model(X_train, y_train, alpha=None):
    """Entraîne un modèle Cox Proportional Hazards avec paramètres du config"""
    if alpha is None:
        alpha = COX_PARAMS["alpha"]
    cox = CoxPHSurvivalAnalysis(alpha=alpha)
    cox.fit(X_train, y_train)
    return cox


def train_rsf_model(
    X_train,
    y_train,
    n_estimators=None,
    min_samples_split=None,
    min_samples_leaf=None,
    max_depth=None,
):
    """Entraîne un modèle Random Survival Forest avec paramètres du config"""
    n_estimators = (
        n_estimators if n_estimators is not None else RSF_PARAMS["n_estimators"]
    )
    min_samples_split = (
        min_samples_split
        if min_samples_split is not None
        else RSF_PARAMS["min_samples_split"]
    )
    min_samples_leaf = (
        min_samples_leaf
        if min_samples_leaf is not None
        else RSF_PARAMS["min_samples_leaf"]
    )
    max_depth = max_depth if max_depth is not None else RSF_PARAMS["max_depth"]
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
    n_estimators=None,
    learning_rate=None,
    max_depth=None,
    subsample=None,
    min_samples_leaf=None,
    min_samples_split=None,
):
    """Entraîne un modèle Gradient Boosting Survival Analysis avec paramètres du config"""
    n_estimators = (
        n_estimators
        if n_estimators is not None
        else GRADIENT_BOOSTING_PARAMS["n_estimators"]
    )
    learning_rate = (
        learning_rate
        if learning_rate is not None
        else GRADIENT_BOOSTING_PARAMS["learning_rate"]
    )
    max_depth = (
        max_depth if max_depth is not None else GRADIENT_BOOSTING_PARAMS["max_depth"]
    )
    subsample = (
        subsample if subsample is not None else GRADIENT_BOOSTING_PARAMS["subsample"]
    )
    min_samples_leaf = (
        min_samples_leaf
        if min_samples_leaf is not None
        else GRADIENT_BOOSTING_PARAMS["min_samples_leaf"]
    )
    min_samples_split = (
        min_samples_split
        if min_samples_split is not None
        else GRADIENT_BOOSTING_PARAMS["min_samples_split"]
    )
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
    cox_path = save_model(cox, "cox", COX_PARAMS)
    models["cox"] = {"model": cox, "path": cox_path, "params": COX_PARAMS}

    # RSF
    print("Entraînement du modèle Random Survival Forest...")
    rsf = train_rsf_model(X_train, y_train)
    rsf_path = save_model(rsf, "rsf", RSF_PARAMS)
    models["rsf"] = {"model": rsf, "path": rsf_path, "params": RSF_PARAMS}

    # Gradient Boosting
    print("Entraînement du modèle Gradient Boosting...")
    xgb = train_gradient_boosting_model(X_train, y_train)
    xgb_path = save_model(xgb, "gradient_boosting", GRADIENT_BOOSTING_PARAMS)
    models["gradient_boosting"] = {
        "model": xgb,
        "path": xgb_path,
        "params": GRADIENT_BOOSTING_PARAMS,
    }

    return models
