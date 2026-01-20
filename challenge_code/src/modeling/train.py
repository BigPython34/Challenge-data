# Model training
import joblib
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    ComponentwiseGradientBoostingSurvivalAnalysis,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.meta import Stacking
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
import warnings
import pickle


# PyCox imports
try:
    from pycox.models import CoxPH as PyCoxCoxPH
    from pycox.preprocessing.label_transforms import LabTransCoxTime
    import torchtuples as tt

    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False

from ..config import (
    SEED,
    MODELING,
)

warnings.filterwarnings("ignore", category=UserWarning)


class PyCoxWrapper:
    """Wrapper for PyCox compatibility with scikit-survival"""

    def __init__(self, model, scaler, labtrans):
        self.model = model
        self.scaler = scaler
        self.labtrans = labtrans

    def predict(self, X):
        """Prediction compatible with scikit-survival"""
        X_np = X.values if hasattr(X, "values") else X
        X_scaled = self.scaler.transform(X_np).astype(np.float32)

        # Prediction of log hazards
        log_h = self.model.predict(X_scaled)
        return log_h.flatten()

    def predict_surv_df(self, X):
        """Prediction of survival curves (PyCox method)"""
        X_np = X.values if hasattr(X, "values") else X
        X_scaled = self.scaler.transform(X_np).astype(np.float32)
        return self.model.predict_surv_df(X_scaled)


class DeepSurvEstimator:
    """Sklearn-like estimator that wraps the PyCox DeepSurv training routine."""

    def __init__(self, **params):
        if not PYCOX_AVAILABLE:
            raise ImportError(
                "PyCox is not installed. Install it with: pip install pycox torchtuples"
            )
        self.params = params.copy()
        self.model_ = None

    def fit(self, X, y):
        self.model_ = train_pycox_deepsurv_model(X, y, **self.params)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("DeepSurvEstimator must be fitted before predicting.")
        return self.model_.predict(X)

    def predict_surv_df(self, X):
        if self.model_ is None:
            raise RuntimeError("DeepSurvEstimator must be fitted before predicting.")
        return self.model_.predict_surv_df(X)

    def get_params(self, deep=False):
        return self.params.copy()


MODEL_CLASS_MAP = {
    "Cox": CoxPHSurvivalAnalysis,
    "RSF": RandomSurvivalForest,
    "GradientBoosting": GradientBoostingSurvivalAnalysis,
    "CoxNet": CoxnetSurvivalAnalysis,
    "ExtraTrees": ExtraSurvivalTrees,
    "ComponentwiseGB": ComponentwiseGradientBoostingSurvivalAnalysis,
}

if PYCOX_AVAILABLE:
    MODEL_CLASS_MAP["DeepSurv"] = DeepSurvEstimator


RANDOM_STATE_MODELS = {"RSF", "GradientBoosting", "ExtraTrees", "ComponentwiseGB"}


def _build_model_from_config(name):
    """Instantiate a survival model described in MODELING config."""

    model_cfg = MODELING["models"].get(name)
    if model_cfg is None:
        raise ValueError(f"Model '{name}' is not defined in MODELING['models'].")

    if name == "DeepSurv" and not PYCOX_AVAILABLE:
        raise ValueError(
            "DeepSurv requested but PyCox/Torchtuples dependencies are missing."
        )

    params = model_cfg.get("params", {}).copy()
    if name in RANDOM_STATE_MODELS:
        params.setdefault("random_state", SEED)

    model_cls = MODEL_CLASS_MAP.get(name)
    if model_cls is None:
        raise ValueError(f"No registered class for model '{name}'.")

    return model_cls(**params)


def _build_stacking_model():
    """Construct a sksurv.meta.Stacking estimator from config settings."""

    stacking_cfg = MODELING.get("stacking", {})
    if not stacking_cfg.get("enabled"):
        return None, None

    base_names = stacking_cfg.get("base_models", [])
    meta_name = stacking_cfg.get("meta_model")
    if not base_names or not meta_name:
        raise ValueError("Stacking requires both 'base_models' and 'meta_model'.")

    base_estimators = []
    for base_name in base_names:
        try:
            base_estimators.append((base_name, _build_model_from_config(base_name)))
        except ValueError as err:
            print(f"[WARN] Stacking base model skipped: {err}")

    if not base_estimators:
        raise ValueError("No valid base models remain for stacking after validation.")

    meta_estimator = _build_model_from_config(meta_name)
    probabilities = stacking_cfg.get("probabilities", True)
    stacking_name = stacking_cfg.get("name", "Stacking")

    stacking_model = Stacking(
        meta_estimator=meta_estimator,
        base_estimators=base_estimators,
        probabilities=probabilities,
    )

    return stacking_name, stacking_model


def train_pycox_deepsurv_model(X_train, y_train, X_val=None, y_val=None, **params):
    """Train a PyCox DeepSurv model (simplified version)"""

    if not PYCOX_AVAILABLE:
        raise ImportError(
            "PyCox not available. Install with: pip install pycox torchtuples"
        )

    # Config
    config = MODELING['models']['DeepSurv']['params'].copy()
    config.update(params)

    hidden_layers = config["hidden_layers"]
    dropout = config["dropout"]
    batch_norm = config["batch_norm"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    patience = config["patience"]
    weight_decay = config["weight_decay"]
    lr_scheduler = config.get("lr_scheduler", False)
    lr_factor = config.get("lr_factor", 0.5)
    lr_patience = config.get("lr_patience", 25)

    # Preprocessing
    durations_full = np.array([y[1] for y in y_train], dtype=np.float32)
    events_full = np.array([y[0] for y in y_train], dtype=np.float32)

    scaler = StandardScaler()
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)

    if X_val is None or y_val is None:
        split = int(0.8 * len(X_train_scaled))
        X_val_scaled = X_train_scaled[split:]
        val_durations = durations_full[split:]
        val_events = events_full[split:]
        y_val_struct = y_train[split:]

        X_train_scaled = X_train_scaled[:split]
        durations = durations_full[:split]
        events = events_full[:split]
        y_train_struct = y_train[:split]
    else:
        X_val_np = X_val.values if hasattr(X_val, "values") else X_val
        X_val_scaled = scaler.transform(X_val_np).astype(np.float32)
        val_durations = np.array([y[1] for y in y_val], dtype=np.float32)
        val_events = np.array([y[0] for y in y_val], dtype=np.float32)
        y_val_struct = y_val
        durations = durations_full
        events = events_full
        y_train_struct = y_train

    # Label transformation
    labtrans = LabTransCoxTime()
    y_train_proc = labtrans.fit_transform(durations, events)
    y_val_proc = labtrans.transform(val_durations, val_events)

    # Network
    net = tt.practical.MLPVanilla(
        in_features=X_train_scaled.shape[1],
        num_nodes=hidden_layers,
        out_features=1,
        batch_norm=batch_norm,
        dropout=dropout,
    )

    model = PyCoxCoxPH(net, torch.optim.Adam)
    model.labtrans = labtrans
    model.optimizer.set_lr(learning_rate)
    model.optimizer.param_groups[0]["weight_decay"] = weight_decay

    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer.optimizer,
            mode="max",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=learning_rate * 0.01,
        )
    else:
        scheduler = None

    # Training loop (simplified)
    best_val = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.fit(
            X_train_scaled,
            y_train_proc,
            batch_size,
            1,
            val_data=(X_val_scaled, y_val_proc),
            verbose=False,
        )

        train_pred = model.predict(X_train_scaled).flatten()
        val_pred = model.predict(X_val_scaled).flatten()

        try:
            cindex_train = concordance_index_ipcw(y_train_struct, y_train_struct, train_pred)[0]
            cindex_val = concordance_index_ipcw(y_train_struct, y_val_struct, val_pred)[0]
        except:
            cindex_train = np.nan
            cindex_val = np.nan

        if scheduler and not np.isnan(cindex_val):
            scheduler.step(cindex_val)

        if not np.isnan(cindex_val) and cindex_val > best_val:
            best_val = cindex_val
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}/{epochs} | C-Index Train: {cindex_train:.4f} | Val: {cindex_val:.4f}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return PyCoxWrapper(model, scaler, labtrans)


def train_cox_model(X_train, y_train, **params):
    """Train a Cox Proportional Hazards model with config parameters"""
    config = MODELING['models']['Cox']['params'].copy()
    config.update(params)
    cox = CoxPHSurvivalAnalysis(**config)
    cox.fit(X_train, y_train)
    return cox


def train_rsf_model(X_train, y_train, **params):
    """Train a Random Survival Forest model with config parameters"""
    config = MODELING['models']['RSF']['params'].copy()
    config.update(params)
    rsf = RandomSurvivalForest(random_state=SEED, **config)
    rsf.fit(X_train, y_train)
    return rsf


def train_gradient_boosting_model(X_train, y_train, **params):
    """Train a Gradient Boosting Survival Analysis model with config parameters"""
    config = MODELING['models']['GradientBoosting']['params'].copy()
    config.update(params)
    xgb_surv = GradientBoostingSurvivalAnalysis(random_state=SEED, **config)
    xgb_surv.fit(X_train, y_train)
    return xgb_surv


def train_coxnet_model(X_train, y_train, **params):
    """Train a Cox model with elastic net regularization (CoxNet)"""
    config = MODELING['models']['CoxNet']['params'].copy()
    config.update(params)
    coxnet = CoxnetSurvivalAnalysis(**config)
    coxnet.fit(X_train, y_train)
    return coxnet


def train_extra_trees_model(X_train, y_train, **params):
    """Train an Extra Survival Trees model"""
    config = MODELING['models']['ExtraTrees']['params'].copy()
    config.update(params)
    extra_trees = ExtraSurvivalTrees(random_state=SEED, **config)
    extra_trees.fit(X_train, y_train)
    return extra_trees


def train_componentwise_gb_model(X_train, y_train, **params):
    """Train a Componentwise Gradient Boosting model"""
    config = MODELING['models']['ComponentwiseGB']['params'].copy()
    config.update(params)
    comp_gb = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=SEED, **config)
    comp_gb.fit(X_train, y_train)
    return comp_gb


def load_training_dataset(dataset_path):
    """Load the prepared training dataset."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"ERROR: Prepared dataset not found! Expected file: {dataset_path}. Please run: python 1_prepare_data.py first."
        )

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    print("Dataset successfully loaded")
    print(f"{dataset['metadata']['n_samples_train']} training samples")
    print(f"{dataset['metadata']['n_features']} features")

    return dataset


def load_training_dataset_csv(X_train_path, y_train_path):
    """Load the training dataset from CSV files."""
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError(
            f"ERROR: Training dataset files not found! Expected files: {X_train_path}, {y_train_path}."
        )

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Exclude patient IDs from X_train
    if "ID" in X_train.columns:
        X_train = X_train.drop(columns=["ID"])
    if "ID" in y_train.columns:
        y_train = y_train.drop(columns=["ID"])

    # Convert y_train to structured array
    if "OS_STATUS" in y_train.columns and "OS_YEARS" in y_train.columns:
        y_train = Surv.from_arrays(
            event=y_train["OS_STATUS"].astype(bool),
            time=y_train["OS_YEARS"].astype(float),
        )
    else:
        raise ValueError(
            "ERROR: y_train must contain 'OS_STATUS' and 'OS_YEARS' columns."
        )

    print("Training dataset successfully loaded")
    print(f"{X_train.shape[0]} training samples")
    print(f"{X_train.shape[1]} features")

    return X_train, y_train


def get_survival_models():
    """Return survival estimators configured in `config.py`, including stacking."""

    models = {}
    for name, config in MODELING["models"].items():
        if not config.get("enabled", False):
            continue

        try:
            models[name] = _build_model_from_config(name)
        except ValueError as err:
            print(f"[WARN] {err}")

    stacking_cfg = MODELING.get("stacking", {})
    if stacking_cfg.get("enabled"):
        try:
            stacking_name, stacking_model = _build_stacking_model()
            if stacking_model is not None:
                models[stacking_name] = stacking_model
        except ValueError as err:
            print(f"[WARN] Unable to configure stacking model: {err}")

    return models
