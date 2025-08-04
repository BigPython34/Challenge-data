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
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
import warnings
import pickle
from src.modeling.evaluate import compare_models

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
    MODEL_DIR,
    COX_PARAMS,
    RSF_PARAMS,
    GRADIENT_BOOSTING_PARAMS,
    COXNET_PARAMS,
    EXTRA_TREES_PARAMS,
    COMPONENTWISE_GB_PARAMS,
    PYCOX_DEEPSURV_PARAMS,
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


def train_pycox_deepsurv_model(X_train, y_train, X_val=None, y_val=None, **params):
    """Train a PyCox DeepSurv model (simplified version)"""

    if not PYCOX_AVAILABLE:
        raise ImportError(
            "PyCox not available. Install with: pip install pycox torchtuples"
        )

    # Config
    config = PYCOX_DEEPSURV_PARAMS.copy()
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
    durations = np.array([y[1] for y in y_train], dtype=np.float32)
    events = np.array([y[0] for y in y_train], dtype=np.float32)

    scaler = StandardScaler()
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)

    if X_val is None or y_val is None:
        split = int(0.8 * len(X_train_scaled))
        X_val_scaled = X_train_scaled[split:]
        val_durations = durations[split:]
        val_events = events[split:]
        X_train_scaled = X_train_scaled[:split]
        durations = durations[:split]
        events = events[:split]
    else:
        X_val_np = X_val.values if hasattr(X_val, "values") else X_val
        X_val_scaled = scaler.transform(X_val_np).astype(np.float32)
        val_durations = np.array([y[1] for y in y_val], dtype=np.float32)
        val_events = np.array([y[0] for y in y_val], dtype=np.float32)

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
            cindex_train = concordance_index_ipcw(y_train, y_train, train_pred)[0]
            cindex_val = concordance_index_ipcw(y_train, y_val, val_pred)[0]
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


def train_cox_model(X_train, y_train, alpha=None):
    """Train a Cox Proportional Hazards model with config parameters"""
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
    """Train a Random Survival Forest model with config parameters"""
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
    """Train a Gradient Boosting Survival Analysis model with config parameters"""
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


def train_coxnet_model(X_train, y_train, l1_ratio=None, n_alphas=None, max_iter=None):
    """Train a Cox model with elastic net regularization (CoxNet)"""
    l1_ratio = l1_ratio if l1_ratio is not None else COXNET_PARAMS["l1_ratio"]
    n_alphas = n_alphas if n_alphas is not None else COXNET_PARAMS["n_alphas"]
    max_iter = max_iter if max_iter is not None else COXNET_PARAMS["max_iter"]

    coxnet = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        normalize=COXNET_PARAMS["normalize"],
        max_iter=max_iter,
    )
    coxnet.fit(X_train, y_train)
    return coxnet


def train_extra_trees_model(
    X_train,
    y_train,
    n_estimators=None,
    min_samples_split=None,
    min_samples_leaf=None,
    max_depth=None,
    max_features=None,
):
    """Train an Extra Survival Trees model"""
    n_estimators = (
        n_estimators if n_estimators is not None else EXTRA_TREES_PARAMS["n_estimators"]
    )
    min_samples_split = (
        min_samples_split
        if min_samples_split is not None
        else EXTRA_TREES_PARAMS["min_samples_split"]
    )
    min_samples_leaf = (
        min_samples_leaf
        if min_samples_leaf is not None
        else EXTRA_TREES_PARAMS["min_samples_leaf"]
    )
    max_depth = max_depth if max_depth is not None else EXTRA_TREES_PARAMS["max_depth"]
    max_features = (
        max_features if max_features is not None else EXTRA_TREES_PARAMS["max_features"]
    )

    extra_trees = ExtraSurvivalTrees(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        random_state=SEED,
    )
    extra_trees.fit(X_train, y_train)
    return extra_trees


def train_componentwise_gb_model(
    X_train, y_train, n_estimators=None, learning_rate=None, subsample=None
):
    """Train a Componentwise Gradient Boosting model"""
    n_estimators = (
        n_estimators
        if n_estimators is not None
        else COMPONENTWISE_GB_PARAMS["n_estimators"]
    )
    learning_rate = (
        learning_rate
        if learning_rate is not None
        else COMPONENTWISE_GB_PARAMS["learning_rate"]
    )
    subsample = (
        subsample if subsample is not None else COMPONENTWISE_GB_PARAMS["subsample"]
    )

    comp_gb = ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=SEED,
    )
    comp_gb.fit(X_train, y_train)
    return comp_gb


def save_model(model, model_name, params=None):
    """Save a model with a name based on its parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if params:
        param_str = "_".join([f"{k}{v}" for k, v in params.items()])
        filename = f"{model_name}_{param_str}_{timestamp}.pkl"
    else:
        filename = f"{model_name}_{timestamp}.pkl"

    filepath = os.path.join(MODEL_DIR, filename)
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")
    return filepath


def load_model(filepath):
    """Load a saved model"""
    return joblib.load(filepath)


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


def evaluate_and_save_best_model(models, X_train, y_train, X_test, y_test, features):
    """Evaluate models, select the best one, and save all relevant data."""

    print("Evaluating models...")
    results, best_model_name = compare_models(models, X_train, y_train, X_test, y_test)

    print("Evaluation completed")
    print(f"Best model: {best_model_name}")

    # Display model performances
    print("Model performances:")
    for model_name, metrics in results.items():
        if "test_accuracy" in metrics:
            print(f"  • {model_name}: {metrics['test_accuracy']:.3f}")

    # Save complete model package
    print("Saving model package...")
    os.makedirs("models", exist_ok=True)

    model_package = {
        "best_model": models[best_model_name],
        "best_model_name": best_model_name,
        "all_models": models,
        "evaluation_results": results,
        "features": features,
        "imputer": None,  # Placeholder for imputer if available
        "training_metadata": {
            "training_date": pd.Timestamp.now().isoformat(),
            "n_training_samples": len(X_train),
            "n_features": len(features),
            "best_model_accuracy": results[best_model_name].get("test_accuracy", "N/A"),
            "feature_names": features,
        },
    }

    package_path = "models/model_package.pkl"
    with open(package_path, "wb") as f:
        pickle.dump(model_package, f)

    print(f"Model package saved: {package_path}")

    # Save the best model alone
    best_model_path = "models/best_model.joblib"
    joblib.dump(models[best_model_name], best_model_path)
    print(f"Best model saved: {best_model_path}")

    # Save readable metadata
    metadata_path = "models/model_info.txt"
    with open(metadata_path, "w") as f:
        f.write("=== TRAINED MODEL INFORMATION ===\n")
        f.write(
            f"Training date: {model_package['training_metadata']['training_date']}\n"
        )
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"Accuracy: {results[best_model_name].get('test_accuracy', 'N/A')}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Number of features: {len(features)}\n")
        f.write("\nPerformance of all models:\n")
        for model_name, metrics in results.items():
            if "test_accuracy" in metrics:
                f.write(f"  • {model_name}: {metrics['test_accuracy']:.3f}\n")

    print(f"Metadata saved: {metadata_path}")
    print(f"Package size: {os.path.getsize(package_path) / 1024 / 1024:.2f} MB")

    return model_package


def generate_visualization_report(models, results, X_test, y_test):
    """Generate a visualization report for the models."""
    from src.visualization.plots import create_visualization_report

    print("Generating visualization report...")
    try:
        create_visualization_report(models, results, X_test, y_test)
        print("Visualization report generated")
    except Exception as e:
        print(f"Warning: Visualization report generation failed: {e}")


def train_and_save_all_models(X_train, y_train, X_val=None, y_val=None):
    """Train and save all scikit-survival models"""
    models = {}

    # Cox Proportional Hazards
    print("Training Cox model...")
    cox = train_cox_model(X_train, y_train)
    cox_path = save_model(cox, "cox", COX_PARAMS)
    models["cox"] = {"model": cox, "path": cox_path, "params": COX_PARAMS}

    # Random Survival Forest
    print("Training Random Survival Forest model...")
    rsf = train_rsf_model(X_train, y_train)
    rsf_path = save_model(rsf, "rsf", RSF_PARAMS)
    models["rsf"] = {"model": rsf, "path": rsf_path, "params": RSF_PARAMS}

    # Gradient Boosting Survival Analysis
    print("Training Gradient Boosting model...")
    xgb = train_gradient_boosting_model(X_train, y_train)
    xgb_path = save_model(xgb, "gradient_boosting", GRADIENT_BOOSTING_PARAMS)
    models["gradient_boosting"] = {
        "model": xgb,
        "path": xgb_path,
        "params": GRADIENT_BOOSTING_PARAMS,
    }

    # CoxNet (Cox avec régularisation)
    print("Training CoxNet model...")
    try:
        coxnet = train_coxnet_model(X_train, y_train)
        coxnet_path = save_model(coxnet, "coxnet", COXNET_PARAMS)
        models["coxnet"] = {
            "model": coxnet,
            "path": coxnet_path,
            "params": COXNET_PARAMS,
        }
    except Exception as e:
        print(f"  Error CoxNet: {e}")

    # Extra Survival Trees
    print("Training Extra Survival Trees model...")
    try:
        extra_trees = train_extra_trees_model(X_train, y_train)
        extra_trees_path = save_model(extra_trees, "extra_trees", EXTRA_TREES_PARAMS)
        models["extra_trees"] = {
            "model": extra_trees,
            "path": extra_trees_path,
            "params": EXTRA_TREES_PARAMS,
        }
    except Exception as e:
        print(f"   Error Extra Trees: {e}")

    # Componentwise Gradient Boosting
    print("Training Componentwise Gradient Boosting model...")
    try:
        comp_gb = train_componentwise_gb_model(X_train, y_train)
        comp_gb_path = save_model(comp_gb, "componentwise_gb", COMPONENTWISE_GB_PARAMS)
        models["componentwise_gb"] = {
            "model": comp_gb,
            "path": comp_gb_path,
            "params": COMPONENTWISE_GB_PARAMS,
        }
    except Exception as e:
        print(f"  Error Componentwise GB: {e}")

    # PyCox DeepSurv
    if PYCOX_AVAILABLE:
        print("Training PyCox DeepSurv model...")
        try:
            pycox_deepsurv = train_pycox_deepsurv_model(X_train, y_train, X_val, y_val)
            pycox_path = save_model(
                pycox_deepsurv, "pycox_deepsurv", PYCOX_DEEPSURV_PARAMS
            )
            models["pycox_deepsurv"] = {
                "model": pycox_deepsurv,
                "path": pycox_path,
                "params": PYCOX_DEEPSURV_PARAMS,
            }
        except Exception as e:
            print(f"   Error PyCox DeepSurv: {e}")
    else:
        print("   PyCox not available, DeepSurv ignored")

    return models
