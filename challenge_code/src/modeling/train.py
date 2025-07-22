# Entraînement des modèles
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
import warnings

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
    """Wrapper pour la compatibilité PyCox avec scikit-survival"""

    def __init__(self, model, scaler, labtrans):
        self.model = model
        self.scaler = scaler
        self.labtrans = labtrans

    def predict(self, X):
        """Prédiction compatible avec scikit-survival"""
        X_np = X.values if hasattr(X, "values") else X
        X_scaled = self.scaler.transform(X_np).astype(np.float32)

        # Prédiction des log hazards
        log_h = self.model.predict(X_scaled)
        return log_h.flatten()

    def predict_surv_df(self, X):
        """Prédiction des courbes de survie (méthode PyCox)"""
        X_np = X.values if hasattr(X, "values") else X
        X_scaled = self.scaler.transform(X_np).astype(np.float32)
        return self.model.predict_surv_df(X_scaled)


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


def train_coxnet_model(X_train, y_train, l1_ratio=None, n_alphas=None, max_iter=None):
    """Entraîne un modèle Cox avec régularisation élastique (CoxNet)"""
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
    """Entraîne un modèle Extra Survival Trees"""
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
    """Entraîne un modèle Componentwise Gradient Boosting"""
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


def train_and_save_all_models(X_train, y_train, X_val=None, y_val=None):
    """Entraîne et sauvegarde tous les modèles de scikit-survival"""
    models = {}

    # Cox Proportional Hazards
    print("Entraînement du modèle Cox...")
    cox = train_cox_model(X_train, y_train)
    cox_path = save_model(cox, "cox", COX_PARAMS)
    models["cox"] = {"model": cox, "path": cox_path, "params": COX_PARAMS}

    # Random Survival Forest
    print("Entraînement du modèle Random Survival Forest...")
    rsf = train_rsf_model(X_train, y_train)
    rsf_path = save_model(rsf, "rsf", RSF_PARAMS)
    models["rsf"] = {"model": rsf, "path": rsf_path, "params": RSF_PARAMS}

    # Gradient Boosting Survival Analysis
    print("Entraînement du modèle Gradient Boosting...")
    xgb = train_gradient_boosting_model(X_train, y_train)
    xgb_path = save_model(xgb, "gradient_boosting", GRADIENT_BOOSTING_PARAMS)
    models["gradient_boosting"] = {
        "model": xgb,
        "path": xgb_path,
        "params": GRADIENT_BOOSTING_PARAMS,
    }

    # CoxNet (Cox avec régularisation)
    print("Entraînement du modèle CoxNet...")
    try:
        coxnet = train_coxnet_model(X_train, y_train)
        coxnet_path = save_model(coxnet, "coxnet", COXNET_PARAMS)
        models["coxnet"] = {
            "model": coxnet,
            "path": coxnet_path,
            "params": COXNET_PARAMS,
        }
    except Exception as e:
        print(f"   ⚠️  Erreur CoxNet: {e}")

    # Extra Survival Trees
    print("Entraînement du modèle Extra Survival Trees...")
    try:
        extra_trees = train_extra_trees_model(X_train, y_train)
        extra_trees_path = save_model(extra_trees, "extra_trees", EXTRA_TREES_PARAMS)
        models["extra_trees"] = {
            "model": extra_trees,
            "path": extra_trees_path,
            "params": EXTRA_TREES_PARAMS,
        }
    except Exception as e:
        print(f"   ⚠️  Erreur Extra Trees: {e}")

    # Componentwise Gradient Boosting
    print("Entraînement du modèle Componentwise Gradient Boosting...")
    try:
        comp_gb = train_componentwise_gb_model(X_train, y_train)
        comp_gb_path = save_model(comp_gb, "componentwise_gb", COMPONENTWISE_GB_PARAMS)
        models["componentwise_gb"] = {
            "model": comp_gb,
            "path": comp_gb_path,
            "params": COMPONENTWISE_GB_PARAMS,
        }
    except Exception as e:
        print(f"   ⚠️  Erreur Componentwise GB: {e}")

    # PyCox DeepSurv
    if PYCOX_AVAILABLE:
        print("Entraînement du modèle PyCox DeepSurv...")
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
            print(f"   ⚠️  Erreur PyCox DeepSurv: {e}")
    else:
        print("   ⚠️  PyCox non disponible, DeepSurv ignoré")

    return models


def train_pycox_deepsurv_model(X_train, y_train, X_val=None, y_val=None, **params):
    """Entraîne un modèle PyCox DeepSurv (Cox Proportional Hazards avec réseau de neurones)"""
    if not PYCOX_AVAILABLE:
        raise ImportError(
            "PyCox non disponible. Installez avec: pip install pycox torchtuples"
        )

    # Utilisation correcte de la configuration avec override possible
    config = PYCOX_DEEPSURV_PARAMS.copy()
    config.update(params)  # Override avec les paramètres passés

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

    print(f"   🧠 Entraînement PyCox DeepSurv:")
    print(f"      📐 Architecture: {hidden_layers}")
    print(f"      🎯 Learning Rate: {learning_rate}, Batch Size: {batch_size}")
    print(f"      🔄 Époques: {epochs}, Patience: {patience}")
    print(f"      🛡️  Dropout: {dropout}, Weight Decay: {weight_decay}")
    if lr_scheduler:
        print(f"      📉 LR Scheduler: factor={lr_factor}, patience={lr_patience}")

    # Préparation des données
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train

    # Conversion des données de survie sksurv vers PyCox
    durations = np.array([y[1] for y in y_train], dtype=np.float32)
    events = np.array([y[0] for y in y_train], dtype=np.float32)

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)

    # Préparation validation si disponible
    if X_val is not None and y_val is not None:
        X_val_np = X_val.values if hasattr(X_val, "values") else X_val
        X_val_scaled = scaler.transform(X_val_np).astype(np.float32)
        val_durations = np.array([y[1] for y in y_val], dtype=np.float32)
        val_events = np.array([y[0] for y in y_val], dtype=np.float32)
    else:
        # Split train/val si pas fourni
        split_idx = int(0.8 * len(X_train_scaled))
        X_val_scaled = X_train_scaled[split_idx:]
        val_durations = durations[split_idx:]
        val_events = events[split_idx:]
        X_train_scaled = X_train_scaled[:split_idx]
        durations = durations[:split_idx]
        events = events[:split_idx]

    # Transformation des labels pour PyCox
    labtrans = LabTransCoxTime()
    y_train_pycox = labtrans.fit_transform(durations, events)
    y_val_pycox = labtrans.transform(val_durations, val_events)

    # Conversion en tenseurs pour PyCox
    train_dataset = (X_train_scaled, y_train_pycox)
    val_dataset = (X_val_scaled, y_val_pycox)

    # Configuration du réseau
    in_features = X_train_scaled.shape[1]
    num_nodes = hidden_layers
    out_features = 1

    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=num_nodes,
        out_features=out_features,
        batch_norm=batch_norm,
        dropout=dropout,
    )

    # Modèle PyCox
    model = PyCoxCoxPH(net, torch.optim.Adam)
    model.labtrans = labtrans  # Assigner labtrans après création

    # Configuration de l'optimiseur et scheduler
    model.optimizer.set_lr(learning_rate)
    if weight_decay > 0:
        model.optimizer.param_groups[0]["weight_decay"] = weight_decay

    # Learning rate scheduler si activé
    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer.optimizer,
            mode="max",  # Pour maximiser le C-index
            factor=lr_factor,
            patience=lr_patience,
            min_lr=learning_rate * 0.01,
        )

    # Fit du modèle avec métriques améliorées
    cindex_train_hist = []
    cindex_val_hist = []
    train_loss_hist = []
    val_loss_hist = []

    # Early stopping amélioré
    best_cindex_val = 0.0
    best_epoch = 0
    patience_counter = 0
    no_improvement_epochs = 0

    print(
        f"   🚀 Début entraînement DeepSurv ({epochs} epochs max, patience={patience})"
    )

    for epoch in range(epochs):
        # Une époque d'entraînement
        model.fit(
            X_train_scaled,
            y_train_pycox,
            batch_size,
            1,  # 1 epoch at a time
            callbacks=None,
            verbose=False,
            val_data=(X_val_scaled, y_val_pycox),
        )

        # Prédictions
        train_pred = model.predict(X_train_scaled).flatten()
        val_pred = model.predict(X_val_scaled).flatten()

        # C-index
        try:
            cindex_train = concordance_index_ipcw(y_train, y_train, train_pred)[0]
            cindex_val = concordance_index_ipcw(y_train, y_val, val_pred)[0]
        except Exception as e:
            cindex_train = np.nan
            cindex_val = np.nan

        cindex_train_hist.append(cindex_train)
        cindex_val_hist.append(cindex_val)

        # Loss computation (try to get it from model if possible)
        train_loss = np.nan
        val_loss = np.nan
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        # Learning rate scheduling
        if scheduler is not None and not np.isnan(cindex_val):
            scheduler.step(cindex_val)

        # Early stopping amélioré
        improvement = False
        if not np.isnan(cindex_val):
            if cindex_val > best_cindex_val:
                best_cindex_val = cindex_val
                best_epoch = epoch + 1
                patience_counter = 0
                improvement = True
            else:
                patience_counter += 1

        if not improvement:
            no_improvement_epochs += 1
        else:
            no_improvement_epochs = 0

        # Affichage avec plus de détails
        current_lr = (
            model.optimizer.param_groups[0]["lr"]
            if hasattr(model.optimizer, "param_groups")
            else learning_rate
        )
        status = "🟢" if improvement else "🔴" if no_improvement_epochs > 10 else "🟡"

        if (epoch + 1) % 10 == 0 or epoch < 10 or improvement:
            print(
                f"Epoch {epoch+1:4d}/{epochs} {status} | LR: {current_lr:.6f} | "
                f"C-Index Train: {cindex_train:.5f} | C-Index Val: {cindex_val:.5f} | "
                f"Best: {best_cindex_val:.5f}@{best_epoch}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"   🔄 Early stopping à l'époque {epoch+1} (patience={patience})")
            print(f"   🏆 Pas d'amélioration depuis {no_improvement_epochs} époques")
            break

    # Créer un wrapper pour la compatibilité avec scikit-survival
    wrapper = PyCoxWrapper(model, scaler, labtrans)

    # Évaluation avec concordance_index_ipcw
    # Reconstruction des données originales pour l'évaluation
    X_train_orig = X_train
    y_train_orig = y_train

    if X_val is not None and y_val is not None:
        X_val_orig = X_val
        y_val_orig = y_val
    else:
        # Utiliser le même split que pour l'entraînement
        split_idx = int(0.8 * len(X_train))
        X_val_orig = (
            X_train.iloc[split_idx:]
            if hasattr(X_train, "iloc")
            else X_train[split_idx:]
        )
        y_val_orig = y_train[split_idx:]
        X_train_orig = (
            X_train.iloc[:split_idx]
            if hasattr(X_train, "iloc")
            else X_train[:split_idx]
        )
        y_train_orig = y_train[:split_idx]

    # Calcul du C-index avec concordance_index_ipcw
    try:
        train_pred = wrapper.predict(X_train_orig)
        val_pred = wrapper.predict(X_val_orig)

        # C-index sur les données d'entraînement
        cindex_train = concordance_index_ipcw(y_train_orig, y_train_orig, train_pred)[0]
        # C-index sur les données de validation
        cindex_val = concordance_index_ipcw(y_train_orig, y_val_orig, val_pred)[0]

        print(f"   📊 C-Index Train: {cindex_train:.5f}")
        print(f"   📊 C-Index Validation: {cindex_val:.5f}")

    except Exception as e:
        print(f"   ⚠️  Erreur lors du calcul du C-index: {e}")

    # Affichage des résultats détaillés
    print(f"\n   📊 === RÉSULTATS D'ENTRAÎNEMENT ===")

    # Statistiques sur la convergence
    best_cindex_val_overall = np.nanmax(cindex_val_hist)
    best_epoch_overall = np.nanargmax(cindex_val_hist) + 1
    final_cindex_train = cindex_train_hist[-1] if cindex_train_hist else np.nan
    final_cindex_val = cindex_val_hist[-1] if cindex_val_hist else np.nan

    # Analyse de la convergence
    if len(cindex_val_hist) > 10:
        # Calcul de la tendance sur les 10 dernières époques
        recent_vals = [x for x in cindex_val_hist[-10:] if not np.isnan(x)]
        if len(recent_vals) > 5:
            trend = np.polyfit(range(len(recent_vals)), recent_vals, 1)[0]
            trend_status = (
                "📈 Croissante"
                if trend > 0.001
                else "📉 Décroissante" if trend < -0.001 else "➡️  Stable"
            )
        else:
            trend_status = "❓ Indéterminée"
    else:
        trend_status = "❓ Trop peu d'époques"

    print(
        f"   🏆 Meilleur C-Index Validation: {best_cindex_val_overall:.5f} (époque {best_epoch_overall})"
    )
    print(
        f"   📈 C-Index Final: Train={final_cindex_train:.5f}, Val={final_cindex_val:.5f}"
    )
    print(f"   📊 Tendance récente: {trend_status}")
    print(f"   ⏱️  Nombre d'époques: {len(cindex_train_hist)}")

    # Recommandations basées sur les résultats
    val_improvement = best_cindex_val_overall - (
        cindex_val_hist[0] if cindex_val_hist else 0
    )
    if val_improvement < 0.01:
        print(f"   ⚠️  Amélioration faible ({val_improvement:.4f}). Considérez:")
        print(f"      • Augmenter le learning rate ou réduire la régularisation")
        print(f"      • Changer l'architecture du réseau")
        print(f"      • Vérifier la qualité des données")
    elif val_improvement > 0.05:
        print(f"   ✅ Bonne amélioration ({val_improvement:.4f})")

    if final_cindex_train - final_cindex_val > 0.1:
        print(
            f"   ⚠️  Possible overfitting (écart: {final_cindex_train - final_cindex_val:.4f})"
        )
        print(f"      • Augmenter dropout ou weight decay")
        print(f"      • Réduire la complexité du modèle")

    return wrapper
