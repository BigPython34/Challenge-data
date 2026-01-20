#!/usr/bin/env python3
"""
Refined Training Script: Train with Outlier Removal

This script is a variant of `2_c_find_best_ensemble.py`.
It performs an additional step:
1. Loads the error analysis from the latest experiment.
2. Identifies the top N "hardest" samples (highest Mean Absolute Error).
3. Removes them from the training set.
4. Runs the standard Cross-Validation and Ensemble search on the cleaned data.

Usage:
    python 2_d_train_clean.py [n_outliers_to_remove]
    Default n_outliers_to_remove = 100
"""
import os
import sys
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sksurv.metrics import concordance_index_ipcw
from src.modeling.train import load_training_dataset_csv, get_survival_models
from src.modeling.error_analysis import analyze_cv_errors
from src.config import TAU, PREPROCESSING, EXPERIMENT
from src.utils.experiment import compute_tag_with_signature, ensure_experiment_dir

def _json_safe(obj):
    """Recursively convert objects to JSON-serializable structures."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)

def get_latest_experiment_dir(base_dir="results/experiments"):
    """Finds the most recently modified experiment directory that contains error analysis."""
    if not os.path.exists(base_dir):
        return None
    
    # Get all subdirectories
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    
    # Sort by modification time (newest first)
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    # Find the first one that has the error analysis file
    for d in subdirs:
        if os.path.exists(os.path.join(d, "error_analysis_detailed.csv")):
            return d
            
    return None

def get_outliers_to_remove(n_remove=100):
    """Identifies IDs of the hardest samples from the latest experiment."""
    exp_dir = get_latest_experiment_dir()
    if not exp_dir:
        print("[WARN] No previous experiment found to identify outliers.")
        return []
    
    error_path = os.path.join(exp_dir, "error_analysis_detailed.csv")
    if not os.path.exists(error_path):
        print(f"[WARN] Error analysis file not found in {exp_dir}.")
        return []
    
    print(f"[INFO] Loading outliers from: {error_path}")
    df = pd.read_csv(error_path)
    
    # If ID is missing, try to recover it from X_train_processed
    if "ID" not in df.columns:
        print("[INFO] ID column missing in error analysis. Attempting to recover from X_train_processed.csv...")
        try:
            x_train_path = "datasets_processed/X_train_processed.csv"
            if os.path.exists(x_train_path):
                x_train_df = pd.read_csv(x_train_path)
                if "ID" in x_train_df.columns:
                    # Check length match
                    if len(df) == len(x_train_df):
                        df["ID"] = x_train_df["ID"].values
                        print("[INFO] Successfully recovered IDs.")
                    else:
                        print(f"[ERROR] Length mismatch: Error analysis has {len(df)}, X_train has {len(x_train_df)}.")
                        return []
                else:
                    print("[ERROR] ID column missing in X_train_processed.csv.")
                    return []
            else:
                print(f"[ERROR] {x_train_path} not found.")
                return []
        except Exception as e:
            print(f"[ERROR] Failed to recover IDs: {e}")
            return []
        
    # Sort by Mean_Abs_Error descending
    if "Mean_Abs_Error" not in df.columns:
        print("[ERROR] Mean_Abs_Error column missing.")
        return []
        
    hardest = df.nlargest(n_remove, "Mean_Abs_Error")
    outlier_ids = hardest["ID"].astype(str).tolist()
    
    print(f"[INFO] Identified {len(outlier_ids)} outliers to remove.")
    return outlier_ids
    
    print(f"[INFO] Identified {len(outlier_ids)} outliers to remove.")
    print(f"       Top 5 outliers: {outlier_ids[:5]}")
    return outlier_ids

def main():
    print("=" * 80)
    print(" REFINED TRAINING: OUTLIER REMOVAL + ENSEMBLE SEARCH")
    print("=" * 80)

    # Parse args
    n_remove = 100
    if len(sys.argv) > 1:
        try:
            n_remove = int(sys.argv[1])
        except ValueError:
            pass
            
    # --- 0) IDENTIFY OUTLIERS (BEFORE CREATING NEW EXPERIMENT DIR) ---
    # We must do this first, otherwise get_latest_experiment_dir might pick up the new empty folder
    outlier_ids = get_outliers_to_remove(n_remove)

    # --- 1) TAG & EXPERIMENT DIR ---
    tag, cfg_signature, _, _ = compute_tag_with_signature()
    # Append suffix to tag to distinguish
    tag = f"{tag}_clean{n_remove}"
    exp_dir = ensure_experiment_dir(tag)
    print(f"[INFO] Experiment tag: {tag}")

    # --- 2) LOAD PROCESSED DATA ---
    print("\n[STEP 1/4] Loading processed training data...")
    # We need to load raw first to get IDs for filtering
    X_raw = pd.read_csv("datasets_processed/X_train_processed.csv")
    y_raw = pd.read_csv("datasets_processed/y_train_processed.csv")
    
    # Filter
    if outlier_ids:
        if "ID" in X_raw.columns:
            mask = ~X_raw["ID"].astype(str).isin(outlier_ids)
            n_before = len(X_raw)
            X_raw = X_raw[mask]
            y_raw = y_raw[mask]
            n_after = len(X_raw)
            print(f"[FILTER] Removed {n_before - n_after} outliers. Remaining: {n_after}")
            
            # Save the list of removed outliers
            with open(os.path.join(exp_dir, "removed_outliers.json"), "w") as f:
                json.dump(outlier_ids, f, indent=2)
        else:
            print("[WARN] ID column not found in X_train_processed. Cannot filter outliers.")

    # Prepare for sksurv
    # Drop ID and CENTER_GROUP for training
    groups = None
    if "CENTER_GROUP" in X_raw.columns:
        groups = X_raw["CENTER_GROUP"].astype(str).values
        X = X_raw.drop(columns=["CENTER_GROUP"])
    else:
        X = X_raw.copy()
        
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    # Convert y to structured array
    from sksurv.util import Surv
    y = Surv.from_arrays(
        event=y_raw["OS_STATUS"].astype(bool),
        time=y_raw["OS_YEARS"].astype(float),
    )

    print(f"   -> Dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    
    training_report = {
        "dataset": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
        "outliers_removed": len(outlier_ids),
        "cv": {},
        "models": {},
        "base_scores": {},
        "best_ensemble": {},
    }

    # --- 2) CONFIGURE MODELS + CV & OOF ---
    print("\n[STEP 2/4] Configuring models and cross-validation...")
    models = get_survival_models()
    model_names = list(models.keys())
    print(f"   -> Models: {model_names}")

    N_SPLITS = 5
    if groups is not None:
        splitter = GroupKFold(n_splits=N_SPLITS)
        split_iter = splitter.split(X, groups=groups)
        training_report["cv"] = {"type": "GroupKFold", "n_splits": N_SPLITS}
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        split_iter = splitter.split(X)
        training_report["cv"] = {
            "type": "KFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": 42,
        }

    oof_predictions = pd.DataFrame(index=X.index, columns=model_names, dtype=float)
    fold_scores = {name: [] for name in model_names}
    
    for name, est in models.items():
        try:
            training_report["models"][name] = _json_safe(est.get_params(deep=False))
        except Exception:
            training_report["models"][name] = "<params_unavailable>"

    print("\n[STEP 2.1] Generating Out-of-Fold (OOF) predictions...")
    for i, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        print(f"\n--- FOLD {i}/{N_SPLITS} ---")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        for name, est in models.items():
            try:
                est.fit(X_tr, y_tr)
                preds = est.predict(X_va)
                oof_predictions.loc[X_va.index, name] = preds
                fold_ipcw = concordance_index_ipcw(y_tr, y_va, preds, tau=TAU)[0]
                fold_scores[name].append(fold_ipcw)
                print(f"  {name:<20} fold IPCW: {fold_ipcw:.4f}")
            except Exception as e:
                print(f"  [WARN] {name} failed on fold: {e}")
                fold_scores[name].append(np.nan)

    if oof_predictions.isnull().values.any():
        nan_cols = [col for col in oof_predictions.columns if oof_predictions[col].isnull().any()]
        oof_predictions.drop(columns=nan_cols, inplace=True, errors="ignore")

    oof_predictions.dropna(axis=1, how="all", inplace=True)
    valid_models = list(oof_predictions.columns)
    print(f"\nValid models with OOF preds: {valid_models}")

    # --- 3) EVALUATE BASE MODELS AND ALL ENSEMBLES ---
    print("\n[STEP 3/4] Evaluating base models and all rank-ensemble combinations...")
    summary_rows = []
    for name in valid_models:
        preds = oof_predictions[name].astype(float).values
        score = concordance_index_ipcw(y, y, preds, tau=TAU)[0]
        summary_rows.append({"name": name, "size": 1, "score": score})
        print(f"  Base OOF IPCW: {name:<20} = {score:.5f}")
        training_report["base_scores"][name] = float(score)

    ensemble_results = []
    for k in range(2, len(valid_models) + 1):
        for combo in itertools.combinations(valid_models, k):
            combo_name = " + ".join(combo)
            combo_ranks = oof_predictions[list(combo)].rank()
            ensemble_rank_score = combo_ranks.mean(axis=1).values
            ens_ipcw = concordance_index_ipcw(y, y, ensemble_rank_score, tau=TAU)[0]
            ensemble_results.append((combo_name, k, ens_ipcw))
            summary_rows.append({"name": combo_name, "size": k, "score": ens_ipcw})
            print(f"  Ensemble OOF IPCW: {combo_name} = {ens_ipcw:.5f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    os.makedirs("reports", exist_ok=True)
    summary_path = os.path.join("reports", "ensemble_ranking_clean.csv")
    summary_df.to_csv(summary_path, index=False)
    
    summary_path_tag = os.path.join(exp_dir, "ensemble_ranking.csv")
    summary_df.to_csv(summary_path_tag, index=False)
    oof_predictions.to_csv(os.path.join(exp_dir, "oof_predictions.csv"), index=True)
    
    # --- ERROR ANALYSIS (On Cleaned Data) ---
    try:
        # We need to pass IDs back if possible, but analyze_cv_errors expects index to match
        # oof_predictions index is RangeIndex usually unless we set it.
        # But here we filtered, so index is preserved from X_raw (which has gaps).
        # analyze_cv_errors uses y["time"] etc.
        analyze_cv_errors(oof_predictions, y, exp_dir)
    except Exception as e:
        print(f"[WARN] Error analysis failed: {e}")

    best_name = summary_df.iloc[0]["name"]
    best_score = summary_df.iloc[0]["score"]
    print(f"\n---> Best combo (CLEAN): {best_name} (OOF IPCW={best_score:.5f})")
    training_report["best_ensemble"] = {
        "name": best_name,
        "oof_ipcw": float(best_score),
        "tau": TAU,
    }

    # --- 4) RETRAIN ALL MODELS ON FULL CLEAN DATA AND SAVE ---
    print("\n[STEP 4/4] Retraining each base model on full CLEAN data and saving...")
    os.makedirs("models", exist_ok=True)
    for name, est in models.items():
        try:
            est.fit(X, y)
            # Save with _clean suffix to avoid overwriting main models?
            # Or overwrite if this is the intended final model.
            # Let's use suffix for safety.
            out_path = os.path.join("models", f"model_{name}_clean.joblib")
            joblib.dump(est, out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  [WARN] Failed to save {name}: {e}")

    with open(os.path.join(exp_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    print("\nDone.")

if __name__ == "__main__":
    main()
