#!/usr/bin/env python3
"""
Unified Training Script: Train All Models, Search Ensembles, Save Models

This script:
  1) Loads processed training data (feature-engineered + preprocessed),
  2) Generates Out-of-Fold predictions for all base models using GroupKFold (by center) when available,
  3) Evaluates every rank-averaging ensemble combination to find the best blend (IPCW C-index, with TAU),
  4) Retrains each base model on 100% of the processed training data and saves them separately.
Optionally, it writes a CSV report of base and ensemble scores.
"""
import os
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


def main():
    print("=" * 80)
    print(" UNIFIED TRAINING: ALL MODELS + ENSEMBLE SEARCH + SAVE MODELS")
    print("=" * 80)

    # --- 0) TAG & EXPERIMENT DIR ---
    tag, cfg_signature, _, _ = compute_tag_with_signature()
    exp_dir = ensure_experiment_dir(tag)
    print(f"[INFO] Experiment tag: {tag} (config sig: {cfg_signature})")

    # --- 1) LOAD PROCESSED DATA ---
    print("\n[STEP 1/4] Loading processed training data...")
    X, y = load_training_dataset_csv(
        X_train_path="datasets_processed/X_train_processed.csv",
        y_train_path="datasets_processed/y_train_processed.csv",
    )
    groups = None
    if "CENTER_GROUP" in X.columns:
        groups = X["CENTER_GROUP"].astype(str).values
        X = X.drop(columns=["CENTER_GROUP"])  # not a feature
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])  # not a feature
    print(f"   -> Dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    # Save dataset shape in training report stub
    training_report = {
        "dataset": {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])},
        "cv": {},
        "models": {},
        "base_scores": {},
        "best_ensemble": {},
    }

    # --- 2) CONFIGURE MODELS + CV & OOF ---
    print("\n[STEP 2/4] Configuring models and cross-validation...")
    models = get_survival_models()  # dict[str, estimator]
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
    # Record model params for traceability
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
                oof_predictions.loc[va_idx, name] = preds
                # Per-fold score (using training fold as reference for IPCW)
                fold_ipcw = concordance_index_ipcw(y_tr, y_va, preds, tau=TAU)[0]
                fold_scores[name].append(fold_ipcw)
                print(f"  {name:<20} fold IPCW: {fold_ipcw:.4f}")
            except Exception as e:
                print(f"  [WARN] {name} failed on fold: {e}")
                fold_scores[name].append(np.nan)

    if oof_predictions.isnull().values.any():
        nan_cols = [
            col for col in oof_predictions.columns if oof_predictions[col].isnull().any()
        ]
        print(
            f"\n[WARN] Removing models with incomplete OOF predictions (NaNs detected): {nan_cols}"
        )
        oof_predictions.drop(columns=nan_cols, inplace=True, errors="ignore")

    # Remove models that failed entirely
    oof_predictions.dropna(axis=1, how="all", inplace=True)
    valid_models = list(oof_predictions.columns)
    print(f"\nValid models with OOF preds: {valid_models}")

    # --- 3) EVALUATE BASE MODELS AND ALL ENSEMBLES ---
    print("\n[STEP 3/4] Evaluating base models and all rank-ensemble combinations...")
    summary_rows = []
    base_scores = {}
    for name in valid_models:
        preds = oof_predictions[name].astype(float).values
        score = concordance_index_ipcw(y, y, preds, tau=TAU)[0]
        base_scores[name] = score
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

    # Collate and save summary
    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    os.makedirs("reports", exist_ok=True)
    summary_path = os.path.join("reports", "ensemble_ranking.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nRanking saved to: {summary_path}")
    # Also save into experiment directory
    summary_path_tag = os.path.join(exp_dir, "ensemble_ranking.csv")
    summary_df.to_csv(summary_path_tag, index=False)
    # Save OOF predictions and fold scores
    oof_predictions.to_csv(os.path.join(exp_dir, "oof_predictions.csv"), index=True)
    
    # --- ERROR ANALYSIS ---
    try:
        analyze_cv_errors(oof_predictions, y, exp_dir)
    except Exception as e:
        print(f"[WARN] Error analysis failed: {e}")

    with open(os.path.join(exp_dir, "fold_scores.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                k: [None if pd.isna(v) else float(v) for v in vals]
                for k, vals in fold_scores.items()
            },
            f,
            indent=2,
        )

    best_name = summary_df.iloc[0]["name"]
    best_score = summary_df.iloc[0]["score"]
    print(f"\n---> Best combo: {best_name} (OOF IPCW={best_score:.5f})")
    training_report["best_ensemble"] = {
        "name": best_name,
        "oof_ipcw": float(best_score),
        "tau": TAU,
    }

    # --- 4) RETRAIN ALL MODELS ON FULL DATA AND SAVE ---
    print("\n[STEP 4/4] Retraining each base model on full data and saving...")
    os.makedirs("models", exist_ok=True)
    for name, est in models.items():
        try:
            est.fit(X, y)
            out_path = os.path.join("models", f"model_{name}.joblib")
            joblib.dump(est, out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  [WARN] Failed to save {name}: {e}")

    # Save best ensemble descriptor for convenience
    meta = {
        "best_combo": best_name,
        "best_oof_ipcw": float(best_score),
        "models": valid_models,
    }
    with open(os.path.join("models", "ensemble_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # Save training report
    with open(
        os.path.join(exp_dir, "training_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(training_report, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
