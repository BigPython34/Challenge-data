#!/usr/bin/env python3
"""Clean training stage with optional outlier filtering."""

import argparse
import itertools
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sksurv.metrics import concordance_index_ipcw

from src.config import TAU
from src.modeling.error_analysis import analyze_cv_errors
from src.modeling.train import get_survival_models
from src.utils.experiment import compute_tag_with_signature, ensure_experiment_dir


def _json_safe(obj):
    """Ensure that objects can be stored inside JSON logs."""

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
    """Return the freshest experiment directory that contains error analysis."""

    if not os.path.exists(base_dir):
        return None

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        return None

    subdirs.sort(key=os.path.getmtime, reverse=True)

    for candidate in subdirs:
        if os.path.exists(os.path.join(candidate, "error_analysis_detailed.csv")):
            return candidate

    return None


def get_outliers_to_remove(n_remove=100):
    """Return the list of IDs that show the largest errors in the latest experiment."""

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

    if "ID" not in df.columns:
        print("[INFO] ID column missing in error analysis. Trying to recover from processed train set...")
        x_train_path = "datasets_processed/X_train_processed.csv"
        if os.path.exists(x_train_path):
            x_train_df = pd.read_csv(x_train_path)
            if "ID" in x_train_df.columns and len(x_train_df) == len(df):
                df["ID"] = x_train_df["ID"].astype(str).values
                print("[INFO] IDs recovered from X_train_processed.csv.")
            else:
                msg = (
                    "[ERROR] Could not match IDs from X_train_processed.csv. "
                    f"({len(x_train_df)} rows vs {len(df)} error rows.)"
                )
                print(msg)
                return []
        else:
            print(f"[ERROR] {x_train_path} not found.")
            return []

    if "Mean_Abs_Error" not in df.columns:
        print("[ERROR] Mean_Abs_Error column missing in error analysis.")
        return []

    hardest = df.nlargest(n_remove, "Mean_Abs_Error")
    outlier_ids = hardest["ID"].astype(str).tolist()
    print(f"[INFO] Identified {len(outlier_ids)} outliers (top 5: {outlier_ids[:5]}).")
    return outlier_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train using the cleaned dataset produced after removing hard outliers."
    )
    parser.add_argument(
        "--outliers",
        "-n",
        type=int,
        default=100,
        help="Number of hardest samples (by mean absolute error) to drop before training.",
    )
    return parser.parse_args()


def run_clean_training(n_outliers: int) -> None:
    """Execute the cleaned training workflow."""

    print("=" * 80)
    print(" REFINED TRAINING: OUTLIER REMOVAL + ENSEMBLE SEARCH")
    print("=" * 80)

    outlier_ids = get_outliers_to_remove(n_outliers)

    tag, cfg_signature, _, _ = compute_tag_with_signature()
    tag = f"{tag}_clean{n_outliers}"
    exp_dir = ensure_experiment_dir(tag)
    print(f"[INFO] Experiment tag: {tag} (config sig: {cfg_signature})")

    print("\n[STEP 1/4] Loading processed training data...")
    X_raw = pd.read_csv("datasets_processed/X_train_processed.csv")
    y_raw = pd.read_csv("datasets_processed/y_train_processed.csv")

    if outlier_ids and "ID" in X_raw.columns:
        mask = ~X_raw["ID"].astype(str).isin(outlier_ids)
        before = len(X_raw)
        X_raw = X_raw[mask].reset_index(drop=True)
        y_raw = y_raw[mask].reset_index(drop=True)
        after = len(X_raw)
        print(f"[FILTER] Removed {before - after} outliers. Remaining: {after}")
        with open(os.path.join(exp_dir, "removed_outliers.json"), "w", encoding="utf-8") as fh:
            json.dump(outlier_ids, fh, indent=2)
    elif outlier_ids:
        print("[WARN] Could not filter outliers because ID column is missing in X_train_processed.")

    groups = None
    if "CENTER_GROUP" in X_raw.columns:
        groups = X_raw["CENTER_GROUP"].astype(str).values
        X = X_raw.drop(columns=["CENTER_GROUP"])
    else:
        X = X_raw.copy()

    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    y = pd.DataFrame(y_raw).copy()
    if "OS_STATUS" not in y.columns or "OS_YEARS" not in y.columns:
        raise ValueError("y_train_processed.csv must contain OS_STATUS and OS_YEARS columns.")

    from sksurv.util import Surv

    y_struct = Surv.from_arrays(
        event=y["OS_STATUS"].astype(bool),
        time=y["OS_YEARS"].astype(float),
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

    print("\n[STEP 2/4] Configuring models and cross-validation...")
    models = get_survival_models()
    model_names = list(models.keys())
    print(f"   -> Models: {model_names}")

    n_splits = 5
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, groups=groups)
        training_report["cv"] = {"type": "GroupKFold", "n_splits": n_splits}
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X)
        training_report["cv"] = {
            "type": "KFold",
            "n_splits": n_splits,
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
    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        print(f"\n--- FOLD {fold_idx}/{n_splits} ---")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_struct[train_idx], y_struct[valid_idx]

        for name, est in models.items():
            try:
                est.fit(X_train, y_train)
                preds = est.predict(X_valid)
                oof_predictions.loc[X_valid.index, name] = preds
                fold_ipcw = concordance_index_ipcw(y_train, y_valid, preds, tau=TAU)[0]
                fold_scores[name].append(fold_ipcw)
                print(f"  {name:<20} fold IPCW: {fold_ipcw:.4f}")
            except Exception as err:
                print(f"  [WARN] {name} failed on fold: {err}")
                fold_scores[name].append(np.nan)

    if oof_predictions.isnull().values.any():
        nan_cols = [col for col in oof_predictions.columns if oof_predictions[col].isnull().any()]
        print(f"[WARN] Dropping models with incomplete OOF predictions: {nan_cols}")
        oof_predictions.drop(columns=nan_cols, inplace=True, errors="ignore")

    oof_predictions.dropna(axis=1, how="all", inplace=True)
    valid_models = list(oof_predictions.columns)
    print(f"\nValid models with OOF preds: {valid_models}")

    print("\n[STEP 3/4] Evaluating base models and ensembles...")
    summary_rows = []
    for name in valid_models:
        preds = oof_predictions[name].astype(float).values
        score = concordance_index_ipcw(y_struct, y_struct, preds, tau=TAU)[0]
        summary_rows.append({"name": name, "size": 1, "score": score})
        training_report["base_scores"][name] = float(score)
        print(f"  Base OOF IPCW: {name:<20} = {score:.5f}")

    for k in range(2, len(valid_models) + 1):
        for combo in itertools.combinations(valid_models, k):
            combo_name = " + ".join(combo)
            combo_ranks = oof_predictions[list(combo)].rank()
            ensemble_rank_score = combo_ranks.mean(axis=1).values
            ens_ipcw = concordance_index_ipcw(y_struct, y_struct, ensemble_rank_score, tau=TAU)[0]
            summary_rows.append({"name": combo_name, "size": k, "score": ens_ipcw})
            print(f"  Ensemble OOF IPCW: {combo_name} = {ens_ipcw:.5f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    os.makedirs("reports", exist_ok=True)
    summary_path = os.path.join("reports", "ensemble_ranking_clean.csv")
    summary_df.to_csv(summary_path, index=False)
    summary_path_tag = os.path.join(exp_dir, "ensemble_ranking.csv")
    summary_df.to_csv(summary_path_tag, index=False)
    oof_predictions.to_csv(os.path.join(exp_dir, "oof_predictions.csv"), index=True)

    try:
        analyze_cv_errors(oof_predictions, y_struct, exp_dir)
    except Exception as err:
        print(f"[WARN] Error analysis failed: {err}")

    with open(os.path.join(exp_dir, "fold_scores.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                k: [None if pd.isna(v) else float(v) for v in vals]
                for k, vals in fold_scores.items()
            },
            fh,
            indent=2,
        )

    best_name = summary_df.iloc[0]["name"]
    best_score = summary_df.iloc[0]["score"]
    training_report["best_ensemble"] = {
        "name": best_name,
        "oof_ipcw": float(best_score),
        "tau": TAU,
    }
    print(f"\n---> Best combo (CLEAN): {best_name} (OOF IPCW={best_score:.5f})")

    print("\n[STEP 4/4] Retraining each base model on clean data...")
    os.makedirs("models", exist_ok=True)
    for name, est in models.items():
        try:
            est.fit(X, y_struct)
            out_path = os.path.join("models", f"model_{name}_clean.joblib")
            joblib.dump(est, out_path)
            print(f"  Saved: {out_path}")
        except Exception as err:
            print(f"  [WARN] Failed to save {name}: {err}")

    meta = {
        "best_combo": best_name,
        "best_oof_ipcw": float(best_score),
        "models": valid_models,
    }
    with open(os.path.join("models", "ensemble_meta_clean.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    with open(os.path.join(exp_dir, "training_report.json"), "w", encoding="utf-8") as fh:
        json.dump(training_report, fh, indent=2)

    print("\nDone.")


def main():
    args = parse_args()
    run_clean_training(args.outliers)


if __name__ == "__main__":
    main()
