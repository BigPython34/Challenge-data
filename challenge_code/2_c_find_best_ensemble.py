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
from src.modeling.profile_strategy import (
    prepare_training_subsets,
    get_profile_mode,
    record_feature_subset,
)
from src.config import TAU, PREPROCESSING, EXPERIMENT, DATA_PATHS, ID_COLUMNS
from src.utils.experiment import compute_tag_with_signature, ensure_experiment_dir


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
    id_col = ID_COLUMNS["patient"]
    center_group_path = DATA_PATHS.get("center_group_train")
    if center_group_path and os.path.exists(center_group_path):
        try:
            center_df = pd.read_csv(center_group_path)
            if id_col not in X.columns:
                raise ValueError(
                    f"ID column '{id_col}' is required to align center metadata but was not found in processed data."
                )
            center_map = center_df.set_index(id_col)["CENTER_GROUP"].astype(str)
            groups = (
                X[id_col]
                .map(center_map)
                .fillna(center_map.mode().iloc[0] if not center_map.empty else "CENTER_OTHER")
                .values
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Unable to load center grouping metadata ({exc}); falling back to dataset column if available.")
            if "CENTER_GROUP" in X.columns:
                groups = X["CENTER_GROUP"].astype(str).values
    elif "CENTER_GROUP" in X.columns:
        groups = X["CENTER_GROUP"].astype(str).values
    X = X.drop(columns=["CENTER_GROUP"], errors="ignore")
    if id_col in X.columns:
        X = X.drop(columns=[id_col], errors="ignore")
    print(f"   -> Dataset: {X.shape[0]} samples, {X.shape[1]} features.")

    strategy_mode = get_profile_mode()
    subsets = prepare_training_subsets(X, y, groups)
    print(f"[INFO] Profile strategy mode: {strategy_mode} ({len(subsets)} subset(s))")

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    overall_report = {"strategy_mode": strategy_mode, "subsets": {}}

    for subset_label, dataset_slice in subsets.items():
        subset_suffix = "" if subset_label == "default" else f"_{subset_label}"
        X_subset = dataset_slice.X.copy()
        y_subset = dataset_slice.y
        groups_subset = dataset_slice.groups

        print("\n" + "-" * 80)
        print(f"[SUBSET] Training on '{subset_label}' ({X_subset.shape[0]} samples, {X_subset.shape[1]} features)")
        print("-" * 80)

        if len(X_subset) < 2:
            raise ValueError(f"Subset '{subset_label}' is too small to train models.")

        record_feature_subset(subset_label, X_subset.columns)

        models = get_survival_models()
        model_names = list(models.keys())
        print(f"   -> Models: {model_names}")

        N_SPLITS = 5
        if groups_subset is not None:
            unique_groups = np.unique(groups_subset)
            n_splits = min(N_SPLITS, len(unique_groups))
            if n_splits < 2:
                raise ValueError(
                    f"Not enough unique centers for GroupKFold (subset {subset_label})."
                )
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X_subset, groups=groups_subset)
            cv_meta = {"type": "GroupKFold", "n_splits": n_splits}
        else:
            n_splits = min(N_SPLITS, len(X_subset))
            if n_splits < 2:
                raise ValueError(
                    f"Not enough samples for KFold (subset {subset_label})."
                )
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_iter = splitter.split(X_subset)
            cv_meta = {
                "type": "KFold",
                "n_splits": n_splits,
                "shuffle": True,
                "random_state": 42,
            }

        training_report = {
            "dataset": {
                "n_samples": int(X_subset.shape[0]),
                "n_features": int(X_subset.shape[1]),
            },
            "cv": cv_meta,
            "models": {},
            "base_scores": {},
            "best_ensemble": {},
        }

        oof_predictions = pd.DataFrame(index=X_subset.index, columns=model_names, dtype=float)
        fold_scores = {name: [] for name in model_names}
        for name, est in models.items():
            try:
                training_report["models"][name] = est.get_params(deep=False)
            except Exception:
                training_report["models"][name] = "<params_unavailable>"

        print("\n[STEP 2/4] Generating OOF predictions for subset...")
        for i, (tr_idx, va_idx) in enumerate(split_iter, start=1):
            print(f"\n--- Subset {subset_label}: FOLD {i}/{n_splits} ---")
            X_tr, X_va = X_subset.iloc[tr_idx], X_subset.iloc[va_idx]
            y_tr, y_va = y_subset[tr_idx], y_subset[va_idx]

            for name, est in models.items():
                try:
                    est.fit(X_tr, y_tr)
                    preds = est.predict(X_va)
                    oof_predictions.loc[X_subset.index[va_idx], name] = preds
                    fold_ipcw = concordance_index_ipcw(y_tr, y_va, preds, tau=TAU)[0]
                    fold_scores[name].append(fold_ipcw)
                    print(f"  {name:<20} fold IPCW: {fold_ipcw:.4f}")
                except Exception as e:
                    print(f"  [WARN] {name} failed on fold: {e}")
                    fold_scores[name].append(np.nan)

        oof_predictions.dropna(axis=1, how="all", inplace=True)
        valid_models = list(oof_predictions.columns)
        if not valid_models:
            raise RuntimeError(f"No valid models produced OOF predictions for {subset_label}.")
        print(f"\nValid models with OOF preds ({subset_label}): {valid_models}")

        print("\n[STEP 3/4] Evaluating ensembles for subset...")
        summary_rows = []
        for name in valid_models:
            preds = oof_predictions[name].astype(float).values
            score = concordance_index_ipcw(y_subset, y_subset, preds, tau=TAU)[0]
            summary_rows.append({"name": name, "size": 1, "score": score})
            training_report["base_scores"][name] = float(score)
            print(f"  Base OOF IPCW: {name:<20} = {score:.5f}")

        for k in range(2, len(valid_models) + 1):
            for combo in itertools.combinations(valid_models, k):
                combo_name = " + ".join(combo)
                combo_ranks = oof_predictions[list(combo)].rank()
                ensemble_rank_score = combo_ranks.mean(axis=1).values
                ens_ipcw = concordance_index_ipcw(y_subset, y_subset, ensemble_rank_score, tau=TAU)[0]
                summary_rows.append({"name": combo_name, "size": k, "score": ens_ipcw})
                print(f"  Ensemble OOF IPCW: {combo_name} = {ens_ipcw:.5f}")

        summary_df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
        summary_path = os.path.join("reports", f"ensemble_ranking{subset_suffix}.csv")
        summary_df.to_csv(summary_path, index=False)
        summary_path_tag = os.path.join(exp_dir, f"ensemble_ranking{subset_suffix}.csv")
        summary_df.to_csv(summary_path_tag, index=False)
        oof_predictions.to_csv(
            os.path.join(exp_dir, f"oof_predictions{subset_suffix}.csv"), index=True
        )
        with open(
            os.path.join(exp_dir, f"fold_scores{subset_suffix}.json"), "w", encoding="utf-8"
        ) as f:
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
        print(
            f"\n---> Best combo for subset {subset_label}: {best_name} (OOF IPCW={best_score:.5f})"
        )
        training_report["best_ensemble"] = {
            "name": best_name,
            "oof_ipcw": float(best_score),
            "tau": TAU,
        }

        print("\n[STEP 4/4] Retraining models on full subset and saving...")
        for name, est in models.items():
            try:
                est.fit(X_subset, y_subset)
                out_path = os.path.join("models", f"model_{name}{subset_suffix}.joblib")
                joblib.dump(est, out_path)
                print(f"  Saved: {out_path}")
            except Exception as e:
                print(f"  [WARN] Failed to save {name} for subset {subset_label}: {e}")

        meta = {
            "best_combo": best_name,
            "best_oof_ipcw": float(best_score),
            "models": valid_models,
            "subset": subset_label,
        }
        with open(
            os.path.join("models", f"ensemble_meta{subset_suffix}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)

        overall_report["subsets"][subset_label] = training_report

    with open(
        os.path.join(exp_dir, "training_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(overall_report, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()