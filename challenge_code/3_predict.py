#!/usr/bin/env python3
"""
Interactive Ensemble Prediction: choose models and produce rank-ensemble submission
"""
import os
import glob
import json
import joblib
import pandas as pd
import numpy as np
from src.config import PREPROCESSING, EXPERIMENT
from src.modeling.profile_strategy import (
    get_profile_mode,
    route_inference_rows,
    align_features_for_subset,
)
from src.utils.experiment import (
    compute_tag_with_signature,
    ensure_experiment_dir,
    save_predictions,
)


def list_saved_models(models_dir: str = "models"):
    paths = sorted(glob.glob(os.path.join(models_dir, "model_*.joblib")))
    names = [
        os.path.splitext(os.path.basename(p))[0].replace("model_", "") for p in paths
    ]
    return names, paths


def model_subset_from_name(name: str) -> str:
    if name.endswith("_complete"):
        return "complete"
    if name.endswith("_molecular"):
        return "molecular"
    return "default"


def prompt_model_selection(names):
    print("\nAvailable models:")
    for i, n in enumerate(names, start=1):
        print(f"  [{i}] {n}")
    sel = input(
        "Enter model indices to ensemble (comma-separated, e.g., 1,3,5) or 'all': "
    ).strip()
    if sel.lower() == "all":
        return names
    idx = [int(s) for s in sel.split(",") if s.strip().isdigit()]
    chosen = [names[i - 1] for i in idx if 1 <= i <= len(names)]
    if not chosen:
        raise ValueError("No valid models selected.")
    return chosen


def predict_and_submit():
    print("=== INTERACTIVE ENSEMBLE PREDICTION ===")
    print("=" * 60)

    # Compute experiment tag and show key preprocessing settings
    tag, cfg_signature, _, _ = compute_tag_with_signature()
    exp_dir = ensure_experiment_dir(tag)
    print(f"Experiment tag: {tag} (config sig: {cfg_signature})")
    print("[REPORT] Preprocessing settings:")
    print(
        f"  - imputer: {PREPROCESSING.get('imputer')}\n"
        f"  - knn.n_neighbors: {PREPROCESSING.get('knn', {}).get('n_neighbors')}\n"
        f"  - clip_quantiles: {PREPROCESSING.get('clip_quantiles')}\n"
        f"  - numeric_scaler: {PREPROCESSING.get('numeric_scaler')}"
    )

    # 1) Load processed test data
    print("\n1) Loading processed test data...")
    X_test_path = "datasets_processed/X_test_processed.csv"
    raw_test = pd.read_csv(X_test_path)
    ids = raw_test["ID"].copy()
    feature_df = raw_test.drop(columns=["ID", "CENTER_GROUP"], errors="ignore")

    strategy_mode = get_profile_mode()
    print(f"[INFO] Profile strategy mode: {strategy_mode}")

    subset_frames = {}
    if strategy_mode == "dual_model":
        masks = route_inference_rows(feature_df)
        for label in ["complete", "molecular"]:
            mask = masks.get(label)
            if mask is None or not mask.any():
                continue
            subset_frames[label] = align_features_for_subset(feature_df.loc[mask], label)
        if not subset_frames:
            raise ValueError(
                "Dual-model inference requires at least one routable subset with recorded features."
            )
    else:
        subset_frames["default"] = align_features_for_subset(feature_df, "default")

    required_subsets = set(subset_frames.keys())

    # 2) List and choose models
    names, paths = list_saved_models()
    if not names:
        print("No saved base models found in 'models/'. Please run training first.")
        return
    if strategy_mode == "dual_model":
        names = [n for n in names if model_subset_from_name(n) in required_subsets]
        if not names:
            print(
                "No subset-specific models detected. Ensure dual-model training has been executed."
            )
            return
    try:
        chosen = prompt_model_selection(names)
    except Exception as e:
        print(f"Selection error: {e}")
        return
    print(f"Chosen models: {chosen}")

    subset_model_map = {}
    for nm in chosen:
        subset = model_subset_from_name(nm)
        subset_model_map.setdefault(subset, []).append(nm)

    missing_subsets = [lbl for lbl in required_subsets if lbl not in subset_model_map]
    if missing_subsets:
        print(
            f"Selection error: missing models for subsets {missing_subsets}. Choose at least one model per active subset."
        )
        return

    # 3) Load chosen models and predict
    if strategy_mode == "dual_model":
        final_scores = pd.Series(index=feature_df.index, dtype=float)
        for subset_label, subset_df in subset_frames.items():
            models_for_subset = subset_model_map.get(subset_label, [])
            subset_preds = []
            for nm in models_for_subset:
                p = os.path.join("models", f"model_{nm}.joblib")
                est = joblib.load(p)
                pred = pd.Series(est.predict(subset_df), index=subset_df.index)
                subset_preds.append(pred.rank(method="average"))
            if not subset_preds:
                raise ValueError(
                    f"No models selected for subset '{subset_label}' while samples are present."
                )
            subset_stack = np.vstack([pred.values for pred in subset_preds])
            subset_scores = subset_stack.mean(axis=0)
            final_scores.loc[subset_df.index] = subset_scores
        if final_scores.isna().any():
            raise RuntimeError("Missing predictions for some samples in dual-model mode.")
        ensemble_score = final_scores.loc[feature_df.index].values
    else:
        preds = []
        default_df = subset_frames["default"]
        for nm in subset_model_map.get("default", []):
            p = os.path.join("models", f"model_{nm}.joblib")
            est = joblib.load(p)
            pred = pd.Series(est.predict(default_df), index=default_df.index)
            preds.append(pred.rank(method="average").values)
        if not preds:
            raise ValueError("No models selected for default subset.")
        ensemble_score = np.mean(np.vstack(preds), axis=0)
    print(f"Generated predictions for {len(ensemble_score)} samples.")

    # 4) Save submission with model names in filename and in experiment dir
    os.makedirs("submissions", exist_ok=True)
    model_tag = "+".join(chosen)[:120]
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    df_pred = pd.DataFrame({"ID": ids, "risk_score": ensemble_score})
    fname = f"submissions/submission_{model_tag}_{ts}.csv"
    df_pred.to_csv(fname, index=False)
    df_pred.to_csv("submissions/latest_submission.csv", index=False)

    # Save a copy under the experiment directory and chosen model list
    exp_pred_name = f"predictions_{model_tag}_{ts}.csv"
    save_predictions(tag, df_pred, filename=exp_pred_name)
    all_models_str = "+".join(chosen)
    with open(
        os.path.join(exp_dir, f"{all_models_str}_models.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"chosen_models": chosen, "timestamp": ts}, f, indent=2)

    print(f"Saved: {fname}")
    print(f"Also saved under experiment: {os.path.join(exp_dir, exp_pred_name)}")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    predict_and_submit()
