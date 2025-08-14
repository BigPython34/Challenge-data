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
from src.utils.experiment import compute_tag, ensure_experiment_dir, save_predictions


def list_saved_models(models_dir: str = "models"):
    paths = sorted(glob.glob(os.path.join(models_dir, "model_*.joblib")))
    names = [
        os.path.splitext(os.path.basename(p))[0].replace("model_", "") for p in paths
    ]
    return names, paths


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
    cfg_slice = {"PREPROCESSING": PREPROCESSING, "EXPERIMENT": EXPERIMENT}
    tag = compute_tag(cfg_slice, prefix=EXPERIMENT.get("name"))
    exp_dir = ensure_experiment_dir(tag)
    print(f"Experiment tag: {tag}")
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
    X_test = pd.read_csv(X_test_path)
    ids = X_test["ID"].copy()
    # Drop non-features if present
    for c in ["ID", "CENTER_GROUP"]:
        if c in X_test.columns:
            X_test = X_test.drop(columns=[c])

    # 2) List and choose models
    names, paths = list_saved_models()
    if not names:
        print("No saved base models found in 'models/'. Please run training first.")
        return
    try:
        chosen = prompt_model_selection(names)
    except Exception as e:
        print(f"Selection error: {e}")
        return
    print(f"Chosen models: {chosen}")

    # 3) Load chosen models and predict
    preds = []
    for nm in chosen:
        p = os.path.join("models", f"model_{nm}.joblib")
        est = joblib.load(p)
        pred = est.predict(X_test)
        preds.append(pd.Series(pred).rank(method="average").values)

    # Average ranks for ensemble score
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
    with open(os.path.join(exp_dir, "chosen_models.json"), "w", encoding="utf-8") as f:
        json.dump({"chosen_models": chosen, "timestamp": ts}, f, indent=2)

    print(f"Saved: {fname}")
    print(f"Also saved under experiment: {os.path.join(exp_dir, exp_pred_name)}")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    predict_and_submit()
