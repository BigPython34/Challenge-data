#!/usr/bin/env python3
"""Generate a rank-ensemble submission from cached base models."""

import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.config import PREPROCESSING
from src.utils.experiment import (
    compute_tag_with_signature,
    ensure_experiment_dir,
    save_predictions,
)


def list_saved_models(models_dir: str = "models") -> tuple[list[str], list[str]]:
    paths = sorted(glob.glob(os.path.join(models_dir, "model_*.joblib")))
    names = [os.path.splitext(os.path.basename(p))[0].replace("model_", "") for p in paths]
    return names, paths


def prompt_model_selection(names: list[str]) -> list[str]:
    print("\nAvailable models:")
    for i, n in enumerate(names, start=1):
        print(f"  [{i}] {n}")
    sel = input("Enter model indices (comma-separated) or 'all': ").strip()
    if sel.lower() == "all":
        return names.copy()
    idx = [int(s) for s in sel.split(",") if s.strip().isdigit()]
    chosen = [names[i - 1] for i in idx if 1 <= i <= len(names)]
    if not chosen:
        raise ValueError("No valid models selected.")
    return chosen


def resolve_models_from_arg(raw: str, names: list[str]) -> list[str]:
    text = raw.strip()
    if not text:
        raise ValueError("Empty --models argument.")
    if text.lower() == "all":
        return names.copy()

    requested = [item.strip() for item in text.split(",") if item.strip()]
    chosen = [name for name in requested if name in names]
    if not chosen:
        raise ValueError("No valid models requested via CLI.")
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build submission from saved models.")
    parser.add_argument(
        "--models",
        "-m",
        help="Comma-separated list of model names to include (use 'all' for everything).",
    )
    return parser.parse_args()


def predict_and_submit() -> None:
    print("=== ENSEMBLE PREDICTION ===")
    print("=" * 60)

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

    print("\n1) Loading processed test data...")
    X_test = pd.read_csv("datasets_processed/X_test_processed.csv")
    ids = X_test.get("ID")
    if ids is None:
        raise ValueError("X_test_processed.csv must contain an ID column.")
    ids = ids.copy()
    for column in ["ID", "CENTER_GROUP"]:
        if column in X_test.columns:
            X_test = X_test.drop(columns=[column])

    names, _ = list_saved_models()
    if not names:
        print("No saved base models found in 'models/'. Please run training first.")
        return

    args = parse_args()
    try:
        chosen = resolve_models_from_arg(args.models, names) if args.models else prompt_model_selection(names)
    except Exception as exc:
        print(f"Selection error: {exc}")
        return

    print(f"Chosen models: {chosen}")

    preds = []
    for model_name in chosen:
        path = os.path.join("models", f"model_{model_name}.joblib")
        estimator = joblib.load(path)
        preds.append(pd.Series(estimator.predict(X_test)).rank(method="average").values)

    ensemble_score = np.mean(np.vstack(preds), axis=0)
    print(f"Generated predictions for {len(ensemble_score)} samples.")

    os.makedirs("submissions", exist_ok=True)
    model_tag = "+".join(chosen)[:120]
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({"ID": ids, "risk_score": ensemble_score})
    filename = f"submissions/submission_{model_tag}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    submission.to_csv("submissions/latest_submission.csv", index=False)

    save_predictions(tag, submission, filename=f"predictions_{model_tag}_{timestamp}.csv")
    models_meta = os.path.join(exp_dir, f"{model_tag}_models.json")
    with open(models_meta, "w", encoding="utf-8") as fh:
        json.dump({"chosen_models": chosen, "timestamp": timestamp}, fh, indent=2)

    print(f"Saved: {filename}")
    print(f"Also saved under experiment: {os.path.join(exp_dir, f'predictions_{model_tag}_{timestamp}.csv')}")
    print("=" * 60)
    print("DONE")


if __name__ == "__main__":
    predict_and_submit()
