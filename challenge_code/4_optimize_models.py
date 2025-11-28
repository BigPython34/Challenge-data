#!/usr/bin/env python3
"""
Script 4/x: Hyperparameter Optimization
"""

import pandas as pd
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
    ExtraSurvivalTrees,
)

from src.config import HYPERPARAM_OPTIMIZATION
from src.modeling.train import load_training_dataset_csv
from src.modeling.optimizer import OptimizationOrchestrator


MODEL_CLASS_REGISTRY = {
    "RSF": RandomSurvivalForest,
    "GradientBoosting": GradientBoostingSurvivalAnalysis,
    "ExtraTrees": ExtraSurvivalTrees,
}


def main():
    print("=" * 80)
    print("SCRIPT 4: HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    print("\n[STEP 1/3] Loading dataset...")
    X, y = load_training_dataset_csv(
        X_train_path="datasets_processed/X_train_processed.csv",
        y_train_path="datasets_processed/y_train_processed.csv",
    )

    if "CENTER_GROUP" in X.columns:
        X = X.drop(columns=["CENTER_GROUP"])

    cfg = HYPERPARAM_OPTIMIZATION.get("models", {})
    enabled_models = [(k, v) for k, v in cfg.items() if v.get("enabled", False)]

    if not enabled_models:
        print("Aucun modèle activé.")
        return

    best_params = {}

    for model_name, model_cfg in enabled_models:
        print(f"\n⚙ Optimizing {model_name}...")
        model_class = MODEL_CLASS_REGISTRY.get(model_name)

        if model_class is None:
            print(f"[WARN] Unknown model: {model_name}")
            continue

        search_space = model_cfg.get("search_space")
        if not search_space:
            print(f"[WARN] No search space for {model_name}, skipping.")
            continue

        optimizer = OptimizationOrchestrator(
            model_name=model_name,
            model_class=model_class,
            search_space=search_space,
            X=X,
            y=y,
            n_splits=model_cfg.get("n_splits", 5),
            study_n_jobs=model_cfg.get("optuna_workers", 1),
            model_n_jobs=model_cfg.get("model_n_jobs", 1),
            timeout_per_trial=120,  # Azure VM friendly
        )

        trials = model_cfg.get("n_trials", 50)
        best_params[model_name] = optimizer.run(n_trials=trials)

    print("\n=== OPTIMIZATION DONE ===")
    for model_name, params in best_params.items():
        print(f"{model_name} → {params}")


if __name__ == "__main__":
    main()
