#!/usr/bin/env python3
"""
Script 4/x: Hyperparameter Optimization
Uses a robust and reusable orchestrator to find the best hyperparameters
for the most promising survival models.
"""
import pandas as pd
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
    ExtraSurvivalTrees,
)

# Import your project's functions and classes
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
    print(" SCRIPT 4: HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    # --- 1. LOAD DATASET ---
    print("\n[STEP 1/3] Loading the prepared dataset...")
    try:
        X, y = load_training_dataset_csv(
            X_train_path="datasets_processed/X_train_processed.csv",
            y_train_path="datasets_processed/y_train_processed.csv",
        )

    except FileNotFoundError as e:
        print(f"   [ERROR] {e}")
        return
    if "CENTER_GROUP" in X.columns:
        X = X.drop(columns=["CENTER_GROUP"])

    optimization_cfg = HYPERPARAM_OPTIMIZATION.get("models", {})
    enabled_models = [
        (name, cfg)
        for name, cfg in optimization_cfg.items()
        if cfg.get("enabled", False)
    ]

    if not enabled_models:
        print("\n[INFO] Aucun modèle configuré pour l'optimisation. Rien à faire.")
        return

    total_steps = 1 + len(enabled_models)
    best_params = {}

    for idx, (model_name, model_cfg) in enumerate(enabled_models, start=2):
        print(f"\n[STEP {idx}/{total_steps}] Optimizing {model_name}...")
        model_class = MODEL_CLASS_REGISTRY.get(model_name)
        if model_class is None:
            print(
                f"   [WARN] Aucun modèle sksurv enregistré pour '{model_name}'. Étape ignorée."
            )
            continue
        search_space = model_cfg.get("search_space")
        if not search_space:
            print(
                f"   [WARN] Aucun espace de recherche défini pour '{model_name}'. Étape ignorée."
            )
            continue

        optimizer = OptimizationOrchestrator(
            model_name=model_name,
            model_class=model_class,
            search_space=search_space,
            X=X,
            y=y,
            n_jobs=model_cfg.get("n_jobs", -1),
        )
        trials = model_cfg.get("n_trials", 100)
        best_params[model_name] = optimizer.run(n_trials=trials)

    print("\n" + "=" * 80)
    print("OPTIMIZATION SCRIPT FINISHED!")
    if not best_params:
        print("Aucun paramètre n'a été trouvé (toutes les étapes ont été ignorées).")
    else:
        for model_name, params in best_params.items():
            print(f"\nBest parameters found for {model_name}:", params)
    print("\nYou can now use these parameters to train your final model.")
    print("=" * 80)


if __name__ == "__main__":
    main()
