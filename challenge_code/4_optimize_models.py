#!/usr/bin/env python3
"""
Script 4/x: Hyperparameter Optimization
Uses a robust and reusable orchestrator to find the best hyperparameters
for the most promising survival models.
"""
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

# Import your project's functions and classes
from src.modeling.train import load_training_dataset_csv
from src.modeling.optimizer import OptimizationOrchestrator


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
    # --- 2. DEFINE SEARCH SPACES ---
    # Define here which parameters to test for each model

    rsf_search_space = {
        "n_estimators": ("int", [100, 1000]),
        "max_depth": ("int", [3, 40]),
        "min_samples_split": ("int", [3, 50]),
        "min_samples_leaf": ("int", [2, 50]),
        "max_features": ("categorical", ["sqrt", 0.3, 0.5, 0.7]),
    }

    gb_search_space = {
        "n_estimators": ("int", [100, 1300]),
        "learning_rate": ("float", [0.005, 0.01, 0.2, True]),  # True for log scale
        "max_depth": ("int", [2, 13]),
        "subsample": ("float", [0.2, 0.7, 1.0]),
        "min_samples_leaf": ("int", [5, 50]),
    }

    # --- 3. RUN OPTIMIZATIONS ---
    # You can choose which model(s) to optimize.

    print("\n[STEP 2/3] Optimizing Random Survival Forest...")
    rsf_optimizer = OptimizationOrchestrator(
        model_name="RSF",
        model_class=RandomSurvivalForest,
        search_space=rsf_search_space,
        X=X,
        y=y,
        n_jobs=-1,  # Use all available CPU cores
    )
    best_rsf_params = rsf_optimizer.run(n_trials=300)  # Run for 300 trials

    print("\n[STEP 3/3] Optimizing Gradient Boosting Survival...")
    gb_optimizer = OptimizationOrchestrator(
        model_name="GradientBoosting",
        model_class=GradientBoostingSurvivalAnalysis,
        search_space=gb_search_space,
        X=X,
        y=y,
        n_jobs=-1,
    )
    best_gb_params = gb_optimizer.run(n_trials=400)  # Maybe more trials for this one

    print("\n" + "=" * 80)
    print("OPTIMIZATION SCRIPT FINISHED!")
    print("\nBest parameters found for RSF:", best_rsf_params)
    print("Best parameters found for Gradient Boosting:", best_gb_params)
    print("\nYou can now use these parameters to train your final model.")
    print("=" * 80)


if __name__ == "__main__":
    main()
