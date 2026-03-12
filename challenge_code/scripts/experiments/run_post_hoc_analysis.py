#!/usr/bin/env python3
"""
Post-Hoc Error Analysis Script

This script allows running the error analysis on an existing experiment
without re-running the full training pipeline.

It requires:
1. An existing experiment directory with `oof_predictions.csv`.
2. The processed training data (`datasets_processed/y_train_processed.csv`).

Usage:
    python run_post_hoc_analysis.py [experiment_dir]

If experiment_dir is not provided, it defaults to the most recent experiment.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from src.modeling.train import load_training_dataset_csv
from src.modeling.error_analysis import analyze_cv_errors

def get_latest_experiment_dir(base_dir="results/experiments"):
    """Finds the most recently modified experiment directory."""
    if not os.path.exists(base_dir):
        return None
    
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    
    # Sort by modification time
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir

def main():
    # 1. Determine Experiment Directory
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        print("No experiment directory provided. Searching for latest...")
        exp_dir = get_latest_experiment_dir()
    
    if not exp_dir or not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    print(f"Analyzing experiment: {exp_dir}")
    
    # 2. Check for OOF Predictions
    oof_path = os.path.join(exp_dir, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        print(f"Error: 'oof_predictions.csv' not found in {exp_dir}.")
        print("This analysis requires Out-of-Fold predictions generated during training.")
        print("Please re-run '2_c_find_best_ensemble.py' to generate them.")
        sys.exit(1)
        
    print(f"Loading OOF predictions from: {oof_path}")
    oof_predictions = pd.read_csv(oof_path, index_col=0)
    
    # 3. Load Ground Truth (y_train) and IDs
    print("Loading training data...")
    try:
        # Load raw processed data to get IDs
        X_raw = pd.read_csv("datasets_processed/X_train_processed.csv")
        if "ID" in X_raw.columns:
            patient_ids = X_raw["ID"].values
        else:
            print("Warning: 'ID' column not found in X_train_processed.csv. Using default index.")
            patient_ids = None

        # We only need y_train, but the function returns X_train too
        _, y_train = load_training_dataset_csv(
            X_train_path="datasets_processed/X_train_processed.csv",
            y_train_path="datasets_processed/y_train_processed.csv"
        )
        
        # Assign IDs to OOF predictions if available and lengths match
        if patient_ids is not None:
            if len(patient_ids) == len(oof_predictions):
                oof_predictions.index = patient_ids
                oof_predictions.index.name = "ID"
                print("Successfully assigned patient IDs to OOF predictions.")
            else:
                print(f"Warning: Length mismatch. IDs: {len(patient_ids)}, OOF: {len(oof_predictions)}")

    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)
        
    # 4. Run Analysis
    print("Running Error Analysis...")
    try:
        analyze_cv_errors(oof_predictions, y_train, exp_dir)
        print("\nAnalysis complete!")
        print(f"Reports saved to: {exp_dir}")
        print(f" - {os.path.join(exp_dir, 'error_analysis_summary.csv')}")
        print(f" - {os.path.join(exp_dir, 'hardest_samples.csv')}")
        print(f" - {os.path.join(exp_dir, 'controversial_samples.csv')}")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
