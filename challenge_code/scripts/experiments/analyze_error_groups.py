#!/usr/bin/env python3
"""
Analyze Error Groups: "Veinards" (Lucky) vs "Malchanceux" (Unlucky)

This script identifies patients where the model made large systematic errors
and analyzes which features distinguish them from the rest of the population.

Usage:
    python analyze_error_groups.py [experiment_dir]
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Configuration
LUCKY_THRESHOLD = 0.3   # Error > 0.3 (Predicted High Risk, Lived Long)
UNLUCKY_THRESHOLD = -0.3 # Error < -0.3 (Predicted Low Risk, Died Soon)
MIN_GROUP_SIZE = 5

def analyze_group_features(df, group_mask, group_name, rest_mask):
    """
    Compare features between a specific group and the rest of the population.
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING GROUP: {group_name}")
    print(f"Size: {group_mask.sum()} patients")
    print(f"{'='*80}")
    
    if group_mask.sum() < MIN_GROUP_SIZE:
        print("Group too small for analysis.")
        return

    group_df = df[group_mask]
    rest_df = df[rest_mask]
    
    results = []
    
    # Identify feature columns (exclude metadata)
    feature_cols = [c for c in df.columns if c not in [
        "ID", "Time", "Event", "Truth_Rank", "Consensus_Risk_Rank", 
        "Model_Disagreement", "Mean_Error", "Mean_Abs_Error",
        "Rank_RSF", "Rank_GradientBoosting", "Rank_ExtraTrees",
        "Error_RSF", "Error_GradientBoosting", "Error_ExtraTrees"
    ]]
    
    for col in feature_cols:
        # Skip if column not in df (might have been dropped)
        if col not in df.columns:
            continue
            
        # Check if numeric or binary
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if not is_numeric:
            continue
            
        val_group = group_df[col].dropna()
        val_rest = rest_df[col].dropna()
        
        if len(val_group) == 0 or len(val_rest) == 0:
            continue
            
        # Check if binary (0/1)
        unique_vals = df[col].dropna().unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        
        if is_binary:
            # Difference in proportions
            prop_group = val_group.mean()
            prop_rest = val_rest.mean()
            diff = prop_group - prop_rest
            
            # Simple score: absolute difference weighted by prevalence (to avoid rare noise)
            # But raw difference is often most interpretable
            score = abs(diff)
            
            results.append({
                "Feature": col,
                "Type": "Binary",
                "Group_Mean": prop_group,
                "Rest_Mean": prop_rest,
                "Diff": diff,
                "Score": score
            })
            
        else:
            # Difference in means (standardized)
            mean_group = val_group.mean()
            mean_rest = val_rest.mean()
            std_rest = val_rest.std()
            
            if std_rest == 0:
                continue
                
            diff = mean_group - mean_rest
            cohen_d = diff / std_rest
            
            results.append({
                "Feature": col,
                "Type": "Numeric",
                "Group_Mean": mean_group,
                "Rest_Mean": mean_rest,
                "Diff": diff,
                "Score": abs(cohen_d)
            })
            
    # Convert to DataFrame and sort
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No features analyzed.")
        return

    res_df = res_df.sort_values("Score", ascending=False)
    
    print("\n--- TOP 15 DISTINCTIVE FEATURES ---")
    print(res_df[["Feature", "Type", "Group_Mean", "Rest_Mean", "Diff"]].head(15).to_string(index=False))
    
    # Suggest Interactions
    print("\n--- SUGGESTED INTERACTIONS ---")
    top_features = res_df.head(5)["Feature"].tolist()
    print(f"Consider combining: {', '.join(top_features)}")

def main():
    # Default path
    default_exp = "results/experiments/archive/251211-2"
    
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        exp_dir = default_exp
        
    print(f"Loading experiment: {exp_dir}")
    
    error_path = os.path.join(exp_dir, "error_analysis_detailed.csv")
    if not os.path.exists(error_path):
        print(f"Error: File not found: {error_path}")
        return

    # Load Errors
    errors_df = pd.read_csv(error_path)
    if "ID" not in errors_df.columns:
        # Try to infer ID from index if not present (older version)
        # But we fixed this, so it should be there.
        pass
        
    # Load Features
    features_path = "datasets_processed/X_train_processed.csv"
    print(f"Loading features: {features_path}")
    features_df = pd.read_csv(features_path)
    
    # Merge
    # Ensure ID is string in both
    errors_df["ID"] = errors_df["ID"].astype(str)
    features_df["ID"] = features_df["ID"].astype(str)
    
    full_df = pd.merge(errors_df, features_df, on="ID", how="inner")
    print(f"Merged dataset shape: {full_df.shape}")
    
    # Define Groups
    # Lucky: Error > Threshold (Positive Error = Pessimistic Model = Lived Longer)
    lucky_mask = full_df["Mean_Error"] > LUCKY_THRESHOLD
    
    # Unlucky: Error < -Threshold (Negative Error = Optimistic Model = Died Sooner)
    unlucky_mask = full_df["Mean_Error"] < UNLUCKY_THRESHOLD
    
    # Rest: Everyone else (or just those with small errors)
    # Let's compare against "Normal" errors (between -0.1 and 0.1) to be sharper?
    # Or just against "Not Group".
    # Comparing against "Rest of population" is standard.
    rest_mask_lucky = ~lucky_mask
    rest_mask_unlucky = ~unlucky_mask
    
    # Analyze Lucky (Veinards)
    analyze_group_features(full_df, lucky_mask, "LUCKY (Veinards) - Lived Longer than Predicted", rest_mask_lucky)
    
    # Analyze Unlucky (Malchanceux)
    analyze_group_features(full_df, unlucky_mask, "UNLUCKY (Malchanceux) - Died Sooner than Predicted", rest_mask_unlucky)

if __name__ == "__main__":
    main()
