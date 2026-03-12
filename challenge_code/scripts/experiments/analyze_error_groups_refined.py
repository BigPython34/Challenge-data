#!/usr/bin/env python3
"""
Refined Error Group Analysis: Finding Specific Drivers behind the Counts

This script digs deeper into the "Lucky" (High Burden) and "Unlucky" (Low Burden) groups
to find specific binary mutations or clinical features that distinguish them.
"""

import os
import sys
import pandas as pd
import numpy as np

# Configuration
LUCKY_THRESHOLD = 0.3
UNLUCKY_THRESHOLD = -0.3

def analyze_subgroup_drivers(df, target_mask, comparison_mask, group_name):
    """
    Compare binary/clinical features between a target subgroup and a comparison group.
    Ex: Compare "Lucky High Burden" vs "Dead High Burden".
    """
    print(f"\n{'='*80}")
    print(f"DEEP DIVE: {group_name}")
    print(f"Target Size: {target_mask.sum()} | Comparison Size: {comparison_mask.sum()}")
    print(f"{'='*80}")
    
    if target_mask.sum() < 5:
        print("Subgroup too small.")
        return

    target_df = df[target_mask]
    comp_df = df[comparison_mask]
    
    results = []
    
    # Exclude counts and metadata
    ignore_cols = [
        "ID", "Time", "Event", "Truth_Rank", "Consensus_Risk_Rank", 
        "Model_Disagreement", "Mean_Error", "Mean_Abs_Error",
        "Rank_RSF", "Rank_GradientBoosting", "Rank_ExtraTrees",
        "Error_RSF", "Error_GradientBoosting", "Error_ExtraTrees"
    ]
    
    for col in df.columns:
        if col in ignore_cols:
            continue
        if "count" in col.lower() or "score" in col.lower() or "total" in col.lower() or "num_" in col.lower():
            continue
            
        # Check type
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Binary Analysis
        unique_vals = df[col].dropna().unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        
        if is_binary:
            prop_target = target_df[col].mean()
            prop_comp = comp_df[col].mean()
            diff = prop_target - prop_comp
            
            # We care about features that are MORE frequent in the target group
            # or MUCH LESS frequent.
            if abs(diff) > 0.05: # 5% difference filter
                results.append({
                    "Feature": col,
                    "Type": "Binary",
                    "Target_Freq": prop_target,
                    "Comp_Freq": prop_comp,
                    "Diff": diff,
                    "Abs_Diff": abs(diff)
                })
        else:
            # Clinical Numeric (WBC, Age, etc)
            mean_target = target_df[col].mean()
            mean_comp = comp_df[col].mean()
            std_comp = comp_df[col].std()
            
            if std_comp == 0: continue
            
            diff = mean_target - mean_comp
            cohen_d = diff / std_comp
            
            if abs(cohen_d) > 0.2: # Effect size filter
                results.append({
                    "Feature": col,
                    "Type": "Numeric",
                    "Target_Mean": mean_target,
                    "Comp_Mean": mean_comp,
                    "Diff": diff,
                    "Abs_Diff": abs(cohen_d) # Use Cohen's d for sorting
                })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No distinctive features found.")
        return

    res_df = res_df.sort_values("Abs_Diff", ascending=False)
    print(res_df.head(20).to_string(index=False))

def main():
    # Load Data
    exp_dir = "results/experiments/archive/251211-2"
    error_path = os.path.join(exp_dir, "error_analysis_detailed.csv")
    features_path = "datasets_processed/X_train_processed.csv"
    
    if not os.path.exists(error_path):
        print("Error file not found.")
        return
        
    errors_df = pd.read_csv(error_path)
    features_df = pd.read_csv(features_path)
    
    # Merge
    errors_df["ID"] = errors_df["ID"].astype(str)
    features_df["ID"] = features_df["ID"].astype(str)
    full_df = pd.merge(errors_df, features_df, on="ID", how="inner")
    
    # Define Groups
    lucky_mask = full_df["Mean_Error"] > LUCKY_THRESHOLD
    unlucky_mask = full_df["Mean_Error"] < UNLUCKY_THRESHOLD
    
    # 1. Analyze "Lucky" (High Burden Survivors)
    # We compare them to "Normal" patients who ALSO have High Burden but died as expected (or had normal error).
    # Let's define "High Burden" as top 25% of cosmic_chromosome_count
    burden_col = "cosmic_chromosome_count"
    high_burden_threshold = full_df[burden_col].quantile(0.75)
    
    high_burden_mask = full_df[burden_col] >= high_burden_threshold
    
    # Target: Lucky AND High Burden
    lucky_high_burden = lucky_mask & high_burden_mask
    # Comparison: NOT Lucky AND High Burden (The ones who died or were predicted correctly as high risk)
    normal_high_burden = (~lucky_mask) & high_burden_mask
    
    analyze_subgroup_drivers(full_df, lucky_high_burden, normal_high_burden, "LUCKY vs NORMAL (within High Burden Population)")
    
    # 2. Analyze "Unlucky" (Low Burden Deaths)
    # Compare to "Normal" patients with Low Burden (who survived or were predicted correctly)
    low_burden_threshold = full_df[burden_col].quantile(0.25)
    low_burden_mask = full_df[burden_col] <= low_burden_threshold
    
    # Target: Unlucky AND Low Burden
    unlucky_low_burden = unlucky_mask & low_burden_mask
    # Comparison: NOT Unlucky AND Low Burden
    normal_low_burden = (~unlucky_mask) & low_burden_mask
    
    analyze_subgroup_drivers(full_df, unlucky_low_burden, normal_low_burden, "UNLUCKY vs NORMAL (within Low Burden Population)")

if __name__ == "__main__":
    main()
