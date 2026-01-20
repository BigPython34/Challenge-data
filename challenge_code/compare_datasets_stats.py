"""
Script de comparaison statistique des datasets (Train, Test, Beat AML, TCGA).
Compare les taux de valeurs manquantes et les moyennes des variables cliniques.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_PATHS, BEAT_AML_PATHS, TCGA_PATHS

def load_data():
    print("Chargement des données...")
    
    # 1. Train
    train_df = pd.read_csv(DATA_PATHS["input_clinical_train"])
    print(f"Train: {train_df.shape}")
    
    # 2. Test
    test_df = pd.read_csv(DATA_PATHS["input_clinical_test"])
    print(f"Test: {test_df.shape}")
    
    # 3. Beat AML
    beat_aml_df = pd.read_csv(BEAT_AML_PATHS["clinical"])
    print(f"Beat AML: {beat_aml_df.shape}")
    
    # 4. TCGA
    tcga_df = pd.read_csv(TCGA_PATHS["clinical"])
    print(f"TCGA: {tcga_df.shape}")
    
    return {
        "Train": train_df,
        "Test": test_df,
        "Beat AML": beat_aml_df,
        "TCGA": tcga_df
    }

def compare_missing_values(datasets):
    print("\n" + "="*60)
    print("COMPARAISON DES VALEURS MANQUANTES (%)")
    print("="*60)
    
    # Identify all columns present in at least one dataset
    all_cols = set()
    for df in datasets.values():
        all_cols.update(df.columns)
    
    # Filter relevant clinical columns
    relevant_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "CYTOGENETICS"]
    cols_to_show = [c for c in relevant_cols if c in all_cols]
    
    results = {}
    for name, df in datasets.items():
        missing_pct = df[cols_to_show].isnull().mean() * 100
        results[name] = missing_pct
        
    comparison_df = pd.DataFrame(results)
    print(comparison_df.round(2))
    return comparison_df

def compare_means(datasets):
    print("\n" + "="*60)
    print("COMPARAISON DES MOYENNES (Variables Numériques)")
    print("="*60)
    
    numeric_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    
    results = {}
    for name, df in datasets.items():
        means = {}
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric just in case, coercing errors
                series = pd.to_numeric(df[col], errors='coerce')
                means[col] = series.mean()
            else:
                means[col] = np.nan
        results[name] = means
        
    comparison_df = pd.DataFrame(results)
    print(comparison_df.round(2))
    return comparison_df

def compare_medians(datasets):
    print("\n" + "="*60)
    print("COMPARAISON DES MÉDIANES (Variables Numériques)")
    print("="*60)
    
    numeric_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    
    results = {}
    for name, df in datasets.items():
        medians = {}
        for col in numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce')
                medians[col] = series.median()
            else:
                medians[col] = np.nan
        results[name] = medians
        
    comparison_df = pd.DataFrame(results)
    print(comparison_df.round(2))
    return comparison_df

def main():
    datasets = load_data()
    
    compare_missing_values(datasets)
    compare_means(datasets)
    compare_medians(datasets)

if __name__ == "__main__":
    main()
