"""
Pipeline de préparation des données pour l'analyse de survie AML.

Ce script orchestre:
1. Chargement et nettoyage des données
2. Création des features auxiliaires (avant imputation)
3. Imputation précoce des colonnes cliniques
4. Feature Engineering complet
5. Prétraitement final (scaling, encoding)
6. Sauvegarde des datasets

Refactoré le 2025-12-04 pour:
- Utiliser les colonnes auxiliaires AVANT le FE pour guider l'imputation
- Centraliser la logique d'imputation dans src/data/imputation.py
- Supprimer le code mort et simplifier le flow
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.config import (
    PREPROCESSING,
    EXPERIMENT,
    CLINICAL_RANGES,
    MOLECULAR_EXTERNAL_SCORES,
    RARE_EVENT_PRUNING_THRESHOLD,
    DATA_PATHS,
    DATA_FUSION,
    BEAT_AML_PATHS,
    TCGA_PATHS,
    ID_COLUMNS,
    TARGET_COLUMNS,
    CENTER_GROUPING,
    FEATURE_ENGINEERING_TOGGLES,
    REDUNDANCY_POLICY,
    MODEL_DIR,
    FLOAT32_POLICY,
    CORE_FEATURES,
    EXPLORATORY_FEATURES,
    FEATURE_SET_POLICY,
)

from src.data.data_extraction.external_data_manager import ExternalDataManager
from src.data.data_cleaning.cleaner import clean_and_validate_data
from src.data.auxiliary_features import (
    create_auxiliary_features_for_imputation,
    merge_auxiliary_features,
    inject_auxiliary_features_for_pipeline,
)
from src.data.imputation import apply_early_imputation
from src.data.features.feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
    MolecularFeatureExtraction,
    CytoMolecularInteractionFeatures,
)
from src.data.features.pruning import (
    prune_highly_correlated_features_pair,
    prune_rare_binary_features,
)
from src.modeling.pipeline_components import get_preprocessing_pipeline
from src.utils.experiment import (
    save_manifest,
    save_feature_list,
    ensure_experiment_dir,
    compute_tag_with_signature,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_float32_policy(
    df: pd.DataFrame,
    *,
    candidates: list[str] | None = None,
    auto_detect: bool = False,
    context: str = "",
) -> pd.DataFrame:
    """Cast selected float columns to float32 according to the global policy."""
    policy = FLOAT32_POLICY or {}
    if not policy.get("enabled", False):
        return df

    protected = set(policy.get("protected_columns", []))
    protected.update(filter(None, [
        ID_COLUMNS.get("patient"), ID_COLUMNS.get("center"),
        TARGET_COLUMNS.get("status"), TARGET_COLUMNS.get("time"), "CENTER_GROUP",
    ]))

    columns_to_cast: list[str] = []
    if candidates:
        columns_to_cast.extend([c for c in candidates if c in df.columns and c not in protected])
    if auto_detect:
        float_cols = df.select_dtypes(include=["float64", "Float64"]).columns
        columns_to_cast.extend([c for c in float_cols if c not in protected])

    # Deduplicate preserving order
    seen = set()
    ordered = [c for c in columns_to_cast if not (c in seen or seen.add(c))]

    if ordered:
        df[ordered] = df[ordered].astype("float32")
        if context:
            print(f"[FLOAT32] Converted {len(ordered)} columns to float32 during {context}.")
    return df


def apply_feature_subset_mode(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter features based on FEATURE_SET_POLICY mode (core/exploratory/default)."""
    cfg = FEATURE_SET_POLICY or {}
    mode = (cfg.get("mode") or "default").lower()
    
    if mode in {"default", "normal", "all", "full"}:
        return train_df, test_df

    requested = CORE_FEATURES if mode == "core" else EXPLORATORY_FEATURES if mode == "exploratory" else None
    if requested is None:
        print(f"[FEATURE SET] Unknown mode '{mode}', ignoring.")
        return train_df, test_df

    keep_cols = [ID_COLUMNS["patient"]]
    if "CENTER_GROUP" in train_df.columns:
        keep_cols.append("CENTER_GROUP")
    
    requested_set = set(requested)
    selected = [c for c in train_df.columns if c in requested_set or c in keep_cols]

    if len(selected) <= len(keep_cols):
        print("[FEATURE SET] No overlap with requested features, ignoring.")
        return train_df, test_df

    missing = [c for c in requested if c not in train_df.columns]
    if missing and cfg.get("warn_on_missing", True):
        print(f"[FEATURE SET] {len(missing)} requested features missing: {missing[:8]}...")

    kept_count = len([c for c in selected if c not in keep_cols])
    print(f"[FEATURE SET] Mode '{mode}' activated ({kept_count} features kept).")

    return train_df.loc[:, selected], test_df.reindex(columns=selected)


def load_external_cohort(
    cohort_name: str,
    paths: dict,
    existing_ids: set[str],
    clinical_cols: list[str],
    molecular_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame | None:
    """Load external cohort data (Beat AML, TCGA, etc.)."""
    cfg = DATA_FUSION.get(cohort_name, {})
    use_for_training = cfg.get("use_for_training", False)
    use_for_imputation = cfg.get("use_for_imputation", False)
    
    if not (use_for_training or use_for_imputation):
        return None

    required = {
        "clinical": paths.get("clinical"),
        "molecular": paths.get("molecular"),
        "target": paths.get("target"),
    }
    missing = [k for k, v in required.items() if not v or not Path(v).is_file()]
    if missing:
        print(f"[{cohort_name.upper()}] Cannot load: missing files {missing}")
        return None

    try:
        clinical_df = pd.read_csv(required["clinical"])
        molecular_df = pd.read_csv(required["molecular"])
        target_df = pd.read_csv(required["target"])

        # Align columns
        clinical_df = clinical_df.reindex(columns=clinical_cols, fill_value=pd.NA)
        target_df = target_df.reindex(columns=target_cols, fill_value=pd.NA)

        # Clean
        clinical_clean, molecular_clean, target_clean = clean_and_validate_data(
            clinical_df, molecular_df, target_df
        )

        # Merge and filter existing IDs
        merged = pd.merge(clinical_clean, target_clean, on=ID_COLUMNS["patient"], how="inner")
        merged[ID_COLUMNS["patient"]] = merged[ID_COLUMNS["patient"]].astype(str)
        
        overlap = set(merged[ID_COLUMNS["patient"]]) & existing_ids
        if overlap:
            merged = merged[~merged[ID_COLUMNS["patient"]].isin(overlap)]

        if merged.empty:
            return None

        reason = "training" if use_for_training else "imputation"
        print(f"[{cohort_name.upper()}] Loaded {len(merged)} patients for {reason}.")
        return merged
    except Exception as e:
        print(f"[{cohort_name.upper()}] Error loading: {e}")
        return None


def run_feature_engineering(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    important_genes: list,
    data_manager: ExternalDataManager,
) -> pd.DataFrame:
    """Execute the complete feature engineering pipeline."""
    print("\n[FE] Starting Feature Engineering...")

    final_df = clinical_df.copy()
    final_df[ID_COLUMNS["patient"]] = final_df[ID_COLUMNS["patient"]].astype(str)
    if not molecular_df.empty:
        molecular_df[ID_COLUMNS["patient"]] = molecular_df[ID_COLUMNS["patient"]].astype(str)

    if FEATURE_ENGINEERING_TOGGLES.get("clinical", True):
        print("[FE] Creating clinical features...")
        final_df = ClinicalFeatureEngineering.create_clinical_features(final_df)

    if FEATURE_ENGINEERING_TOGGLES.get("cytogenetic", True):
        print("[FE] Creating cytogenetic features...")
        cyto_features = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
            final_df[[ID_COLUMNS["patient"], "CYTOGENETICS"]].copy()
        )
        final_df = pd.merge(final_df, cyto_features, on=ID_COLUMNS["patient"], how="left")

    if FEATURE_ENGINEERING_TOGGLES.get("molecular", True):
        print("[FE] Creating molecular features...")
        mol_features = MolecularFeatureExtraction.create_all_molecular_features(
            base_df=final_df[[ID_COLUMNS["patient"]]],
            maf_df=molecular_df,
            important_genes=important_genes,
            external_data_manager=data_manager,
        )
        if not mol_features.empty:
            final_df = pd.merge(final_df, mol_features, on=ID_COLUMNS["patient"], how="left")
            mol_cols = [c for c in mol_features.columns if c != ID_COLUMNS["patient"]]
            final_df[mol_cols] = final_df[mol_cols].fillna(0)

    final_df = final_df.drop(columns=["CYTOGENETICS"], errors="ignore")

    print(f"[FE] Feature Engineering complete. Shape: {final_df.shape}")
    missing_pct = final_df.isnull().sum().sum() / final_df.size * 100
    print(f"[FE] Residual missing rate: {missing_pct:.2f}%")

    if FEATURE_ENGINEERING_TOGGLES.get("cyto_molecular_interaction", True):
        final_df = CytoMolecularInteractionFeatures.create_interaction_features(final_df)

    return final_df


def apply_center_grouping(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group rare centers into 'CENTER_OTHER'."""
    if not CENTER_GROUPING.get("enabled", False):
        return train_df, test_df

    print("\n[PREP] Grouping rare centers...")
    threshold = CENTER_GROUPING.get("rare_center_threshold", 40)
    other_label = CENTER_GROUPING.get("other_label", "CENTER_OTHER")

    counts = train_df[ID_COLUMNS["center"]].value_counts()
    major = counts[counts >= threshold].index.tolist()

    print(f"   -> {len(major)} major centers, {len(counts) - len(major)} grouped as '{other_label}'")

    for df in [train_df, test_df]:
        df[ID_COLUMNS["center"]] = df[ID_COLUMNS["center"]].apply(
            lambda x: x if x in major else other_label
        )

    return train_df, test_df


def create_monocyte_indicator(df: pd.DataFrame) -> pd.Series | None:
    """Create MONOCYTES_missing indicator before imputation."""
    if "MONOCYTES" not in df.columns:
        return None
    mask = df["MONOCYTES"].isna().astype("int8")
    df["MONOCYTES_missing"] = mask
    return mask.copy()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Execute the full data preparation pipeline."""
    
    # --- Initialize ---
    data_manager = ExternalDataManager(
        cosmic_path=DATA_PATHS["cosmic_file"],
        oncokb_path=DATA_PATHS["oncokb_file"],
        clinvar_path=DATA_PATHS.get("clinvar_vcf"),
    )

    tag, cfg_signature, full_cfg_snapshot, cfg_slice = compute_tag_with_signature()
    save_manifest(tag, full_config=cfg_slice, extra={
        "preprocessing_report": {
            "imputer_strategy": PREPROCESSING.get("imputer"),
            "early_imputation": PREPROCESSING.get("early_imputation"),
            "single_imputation_mode": PREPROCESSING.get("single_imputation_mode"),
        },
        "clinical_ranges": CLINICAL_RANGES,
    })

    try:
        base = ensure_experiment_dir(tag)
        with open(os.path.join(base, "config_full.json"), "w", encoding="utf-8") as f:
            json.dump(full_cfg_snapshot, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[TRACE] Could not save config_full.json: {e}")

    print("[CONFIG] Imputation settings:")
    print(f"   - Early imputation: {PREPROCESSING.get('early_imputation', {}).get('enabled', False)}")
    print(f"   - Use auxiliary columns: {PREPROCESSING.get('early_imputation', {}).get('use_auxiliary_columns', False)}")
    print(f"   - Single imputation mode: {PREPROCESSING.get('single_imputation_mode', False)}")

    # ==========================================================================
    # STEP 1: LOAD AND CLEAN DATA
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: LOAD AND CLEAN DATA")
    print("=" * 60)

    clinical_train_raw = pd.read_csv(DATA_PATHS["input_clinical_train"])
    molecular_train_raw = pd.read_csv(DATA_PATHS["input_molecular_train"])
    target_train_raw = pd.read_csv(DATA_PATHS["input_target_train"])
    clinical_test_raw = pd.read_csv(DATA_PATHS["input_clinical_test"])
    molecular_test_raw = pd.read_csv(DATA_PATHS["input_molecular_test"])

    # Clean data
    clinical_train, molecular_train, target_train = clean_and_validate_data(
        clinical_train_raw, molecular_train_raw, target_train_raw
    )
    
    fake_target_test = pd.DataFrame({
        ID_COLUMNS["patient"]: clinical_test_raw[ID_COLUMNS["patient"]],
        TARGET_COLUMNS["time"]: 1,
        TARGET_COLUMNS["status"]: 0,
    })
    clinical_test, molecular_test, _ = clean_and_validate_data(
        clinical_test_raw, molecular_test_raw, fake_target_test
    )

    # Merge clinical + target
    train_df = pd.merge(clinical_train, target_train, on=ID_COLUMNS["patient"], how="inner")
    test_df = clinical_test.copy()

    # Apply float32 policy
    float32_candidates = FLOAT32_POLICY.get("columns") or PREPROCESSING.get("continuous_features", [])
    train_df = apply_float32_policy(train_df, candidates=float32_candidates, context="post-clean (train)")
    test_df = apply_float32_policy(test_df, candidates=float32_candidates, context="post-clean (test)")

    # Load External Data (Beat AML, TCGA, etc.)
    external_imputation_dfs = []
    external_training_dfs = []
    
    existing_ids = set(train_df[ID_COLUMNS["patient"]].astype(str))
    
    cohorts_to_load = [
        ("beat_aml", BEAT_AML_PATHS),
        ("tcga", TCGA_PATHS),
    ]
    
    for cohort_name, paths in cohorts_to_load:
        cohort_data = load_external_cohort(
            cohort_name=cohort_name,
            paths=paths,
            existing_ids=existing_ids,
            clinical_cols=clinical_train_raw.columns.tolist(),
            molecular_cols=molecular_train_raw.columns.tolist(),
            target_cols=target_train_raw.columns.tolist(),
        )
        
        if cohort_data is not None:
            cfg = DATA_FUSION.get(cohort_name, {})
            if cfg.get("use_for_imputation", False):
                external_imputation_dfs.append(cohort_data)
            if cfg.get("use_for_training", False):
                external_training_dfs.append(cohort_data)

    # Combine external datasets
    external_imputation_df = pd.concat(external_imputation_dfs, ignore_index=True) if external_imputation_dfs else None
    external_training_df = pd.concat(external_training_dfs, ignore_index=True) if external_training_dfs else None

    # Apply center grouping
    train_df, test_df = apply_center_grouping(train_df, test_df)

    # Create MONOCYTES_missing indicator BEFORE imputation
    mono_indicator_train = None
    mono_indicator_test = None
    if EXPERIMENT.get("keep_monocyte_indicator", False):
        print("\n[PREP] Creating MONOCYTES_missing indicator...")
        mono_indicator_train = create_monocyte_indicator(train_df)
        mono_indicator_test = create_monocyte_indicator(test_df)

    # Determine if early imputation is enabled first
    early_imputation_enabled = PREPROCESSING.get("early_imputation", {}).get("enabled", False)

    # ==========================================================================
    # STEP 2: CREATE AUXILIARY FEATURES (BEFORE IMPUTATION) - ONLY IF EARLY IMP.
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: CREATE AUXILIARY FEATURES FOR IMPUTATION")
    print("=" * 60)

    aux_cols_created = []
    if early_imputation_enabled and PREPROCESSING.get("imputer_auxiliary_features", {}).get("enabled", True):
        # Create basic mutation flags and cyto risk from raw data (only for early imputation)
        train_aux = create_auxiliary_features_for_imputation(train_df, molecular_train)
        test_aux = create_auxiliary_features_for_imputation(test_df, molecular_test)

        if not train_aux.empty:
            train_df = merge_auxiliary_features(train_df, train_aux)
            test_df = merge_auxiliary_features(test_df, test_aux)
            aux_cols_created = [c for c in train_aux.columns if c != ID_COLUMNS["patient"]]
            print(f"[AUX] {len(aux_cols_created)} auxiliary features created for early imputation")
    else:
        if not early_imputation_enabled:
            print("[AUX] Skipping auxiliary features (early_imputation disabled)")
        else:
            print("[AUX] Skipping auxiliary features (imputer_auxiliary_features disabled)")

    # ==========================================================================
    # STEP 3: EARLY IMPUTATION (CLINICAL COLUMNS) - OPTIONAL
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: EARLY IMPUTATION (CLINICAL COLUMNS)")
    print("=" * 60)
    
    if early_imputation_enabled:
        # Prepare fit data for early imputation
        extra_fit_df = None
        if external_imputation_df is not None:
            extra_fit_df = external_imputation_df.copy()
        
        train_df, test_df = apply_early_imputation(
            train_df, test_df,
            extra_fit_df=extra_fit_df,
            auxiliary_columns=aux_cols_created,
        )

        # Remove auxiliary columns after imputation (they will be recreated by FE)
        if aux_cols_created:
            train_df = train_df.drop(columns=aux_cols_created, errors="ignore")
            test_df = test_df.drop(columns=aux_cols_created, errors="ignore")
            print(f"[PREP] Removed {len(aux_cols_created)} auxiliary columns (will be recreated by FE).")
    else:
        print("[PREP] Early imputation DISABLED (using pipeline imputation instead)")

    # Restore MONOCYTES_missing indicator (may have been modified by imputation)
    if EXPERIMENT.get("keep_monocyte_indicator", False):
        if mono_indicator_train is not None:
            train_df["MONOCYTES_missing"] = mono_indicator_train
        if mono_indicator_test is not None:
            test_df["MONOCYTES_missing"] = mono_indicator_test

    train_df = apply_float32_policy(train_df, candidates=float32_candidates, context="after imputation (train)")
    test_df = apply_float32_policy(test_df, candidates=float32_candidates, context="after imputation (test)")

    # ==========================================================================
    # STEP 4: FEATURE ENGINEERING
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 60)

    # Get frequent genes
    important_genes = MolecularFeatureExtraction.get_frequent_genes(
        train_maf=molecular_train, test_maf=molecular_test
    )

    # Run FE
    X_train = run_feature_engineering(train_df, molecular_train, important_genes, data_manager)
    X_test = run_feature_engineering(test_df, molecular_test, important_genes, data_manager)

    feature_auto_cast = FLOAT32_POLICY.get("auto_detect_feature_frames", False)
    X_train = apply_float32_policy(X_train, candidates=float32_candidates, auto_detect=feature_auto_cast, context="FE (train)")
    X_test = apply_float32_policy(X_test, candidates=float32_candidates, auto_detect=feature_auto_cast, context="FE (test)")

    # ==========================================================================
    # STEP 5: FINALIZE AND PREPROCESS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: FINALIZE AND PREPROCESS")
    print("=" * 60)

    # Save IDs and target
    train_ids = X_train[ID_COLUMNS["patient"]].copy()
    test_ids = X_test[ID_COLUMNS["patient"]].copy()
    y_train = X_train[[TARGET_COLUMNS["status"], TARGET_COLUMNS["time"]]].copy()

    # Prepare for preprocessing
    drop_cols = [TARGET_COLUMNS["status"], TARGET_COLUMNS["time"], ID_COLUMNS["patient"]]
    X_train_proc = X_train.drop(columns=drop_cols, errors="ignore")
    X_test_proc = X_test.drop(columns=[ID_COLUMNS["patient"]], errors="ignore")

    # Explicit drops from config
    if REDUNDANCY_POLICY.get("explicit_drop"):
        X_train_proc = X_train_proc.drop(columns=REDUNDANCY_POLICY["explicit_drop"], errors="ignore")
        X_test_proc = X_test_proc.drop(columns=REDUNDANCY_POLICY["explicit_drop"], errors="ignore")

    # Drop CENTER column if not using OHE
    if not EXPERIMENT.get("use_center_ohe", False):
        X_train_proc = X_train_proc.drop(columns=[ID_COLUMNS["center"]], errors="ignore")
        X_test_proc = X_test_proc.drop(columns=[ID_COLUMNS["center"]], errors="ignore")

    # Align columns
    X_test_proc = X_test_proc.reindex(columns=X_train_proc.columns)

    # Inject auxiliary features for pipeline imputation (if enabled and early_imputation disabled)
    aux_cfg = PREPROCESSING.get("imputer_auxiliary_features", {})
    early_enabled = PREPROCESSING.get("early_imputation", {}).get("enabled", False)
    if aux_cfg.get("enabled", False) and not early_enabled:
        print("[PREP] Injecting auxiliary features for pipeline imputation...")
        X_train_proc, X_test_proc, aux_injected_cols = inject_auxiliary_features_for_pipeline(
            X_train_proc, X_test_proc
        )
        if aux_injected_cols:
            print(f"[PREP] Injected {len(aux_injected_cols)} auxiliary columns: {aux_injected_cols[:5]}{'...' if len(aux_injected_cols) > 5 else ''}")

    # ==========================================================================
    # HANDLE IMPUTATION: MAIN + EXTERNAL (if applicable)
    # ==========================================================================
    
    X_external_final = None
    
    if external_training_df is not None:
        # Two separate imputers: main (train+test) and external
        print("\n[IMPUTE] Using TWO separate imputers:")
        print("  - Imputer #1: fit on (train + test)")
        print(f"  - Imputer #2: fit on external training data ({len(external_training_df)} samples)")
        
        # Build pipeline and fit on train+test
        preprocessor = get_preprocessing_pipeline(X_train_proc, PREPROCESSING.get("imputer", "knn"))
        include_test = PREPROCESSING.get("imputer_fit_scope", {}).get("include_test_rows", False)
        fit_df = pd.concat([X_train_proc, X_test_proc], ignore_index=True) if include_test else X_train_proc
        preprocessor.fit(fit_df)
        
        # Transform train+test
        X_train_final = preprocessor.transform(X_train_proc)
        X_test_final = preprocessor.transform(X_test_proc)
        
        # Handle external training data separately
        print(f"\n[IMPUTE] Processing external training data...")
        
        # Feature engineering on external data
        X_ext = run_feature_engineering(
            external_training_df, molecular_test, important_genes, data_manager
        )
        X_ext = X_ext.set_index(ID_COLUMNS["patient"])
        
        # Drop center column (same as train/test)
        X_ext = X_ext.drop(columns=[ID_COLUMNS["center"]], errors="ignore")
        
        # Inject auxiliary features (same as train/test)
        if aux_cfg.get("enabled", False) and not early_enabled:
            ext_aux = create_auxiliary_features_for_imputation(external_training_df, molecular_test)
            if not ext_aux.empty:
                ext_aux = ext_aux.set_index(ID_COLUMNS["patient"])
                X_ext = X_ext.merge(ext_aux, left_index=True, right_index=True, how="left")
                # Re-inject with prefix
                X_ext_with_prefix, _, _ = inject_auxiliary_features_for_pipeline(
                    pd.DataFrame(X_ext), pd.DataFrame(X_ext)
                )
                X_ext = X_ext_with_prefix.iloc[:len(X_ext)]
        
        # Align columns with train
        X_ext = X_ext.reindex(columns=X_train_proc.columns)
        
        # Create second imputer for external data
        preprocessor_ext = get_preprocessing_pipeline(X_ext, PREPROCESSING.get("imputer", "knn"))
        preprocessor_ext.fit(X_ext)
        X_ext_final = preprocessor_ext.transform(X_ext)
        
        # Re-add ID
        X_ext_final.insert(0, ID_COLUMNS["patient"], external_training_df[ID_COLUMNS["patient"]].values)
        X_external_final = X_ext_final
    
    elif external_imputation_df is not None:
        # Single imputer on (train+test+external)
        print("\n[IMPUTE] Using SINGLE imputer on (train + test + external)")
        
        fit_dfs = [X_train_proc, X_test_proc]
        
        # Feature engineering on external imputation data
        X_ext = run_feature_engineering(
            external_imputation_df, molecular_test, important_genes, data_manager
        )
        X_ext = X_ext.set_index(ID_COLUMNS["patient"])
        X_ext = X_ext.drop(columns=[ID_COLUMNS["center"]], errors="ignore")
        
        # Inject auxiliary features (same as train/test)
        if aux_cfg.get("enabled", False) and not early_enabled:
            ext_aux = create_auxiliary_features_for_imputation(external_imputation_df, molecular_test)
            if not ext_aux.empty:
                ext_aux = ext_aux.set_index(ID_COLUMNS["patient"])
                X_ext = X_ext.merge(ext_aux, left_index=True, right_index=True, how="left")
                # Re-inject with prefix
                X_ext_with_prefix, _, _ = inject_auxiliary_features_for_pipeline(
                    pd.DataFrame(X_ext), pd.DataFrame(X_ext)
                )
                X_ext = X_ext_with_prefix.iloc[:len(X_ext)]
        
        # Align columns
        X_ext = X_ext.reindex(columns=X_train_proc.columns)
        fit_dfs.append(X_ext)
        
        # Build pipeline and fit on combined data
        preprocessor = get_preprocessing_pipeline(X_train_proc, PREPROCESSING.get("imputer", "knn"))
        fit_df = pd.concat(fit_dfs, ignore_index=True)
        preprocessor.fit(fit_df)
        
        # Transform all
        X_train_final = preprocessor.transform(X_train_proc)
        X_test_final = preprocessor.transform(X_test_proc)
    
    else:
        # Default: single imputer on (train+test) or (train) only
        preprocessor = get_preprocessing_pipeline(X_train_proc, PREPROCESSING.get("imputer", "knn"))
        include_test = PREPROCESSING.get("imputer_fit_scope", {}).get("include_test_rows", False)
        fit_df = pd.concat([X_train_proc, X_test_proc], ignore_index=True) if include_test else X_train_proc
        preprocessor.fit(fit_df)

        X_train_final = preprocessor.transform(X_train_proc)
        X_test_final = preprocessor.transform(X_test_proc)

    # Re-add IDs
    X_train_final.insert(0, ID_COLUMNS["patient"], train_ids.values)
    X_test_final.insert(0, ID_COLUMNS["patient"], test_ids.values)

    # Concatenate External Data for training if applicable
    if X_external_final is not None:
        print(f"\n[FINAL] Concatenating {len(X_external_final)} external samples to training set")
        # Align columns
        X_external_final = X_external_final.reindex(columns=X_train_final.columns)
        # Merge
        X_train_final = pd.concat([X_train_final, X_external_final], ignore_index=True)
        print(f"[FINAL] Training set now has {len(X_train_final)} samples (original: {len(train_ids)} + External: {len(X_external_final)})")
        
        # Also concatenate External target
        ext_targets = external_training_df[[TARGET_COLUMNS["status"], TARGET_COLUMNS["time"]]].reset_index(drop=True)
        y_train = pd.concat([y_train, ext_targets], ignore_index=True)

    # Add CENTER_GROUP if configured
    if EXPERIMENT.get("include_center_group_feature", False):
        center_map_train = train_df.set_index(ID_COLUMNS["patient"])[ID_COLUMNS["center"]].astype(str)
        center_map_test = test_df.set_index(ID_COLUMNS["patient"])[ID_COLUMNS["center"]].astype(str)
        X_train_final.insert(1, "CENTER_GROUP", X_train_final[ID_COLUMNS["patient"]].map(center_map_train).fillna("CENTER_OTHER"))
        X_test_final.insert(1, "CENTER_GROUP", X_test_final[ID_COLUMNS["patient"]].map(center_map_test).fillna("CENTER_OTHER"))

    # Apply feature subset mode
    X_train_final, X_test_final = apply_feature_subset_mode(X_train_final, X_test_final)

    # Drop zero variance columns
    if PREPROCESSING.get("drop_zero_variance", True):
        protected = set(PREPROCESSING.get("zero_variance_protected_columns", []) or [])
        protected_pfx = tuple(PREPROCESSING.get("zero_variance_protected_prefixes", []) or [])
        drop = []

        # Identify original training rows (excluding Beat AML) to respect user logic:
        # "Check constant in Train OR Test" (ignoring Beat AML for this check)
        n_original_train = len(train_ids)

        for col in X_train_final.columns:
            if col in [ID_COLUMNS["patient"], "CENTER_GROUP"] or col in protected:
                continue
            if protected_pfx and col.startswith(protected_pfx):
                continue
            
            # Check variance on Original Train only (first n_original_train rows)
            is_constant_train = X_train_final[col].iloc[:n_original_train].nunique(dropna=False) <= 1
            is_constant_test = X_test_final[col].nunique(dropna=False) <= 1

            if is_constant_train or is_constant_test:
                drop.append(col)
        if drop:
            print(f"[PREP] Dropping {len(drop)} zero-variance columns: {drop}")
            X_train_final = X_train_final.drop(columns=drop, errors="ignore")
            X_test_final = X_test_final.drop(columns=drop, errors="ignore")

    # Prune highly correlated features
    if EXPERIMENT.get("prune_feature", False):
        thr = float(EXPERIMENT.get("prune_feature_threshold", 0.96))
        print(f"\n[PREP] Pruning features with correlation > {thr}...")
        X_train_final, X_test_final = prune_highly_correlated_features_pair(
            X_train_final, X_test_final, threshold=thr, id_cols=("ID", "CENTER_GROUP")
        )

    # Prune rare binary features
    X_train_final, X_test_final = prune_rare_binary_features(
        X_train_final, X_test_final, RARE_EVENT_PRUNING_THRESHOLD, ["ID", "CENTER_GROUP"]
    )

    # Final float32 cast
    X_train_final = apply_float32_policy(X_train_final, auto_detect=True, context="final (train)")
    X_test_final = apply_float32_policy(X_test_final, auto_detect=True, context="final (test)")

    # ==========================================================================
    # STEP 6: SAVE
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: SAVE PROCESSED DATASETS")
    print("=" * 60)

    output_dir = DATA_PATHS["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train_final.to_csv(os.path.join(output_dir, "X_train_processed.csv"), index=False)
    X_test_final.to_csv(os.path.join(output_dir, "X_test_processed.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train_processed.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

    try:
        save_feature_list(tag, X_train_final.columns.tolist())
    except Exception as e:
        print(f"[TRACE] Could not save feature list: {e}")

    print(f"\n[OK] Pipeline complete. Output saved to '{output_dir}'")
    print(f"  - Train: {X_train_final.shape}, Test: {X_test_final.shape}")


if __name__ == "__main__":
    main()
