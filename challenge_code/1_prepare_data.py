import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.data_extraction.external_data_manager import ExternalDataManager
from src.modeling.pipeline_components import get_preprocessing_pipeline
import joblib
from src.config import (
    PREPROCESSING,
    EXPERIMENT,
    CLINICAL_RANGES,
    CLINICAL_NUMERIC_COLUMNS,
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
from src.utils.experiment import (
    save_manifest,
    save_feature_list,
    ensure_experiment_dir,
    compute_tag_with_signature,
)



from src.data.data_cleaning.cleaner import clean_and_validate_data

from src.data.features.feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
    MolecularFeatureExtraction,
    CytoMolecularInteractionFeatures,
)


# --- 2. IMPORTATION DES OUTILS DE MACHINE LEARNING ---
from sklearn.experimental import (
    enable_iterative_imputer,
)  # noqa: F401 (side-effect import)
from src.data.data_cleaning.imputer import AdvancedImputer, supervised_monocyte_imputation
from src.data.features.pruning import (
    prune_highly_correlated_features_pair,
    prune_rare_binary_features,
)

AUX_IMPUTE_PREFIX = PREPROCESSING.get("imputer_auxiliary_features", {}).get("prefix", "__aux_impute__")


def _get_beat_aml_usage() -> dict[str, bool]:
    """Retourne la stratégie d'utilisation de Beat AML (train vs imputation)."""
    beat_cfg = DATA_FUSION.get("beat_aml")
    if beat_cfg is not None:
        return {
            "use_for_training": bool(beat_cfg.get("use_for_training", False)),
            "use_for_imputation": bool(beat_cfg.get("use_for_imputation", False)),
        }

    # Rétrocompatibilité avec l'ancienne clé booléenne
    legacy_flag = DATA_FUSION.get("include_beat_aml", False)
    return {"use_for_training": bool(legacy_flag), "use_for_imputation": False}


def _align_columns(source: pd.DataFrame, template_cols: list[str], label: str) -> pd.DataFrame:
    """Garantit que les colonnes sont alignées sur un template commun."""
    missing_cols = [col for col in template_cols if col not in source.columns]
    if missing_cols:
        print(f"[FUSION] Colonnes manquantes ajoutées dans {label}: {missing_cols}")
    extra_cols = [col for col in source.columns if col not in template_cols]
    if extra_cols:
        print(f"[FUSION] Colonnes ignorées dans {label}: {extra_cols}")
    return source.reindex(columns=template_cols, fill_value=pd.NA)


def _load_beat_aml_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    required_files = {
        "clinical": BEAT_AML_PATHS.get("clinical"),
        "molecular": BEAT_AML_PATHS.get("molecular"),
        "target": BEAT_AML_PATHS.get("target"),
    }
    missing = [name for name, path in required_files.items() if not path or not Path(path).is_file()]
    if missing:
        print(f"[FUSION] Impossible de charger Beat AML: fichiers manquants {missing}.")
        return None

    beat_clinical = pd.read_csv(required_files["clinical"])
    beat_molecular = pd.read_csv(required_files["molecular"])
    beat_target = pd.read_csv(required_files["target"])
    return beat_clinical, beat_molecular, beat_target


def _prepare_beat_aml_dataset(
    *,
    existing_ids: set[str],
    clinical_columns: list[str],
    molecular_columns: list[str],
    target_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    sources = _load_beat_aml_sources()
    if sources is None:
        return None

    beat_clinical, beat_molecular, beat_target = sources
    beat_clinical_aligned = _align_columns(beat_clinical, clinical_columns, "clinical")
    beat_molecular_aligned = _align_columns(beat_molecular, molecular_columns, "molecular")
    beat_target_aligned = _align_columns(beat_target, target_columns, "target")

    for df in (beat_clinical_aligned, beat_molecular_aligned, beat_target_aligned):
        if ID_COLUMNS["patient"] in df.columns:
            df[ID_COLUMNS["patient"]] = df[ID_COLUMNS["patient"]].astype(str)

    existing_ids = {str(x) for x in existing_ids if pd.notna(x)}
    if existing_ids:
        beat_ids = set(beat_clinical_aligned[ID_COLUMNS["patient"]])
        overlap = existing_ids & beat_ids
        if overlap:
            preview = sorted(list(overlap))[:5]
            print(
                f"[FUSION] {len(overlap)} patients Beat AML déjà présents dans le train: {preview}..."
            )
            mask = ~beat_clinical_aligned[ID_COLUMNS["patient"]].isin(overlap)
            beat_clinical_aligned = beat_clinical_aligned[mask]
            beat_target_aligned = beat_target_aligned[
                ~beat_target_aligned[ID_COLUMNS["patient"]].isin(overlap)
            ]
            beat_molecular_aligned = beat_molecular_aligned[
                ~beat_molecular_aligned[ID_COLUMNS["patient"]].isin(overlap)
            ]

    if beat_clinical_aligned.empty or beat_target_aligned.empty:
        print("[FUSION] Aucun patient Beat AML exploitable après filtrage / alignement.")
        return None

    return beat_clinical_aligned, beat_molecular_aligned, beat_target_aligned


def _get_tcga_usage() -> dict[str, bool]:
    """Retourne la stratégie d'utilisation de TCGA (train vs imputation)."""
    tcga_cfg = DATA_FUSION.get("tcga")
    if tcga_cfg is not None:
        return {
            "use_for_training": bool(tcga_cfg.get("use_for_training", False)),
            "use_for_imputation": bool(tcga_cfg.get("use_for_imputation", False)),
        }
    return {"use_for_training": False, "use_for_imputation": False}


def _load_tcga_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    required_files = {
        "clinical": TCGA_PATHS.get("clinical"),
        "molecular": TCGA_PATHS.get("molecular"),
        "target": TCGA_PATHS.get("target"),
    }
    missing = [name for name, path in required_files.items() if not path or not Path(path).is_file()]
    if missing:
        print(f"[FUSION] Impossible de charger TCGA: fichiers manquants {missing}.")
        return None

    tcga_clinical = pd.read_csv(required_files["clinical"])
    tcga_molecular = pd.read_csv(required_files["molecular"])
    tcga_target = pd.read_csv(required_files["target"])
    return tcga_clinical, tcga_molecular, tcga_target


def _prepare_tcga_dataset(
    *,
    existing_ids: set[str],
    clinical_columns: list[str],
    molecular_columns: list[str],
    target_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    sources = _load_tcga_sources()
    if sources is None:
        return None

    tcga_clinical, tcga_molecular, tcga_target = sources
    tcga_clinical_aligned = _align_columns(tcga_clinical, clinical_columns, "clinical")
    tcga_molecular_aligned = _align_columns(tcga_molecular, molecular_columns, "molecular")
    tcga_target_aligned = _align_columns(tcga_target, target_columns, "target")

    for df in (tcga_clinical_aligned, tcga_molecular_aligned, tcga_target_aligned):
        if ID_COLUMNS["patient"] in df.columns:
            df[ID_COLUMNS["patient"]] = df[ID_COLUMNS["patient"]].astype(str)

    existing_ids = {str(x) for x in existing_ids if pd.notna(x)}
    if existing_ids:
        tcga_ids = set(tcga_clinical_aligned[ID_COLUMNS["patient"]])
        overlap = existing_ids & tcga_ids
        if overlap:
            preview = sorted(list(overlap))[:5]
            print(
                f"[FUSION] {len(overlap)} patients TCGA déjà présents dans le train: {preview}..."
            )
            mask = ~tcga_clinical_aligned[ID_COLUMNS["patient"]].isin(overlap)
            tcga_clinical_aligned = tcga_clinical_aligned[mask]
            tcga_target_aligned = tcga_target_aligned[
                ~tcga_target_aligned[ID_COLUMNS["patient"]].isin(overlap)
            ]
            tcga_molecular_aligned = tcga_molecular_aligned[
                ~tcga_molecular_aligned[ID_COLUMNS["patient"]].isin(overlap)
            ]

    if tcga_clinical_aligned.empty or tcga_target_aligned.empty:
        print("[FUSION] Aucun patient TCGA exploitable après filtrage / alignement.")
        return None

    return tcga_clinical_aligned, tcga_molecular_aligned, tcga_target_aligned


def _build_auxiliary_columns_for_fit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_columns: list[str],
    *,
    extra_fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build auxiliary columns to enrich the fit matrix for imputation.
    Returns a DataFrame with [target_columns + auxiliary_columns] for fitting.
    The auxiliary columns help the imputer learn better correlations.
    """
    cfg = PREPROCESSING.get("imputer_auxiliary_features", {})
    if not cfg.get("enabled", False):
        return pd.DataFrame()  # No auxiliary columns

    aux_columns = cfg.get("columns") or []
    if not aux_columns:
        return pd.DataFrame()

    # Filter to columns actually present in both train and test
    available_aux = [
        col for col in aux_columns
        if col in train_df.columns and col in test_df.columns and col not in target_columns
    ]
    if not available_aux:
        return pd.DataFrame()

    # Build combined matrix: target columns + auxiliary columns
    combined_cols = target_columns + available_aux
    fit_parts = [train_df[combined_cols]]
    
    include_test = PREPROCESSING.get("imputer_fit_scope", {}).get("include_test_rows", False)
    if include_test:
        fit_parts.append(test_df[combined_cols])
    if extra_fit_df is not None:
        extra_subset = extra_fit_df.reindex(columns=combined_cols)
        if len(extra_subset) > 0:
            fit_parts.append(extra_subset)

    fit_matrix = pd.concat(fit_parts, ignore_index=True)
    print(f"[PREP] Auxiliary columns for early imputation: {available_aux}")
    return fit_matrix, available_aux


def _inject_auxiliary_imputer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    extra_dfs: list[pd.DataFrame] | None = None,
) -> list[str]:
    """
    Inject auxiliary columns AFTER feature engineering for late imputation mode.
    Note: In early imputation mode, this is typically skipped or returns empty.
    """
    cfg = PREPROCESSING.get("imputer_auxiliary_features", {})
    if not cfg.get("enabled", False):
        return []

    # Skip injection if early imputation already used auxiliary columns
    early_cfg = PREPROCESSING.get("early_imputation", {})
    if early_cfg.get("enabled", False) and early_cfg.get("use_auxiliary_columns", True):
        print("[PREP] Auxiliary columns already used in early imputation, skipping late injection.")
        return []

    columns = cfg.get("columns") or []
    if not columns:
        return []

    added_cols: list[str] = []
    skip_missing: list[str] = []
    extra_dfs = extra_dfs or []

    for col in columns:
        if col not in train_df.columns or col not in test_df.columns:
            skip_missing.append(col)
            continue

        new_col = f"{AUX_IMPUTE_PREFIX}{col}"
        train_series = train_df[col]
        test_series = test_df[col]

        if pd.api.types.is_numeric_dtype(train_series) and pd.api.types.is_numeric_dtype(test_series):
            train_df[new_col] = train_series.astype("float32")
            test_df[new_col] = test_series.astype("float32")
            for extra_df in extra_dfs:
                if col in extra_df.columns:
                    extra_df[new_col] = extra_df[col].astype("float32")
                else:
                    extra_df[new_col] = np.nan
        else:
            series_list = [train_series, test_series]
            for extra_df in extra_dfs:
                if col in extra_df.columns:
                    series_list.append(extra_df[col])
                else:
                    series_list.append(pd.Series([pd.NA] * len(extra_df), index=extra_df.index))

            combined = pd.concat(series_list, ignore_index=True)
            combined_str = combined.astype("string")
            mask_na = combined_str.isna()
            filled = combined_str.fillna("__aux_missing__")
            codes, _ = pd.factorize(filled)
            codes = codes.astype("float32")
            codes[mask_na.to_numpy()] = np.nan
            splits = [len(train_series), len(test_series)] + [len(s) for s in series_list[2:]]
            start = 0
            targets = [train_df, test_df] + extra_dfs
            for target_df, seg_len in zip(targets, splits):
                target_slice = codes[start : start + seg_len]
                target_df[new_col] = target_slice
                start += seg_len

        added_cols.append(new_col)

    if added_cols:
        print(
            f"[PREP] Colonnes auxiliaires ajoutées pour l'imputation: {added_cols}"
        )
    if skip_missing:
        print(
            "[PREP] Colonnes auxiliaires indisponibles (absentes des datasets): "
            f"{skip_missing}"
        )
    return added_cols


def _drop_auxiliary_imputer_columns(
    df: pd.DataFrame, auxiliary_cols: list[str]
) -> None:
    if not auxiliary_cols:
        return
    drop_cols = [col for col in auxiliary_cols if col in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")


def _apply_feature_subset_mode(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = FEATURE_SET_POLICY or {}
    mode = (cfg.get("mode") or "default").lower()
    if mode in {"default", "normal", "all", "full"}:
        return train_df, test_df

    if mode == "core":
        requested = CORE_FEATURES
    elif mode == "exploratory":
        requested = EXPLORATORY_FEATURES
    else:
        print(f"[FEATURE SET] Mode '{mode}' inconnu, configuration ignorée.")
        return train_df, test_df

    # Preserve order as defined in training dataframe to avoid column mismatch.
    keep_cols = [ID_COLUMNS["patient"]]
    if "CENTER_GROUP" in train_df.columns:
        keep_cols.append("CENTER_GROUP")
    requested_set = set(requested)
    selected_cols = [col for col in train_df.columns if col in requested_set or col in keep_cols]

    if len(selected_cols) <= len(keep_cols):
        print("[FEATURE SET] Aucun recouvrement avec les colonnes demandées; configuration ignorée.")
        return train_df, test_df

    missing = [col for col in requested if col not in train_df.columns]
    if missing and cfg.get("warn_on_missing", True):
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        print(
            f"[FEATURE SET] {len(missing)} colonnes demandées absentes du train: {preview}{suffix}"
        )

    kept_feature_count = len([col for col in selected_cols if col not in keep_cols])
    print(f"[FEATURE SET] Mode '{mode}' activé ({kept_feature_count} features conservées).")

    train_df = train_df.loc[:, selected_cols]
    test_df = test_df.reindex(columns=selected_cols)

    return train_df, test_df



def maybe_include_beat_aml(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optionally append Beat AML datasets to the primary training data."""

    beat_usage = _get_beat_aml_usage()
    if not beat_usage.get("use_for_training", False):
        print("[FUSION] Ajout Beat AML désactivé pour l'entraînement.")
        return clinical_df, molecular_df, target_df

    prepared = _prepare_beat_aml_dataset(
        existing_ids=set(clinical_df[ID_COLUMNS["patient"]].astype(str)),
        clinical_columns=clinical_df.columns.tolist(),
        molecular_columns=molecular_df.columns.tolist(),
        target_columns=target_df.columns.tolist(),
    )
    if prepared is None:
        return clinical_df, molecular_df, target_df

    beat_clinical_aligned, beat_molecular_aligned, beat_target_aligned = prepared

    combined_clinical = pd.concat([clinical_df, beat_clinical_aligned], ignore_index=True)
    combined_target = pd.concat([target_df, beat_target_aligned], ignore_index=True)
    combined_molecular = pd.concat([molecular_df, beat_molecular_aligned], ignore_index=True)

    print(
        "[FUSION] Ajout Beat AML: +{patients} patients cliniques, +{mutations} mutations, +{targets} cibles.".format(
            patients=len(beat_clinical_aligned),
            mutations=len(beat_molecular_aligned),
            targets=len(beat_target_aligned),
        )
    )

    return combined_clinical, combined_molecular, combined_target


def maybe_include_tcga(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optionally append TCGA datasets to the primary training data."""

    tcga_usage = _get_tcga_usage()
    if not tcga_usage.get("use_for_training", False):
        print("[FUSION] Ajout TCGA désactivé pour l'entraînement.")
        return clinical_df, molecular_df, target_df

    prepared = _prepare_tcga_dataset(
        existing_ids=set(clinical_df[ID_COLUMNS["patient"]].astype(str)),
        clinical_columns=clinical_df.columns.tolist(),
        molecular_columns=molecular_df.columns.tolist(),
        target_columns=target_df.columns.tolist(),
    )
    if prepared is None:
        return clinical_df, molecular_df, target_df

    tcga_clinical_aligned, tcga_molecular_aligned, tcga_target_aligned = prepared

    combined_clinical = pd.concat([clinical_df, tcga_clinical_aligned], ignore_index=True)
    combined_target = pd.concat([target_df, tcga_target_aligned], ignore_index=True)
    combined_molecular = pd.concat([molecular_df, tcga_molecular_aligned], ignore_index=True)

    print(
        "[FUSION] Ajout TCGA: +{patients} patients cliniques, +{mutations} mutations, +{targets} cibles.".format(
            patients=len(tcga_clinical_aligned),
            mutations=len(tcga_molecular_aligned),
            targets=len(tcga_target_aligned),
        )
    )

    return combined_clinical, combined_molecular, combined_target


def run_feature_engineering(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    important_genes: list,
    data_manager: ExternalDataManager,
) -> pd.DataFrame:
    """
    Exécute le pipeline de feature engineering complet sur un jeu de données (train ou test).
    """
    print("\n[FE] Démarrage du Feature Engineering...")

    final_df = clinical_df.copy()
    final_df[ID_COLUMNS["patient"]] = final_df[ID_COLUMNS["patient"]].astype(str)
    if not molecular_df.empty:
        molecular_df[ID_COLUMNS["patient"]] = molecular_df[ID_COLUMNS["patient"]].astype(str)

    if FEATURE_ENGINEERING_TOGGLES.get("clinical", True):
        print("[FE] Création des features cliniques...")
        final_df = ClinicalFeatureEngineering.create_clinical_features(final_df)

    if FEATURE_ENGINEERING_TOGGLES.get("cytogenetic", True):
        print("[FE] Création des features cytogénétiques...")
        cyto_features = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
            final_df[[ID_COLUMNS["patient"], "CYTOGENETICS"]].copy()
        )
        final_df = pd.merge(final_df, cyto_features, on=ID_COLUMNS["patient"], how="left")

    if FEATURE_ENGINEERING_TOGGLES.get("molecular", True):
        print("[FE] Création des features moléculaires...")

        all_molecular_features = MolecularFeatureExtraction.create_all_molecular_features(
            base_df=final_df[[ID_COLUMNS["patient"]]],
            maf_df=molecular_df,
            important_genes=important_genes,  # Utilisation de la liste fournie
            external_data_manager=data_manager,
        )
        if not all_molecular_features.empty:
            final_df = pd.merge(final_df, all_molecular_features, on=ID_COLUMNS["patient"], how="left")
            mol_cols = [c for c in all_molecular_features.columns if c != ID_COLUMNS["patient"]]
            final_df[mol_cols] = final_df[mol_cols].fillna(0)

    final_df = final_df.drop(columns=["CYTOGENETICS"], errors="ignore")

    print(f"[FE] Feature Engineering terminé. Shape du dataframe : {final_df.shape}")
    missing_percentage = final_df.isnull().sum().sum() / final_df.size * 100
    print(f"[FE] Taux de valeurs manquantes résiduelles : {missing_percentage:.2f}%")
    if FEATURE_ENGINEERING_TOGGLES.get("cyto_molecular_interaction", True):
        final_df = CytoMolecularInteractionFeatures.create_interaction_features(final_df)
    return final_df


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
    protected.update(
        filter(
            None,
            [
                ID_COLUMNS.get("patient"),
                ID_COLUMNS.get("center"),
                TARGET_COLUMNS.get("status"),
                TARGET_COLUMNS.get("time"),
                "CENTER_GROUP",
            ],
        )
    )

    columns_to_cast: list[str] = []
    if candidates:
        columns_to_cast.extend(
            [col for col in candidates if col in df.columns and col not in protected]
        )
    if auto_detect:
        float_like_cols = df.select_dtypes(include=["float64", "Float64"]).columns
        columns_to_cast.extend([col for col in float_like_cols if col not in protected])

    # Deduplicate while preserving order
    seen = set()
    ordered_columns: list[str] = []
    for col in columns_to_cast:
        if col not in seen:
            seen.add(col)
            ordered_columns.append(col)

    if not ordered_columns:
        return df

    df[ordered_columns] = df[ordered_columns].astype("float32")
    if context:
        print(
            f"[FLOAT32] Converted {len(ordered_columns)} columns to float32 during {context}."
        )
    return df


def apply_early_continuous_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    extra_fit_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the configured continuous-feature imputation BEFORE feature engineering.
    
    This is the recommended approach because:
    1. Ratios and derived features will be computable (no NaN in sources)
    2. Auxiliary columns can guide the imputer to learn better correlations
    3. The pipeline finale becomes simpler (no double imputation)
    
    When use_auxiliary_columns=True, we fit the imputer on [target_cols + aux_cols]
    but only transform the target_cols. This allows the imputer to learn from
    correlated features without modifying them.
    """
    early_cfg = PREPROCESSING.get("early_imputation", {})
    if not early_cfg.get("enabled", False):
        print("[PREP] Early continuous imputation disabled in config.")
        return train_df, test_df

    candidate_columns = early_cfg.get("columns") or PREPROCESSING.get("continuous_features", [])
    protected = {TARGET_COLUMNS["status"], TARGET_COLUMNS["time"]}
    columns = [
        col
        for col in candidate_columns
        if col in train_df.columns and col in test_df.columns and col not in protected
    ]

    if not columns:
        print("[PREP] No continuous columns available for early imputation.")
        return train_df, test_df

    strategy = early_cfg.get("strategy", PREPROCESSING.get("imputer", "iterative"))
    n_neighbors = early_cfg.get("n_neighbors")
    
    print(
        f"[PREP] Early imputation on {len(columns)} continuous features using '{strategy}' strategy."
    )
    before_train = train_df[columns].isna().mean().mean()
    before_test = test_df[columns].isna().mean().mean()

    # --- Build fit matrix with optional auxiliary columns ---
    use_aux = early_cfg.get("use_auxiliary_columns", True)
    aux_cols = []
    
    if use_aux:
        result = _build_auxiliary_columns_for_fit(
            train_df, test_df, columns, extra_fit_df=extra_fit_df
        )
        if isinstance(result, tuple) and len(result) == 2:
            fit_matrix, aux_cols = result
        else:
            fit_matrix = None
            aux_cols = []
    else:
        fit_matrix = None

    # Fallback: build fit matrix without auxiliary columns
    if fit_matrix is None or fit_matrix.empty:
        fit_matrix = train_df[columns].copy()
        include_test_rows = PREPROCESSING.get("imputer_fit_scope", {}).get("include_test_rows", False)
        if include_test_rows:
            fit_matrix = pd.concat([fit_matrix, test_df[columns]], ignore_index=True)
            print("[PREP] -> Ajout des lignes de test pour l'ajustement de l'imputeur continu.")
        if extra_fit_df is not None:
            extra_subset = extra_fit_df.reindex(columns=columns)
            if len(extra_subset) > 0:
                fit_matrix = pd.concat([fit_matrix, extra_subset], ignore_index=True)
                print(f"[PREP] -> {len(extra_subset)} lignes externes ajoutées pour ajuster l'imputeur continu.")
        aux_cols = []

    # --- Fit imputer on full matrix (targets + auxiliary) ---
    imputer = AdvancedImputer(strategy=strategy, n_neighbors=n_neighbors)
    imputer.fit(fit_matrix)

    # --- Transform only the target columns ---
    # If we used auxiliary columns, we need to transform with them present
    if aux_cols:
        # Build transform matrices with auxiliary columns
        train_transform_df = train_df[columns + aux_cols].copy()
        test_transform_df = test_df[columns + aux_cols].copy()
        
        train_full_imputed = imputer.transform(train_transform_df)
        test_full_imputed = imputer.transform(test_transform_df)
        
        # Extract only the target columns (first len(columns) columns)
        train_imputed = train_full_imputed.iloc[:, :len(columns)].astype("float32")
        test_imputed = test_full_imputed.iloc[:, :len(columns)].astype("float32")
    else:
        train_imputed = imputer.transform(train_df[columns]).astype("float32")
        test_imputed = imputer.transform(test_df[columns]).astype("float32")

    train_df.loc[:, columns] = train_imputed.values
    test_df.loc[:, columns] = test_imputed.values

    after_train = train_df[columns].isna().mean().mean()
    after_test = test_df[columns].isna().mean().mean()
    print(
        f"[PREP] Train missing rate: {before_train:.2%} -> {after_train:.2%} | "
        f"Test missing rate: {before_test:.2%} -> {after_test:.2%}"
    )
    if aux_cols:
        print(f"[PREP] Imputation guided by {len(aux_cols)} auxiliary features: {aux_cols}")

    if early_cfg.get("respect_ranges", False):
        range_map = early_cfg.get("range_map") or CLINICAL_RANGES
        for col in columns:
            bounds = range_map.get(col)
            if bounds:
                train_df[col] = train_df[col].clip(*bounds)
                test_df[col] = test_df[col].clip(*bounds)

    artifact_path = early_cfg.get("artifact_path")
    if artifact_path:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(imputer, artifact_path)
        print(f"[PREP] Saved early imputer to {artifact_path}.")

    return train_df, test_df


# --- 5. SCRIPT PRINCIPAL ---
def main():
    """Exécute la pipeline de préparation de données de A à Z."""
    data_manager = ExternalDataManager(
        cosmic_path=DATA_PATHS["cosmic_file"],
        oncokb_path=DATA_PATHS["oncokb_file"],
        clinvar_path=DATA_PATHS.get("clinvar_vcf"),
    )

    tag, cfg_signature, full_cfg_snapshot, cfg_slice = compute_tag_with_signature()
    # Save manifest with preprocessing numeric details for traceability
    extras = {
        "preprocessing_report": {
            "imputer_strategy": PREPROCESSING.get("imputer"),
            "knn_n_neighbors": PREPROCESSING.get("knn", {}).get("n_neighbors"),
            "iterative": PREPROCESSING.get("iterative"),
            "clip_quantiles": PREPROCESSING.get("clip_quantiles"),
            "numeric_scaler": PREPROCESSING.get("numeric_scaler"),
            "monocyte_imputer": PREPROCESSING.get("monocyte_imputer"),
        },
        "clinical_ranges": CLINICAL_RANGES,
        "config_signature": cfg_signature,
    }
    exp_dir = save_manifest(tag, full_config=cfg_slice, extra=extras)
    # Save the full config snapshot for complete traceability
    try:
        full_cfg = full_cfg_snapshot
        base = ensure_experiment_dir(tag)
        with open(os.path.join(base, "config_full.json"), "w", encoding="utf-8") as f:
            json.dump(full_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[TRACE] Impossible d'enregistrer config_full.json: {e}")
    print("[REPORT] Prétraitement: ")
    print(
        f"         - imputer: {PREPROCESSING.get('imputer')}\n"
        f"         - knn.n_neighbors: {PREPROCESSING.get('knn', {}).get('n_neighbors')}\n"
        f"         - iterative: {PREPROCESSING.get('iterative')}\n"
        f"         - clip_quantiles: {PREPROCESSING.get('clip_quantiles')}\n"
        f"         - numeric_scaler: {PREPROCESSING.get('numeric_scaler')}"
    )
    print("[REPORT] Bornes cliniques utilisées:")
    for k, v in CLINICAL_RANGES.items():
        print(f"         - {k}: {v}")

    # --- Configuration des chemins ---
    output_dir = DATA_PATHS["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    # Output paths (written below)


    print("=" * 50)
    print("ÉTAPE 1 & 2: CHARGEMENT ET NETTOYAGE")
    print("=" * 50)
    clinical_train_raw = pd.read_csv(DATA_PATHS["input_clinical_train"])
    molecular_train_raw = pd.read_csv(DATA_PATHS["input_molecular_train"])
    target_train_raw = pd.read_csv(DATA_PATHS["input_target_train"])
    clinical_test_raw = pd.read_csv(DATA_PATHS["input_clinical_test"])
    molecular_test_raw = pd.read_csv(DATA_PATHS["input_molecular_test"])

    beat_usage = _get_beat_aml_usage()
    tcga_usage = _get_tcga_usage()
    imputer_fit_scope = PREPROCESSING.get("imputer_fit_scope", {})
    include_test_in_imputer = bool(imputer_fit_scope.get("include_test_rows", False))
    external_imputation_df = None
    external_imputation_payload: dict[str, pd.DataFrame] | None = None

    ext_clinical_list = []
    ext_molecular_list = []
    ext_target_list = []
    ext_merged_list = []

    # --- Beat AML for Imputation ---
    if beat_usage.get("use_for_imputation") and not beat_usage.get("use_for_training"):
        print("[FUSION] Chargement Beat AML pour l'imputation uniquement...")
        imputation_sources = _prepare_beat_aml_dataset(
            existing_ids=set(clinical_train_raw[ID_COLUMNS["patient"]].astype(str)),
            clinical_columns=clinical_train_raw.columns.tolist(),
            molecular_columns=molecular_train_raw.columns.tolist(),
            target_columns=target_train_raw.columns.tolist(),
        )
        if imputation_sources is not None:
            (
                beat_clinical_impute,
                beat_molecular_impute,
                beat_target_impute,
            ) = imputation_sources
            print("[FUSION] Nettoyage des données Beat AML (imputation only)...")
            (
                beat_clinical_impute_clean,
                beat_molecular_impute_clean,
                beat_target_impute_clean,
            ) = clean_and_validate_data(
                beat_clinical_impute.copy(),
                beat_molecular_impute.copy(),
                beat_target_impute.copy(),
            )
            merged_beat = pd.merge(
                beat_clinical_impute_clean,
                beat_target_impute_clean,
                on=ID_COLUMNS["patient"],
                how="inner",
            )
            if merged_beat.empty:
                print("[FUSION] Aucun patient Beat AML exploitable pour l'imputation après nettoyage.")
            else:
                merged_beat[ID_COLUMNS["patient"]] = merged_beat[ID_COLUMNS["patient"]].astype(str)
                ext_clinical_list.append(beat_clinical_impute_clean)
                ext_molecular_list.append(beat_molecular_impute_clean)
                ext_target_list.append(beat_target_impute_clean)
                ext_merged_list.append(merged_beat)
                print(
                    f"[FUSION] {len(merged_beat)} patients Beat AML serviront uniquement à ajuster les imputeurs."
                )
        else:
            print("[FUSION] Impossible d'obtenir un jeu Beat AML pour l'imputation.")

    # --- TCGA for Imputation ---
    if tcga_usage.get("use_for_imputation") and not tcga_usage.get("use_for_training"):
        print("[FUSION] Chargement TCGA pour l'imputation uniquement...")
        imputation_sources = _prepare_tcga_dataset(
            existing_ids=set(clinical_train_raw[ID_COLUMNS["patient"]].astype(str)),
            clinical_columns=clinical_train_raw.columns.tolist(),
            molecular_columns=molecular_train_raw.columns.tolist(),
            target_columns=target_train_raw.columns.tolist(),
        )
        if imputation_sources is not None:
            (
                tcga_clinical_impute,
                tcga_molecular_impute,
                tcga_target_impute,
            ) = imputation_sources
            print("[FUSION] Nettoyage des données TCGA (imputation only)...")
            (
                tcga_clinical_impute_clean,
                tcga_molecular_impute_clean,
                tcga_target_impute_clean,
            ) = clean_and_validate_data(
                tcga_clinical_impute.copy(),
                tcga_molecular_impute.copy(),
                tcga_target_impute.copy(),
            )
            merged_tcga = pd.merge(
                tcga_clinical_impute_clean,
                tcga_target_impute_clean,
                on=ID_COLUMNS["patient"],
                how="inner",
            )
            if merged_tcga.empty:
                print("[FUSION] Aucun patient TCGA exploitable pour l'imputation après nettoyage.")
            else:
                merged_tcga[ID_COLUMNS["patient"]] = merged_tcga[ID_COLUMNS["patient"]].astype(str)
                ext_clinical_list.append(tcga_clinical_impute_clean)
                ext_molecular_list.append(tcga_molecular_impute_clean)
                ext_target_list.append(tcga_target_impute_clean)
                ext_merged_list.append(merged_tcga)
                print(
                    f"[FUSION] {len(merged_tcga)} patients TCGA serviront uniquement à ajuster les imputeurs."
                )
        else:
            print("[FUSION] Impossible d'obtenir un jeu TCGA pour l'imputation.")

    # --- Combine External Imputation Data ---
    if ext_merged_list:
        external_imputation_df = pd.concat(ext_merged_list, ignore_index=True)
        external_imputation_payload = {
            "clinical": pd.concat(ext_clinical_list, ignore_index=True),
            "molecular": pd.concat(ext_molecular_list, ignore_index=True),
            "target": pd.concat(ext_target_list, ignore_index=True),
            "merged": external_imputation_df.copy(),
        }
        print(f"[FUSION] Total {len(external_imputation_df)} patients externes pour l'imputation.")

    clinical_train_raw, molecular_train_raw, target_train_raw = maybe_include_beat_aml(
        clinical_train_raw,
        molecular_train_raw,
        target_train_raw,
    )
    clinical_train_raw, molecular_train_raw, target_train_raw = maybe_include_tcga(
        clinical_train_raw,
        molecular_train_raw,
        target_train_raw,
    )

    clinical_train_clean, molecular_train_clean, target_train_clean = (
        clean_and_validate_data(
            clinical_train_raw, molecular_train_raw, target_train_raw
        )
    )

    # --- FILTER: Remove patients with NEITHER clinical NOR molecular data ---
    # 1. Identify patients with at least one non-NaN clinical value
    # We use CLINICAL_NUMERIC_COLUMNS to check for "meaningful" clinical data
    has_clinical_data_mask = clinical_train_clean[CLINICAL_NUMERIC_COLUMNS].notna().any(axis=1)
    ids_with_clinical = set(clinical_train_clean.loc[has_clinical_data_mask, ID_COLUMNS["patient"]])
    
    # 2. Identify patients with molecular data (presence in molecular dataframe)
    ids_with_molecular = set(molecular_train_clean[ID_COLUMNS["patient"]])
    
    # 3. Keep union
    valid_ids = ids_with_clinical.union(ids_with_molecular)
    
    n_total = len(clinical_train_clean)
    clinical_train_clean = clinical_train_clean[clinical_train_clean[ID_COLUMNS["patient"]].isin(valid_ids)]
    target_train_clean = target_train_clean[target_train_clean[ID_COLUMNS["patient"]].isin(valid_ids)]
    n_removed = n_total - len(clinical_train_clean)
    
    if n_removed > 0:
        print(f"[FILTER] Removed {n_removed} patients having neither clinical nor molecular data.")
    else:
        print("[FILTER] No patients removed (all have at least clinical or molecular data).")
    # -----------------------------------------------------------------------

    fake_target_test = pd.DataFrame(
        {
            ID_COLUMNS["patient"]: clinical_test_raw[ID_COLUMNS["patient"]],
            TARGET_COLUMNS["time"]: 1,
            TARGET_COLUMNS["status"]: 0,
        }
    )
    clinical_test_clean, molecular_test_clean, _ = clean_and_validate_data(
        clinical_test_raw, molecular_test_raw, fake_target_test
    )
    train_df = pd.merge(
        clinical_train_clean, target_train_clean, on=ID_COLUMNS["patient"], how="inner"
    )
    test_df = clinical_test_clean.copy()

    float32_candidates = FLOAT32_POLICY.get("columns") or PREPROCESSING.get(
        "continuous_features", []
    )
    train_df = apply_float32_policy(
        train_df,
        candidates=float32_candidates,
        context="post-clean merge (train)",
    )
    test_df = apply_float32_policy(
        test_df,
        candidates=float32_candidates,
        context="post-clean merge (test)",
    )

    # Optionally warm the MyVariant cache once for all variants present in train+test
    try:
        if MOLECULAR_EXTERNAL_SCORES.get("myvariant", {}).get(
            "prefetch_on_prepare", False
        ):
            print(
                "[PREP] Préchargement des annotations MyVariant (cache une seule fois)…"
            )
            mol_union = pd.concat(
                [molecular_train_clean, molecular_test_clean], ignore_index=True
            )
            data_manager.ensure_myvariant_cache_for_df(mol_union, snv_only=True)
        else:
            print(
                "[PREP] MyVariant: utilisation du cache existant (pas de re-téléchargement)."
            )
    except Exception as e:
        print(f"[PREP] Préchargement MyVariant ignoré (erreur non bloquante): {e}")

    # Optionally prefetch CADD scores to warm the cache (no-ops if disabled)
    try:
        cadd_cfg = MOLECULAR_EXTERNAL_SCORES.get("cadd", {})
        if cadd_cfg.get("enabled", False) and cadd_cfg.get(
            "prefetch_on_prepare", False
        ):
            print("[PREP] Préchargement des scores CADD (peut être long, réseau)...")
            data_manager.fetch_and_cache_cadd_scores(molecular_train_clean)
            data_manager.fetch_and_cache_cadd_scores(molecular_test_clean)
        else:
            print(
                "[PREP] CADD: utilisation du cache existant (pas de re-téléchargement)."
            )
    except Exception as e:
        print(f"[PREP] Préchargement CADD ignoré (erreur non bloquante): {e}")

    if CENTER_GROUPING.get("enabled", False):
        print("\n[PREP] Regroupement des centres rares...")
        threshold = CENTER_GROUPING.get("rare_center_threshold", 40)

        # Apprendre les effectifs sur le jeu d'entraînement UNIQUEMENT
        center_counts = train_df[ID_COLUMNS["center"]].value_counts()
        major_centers = center_counts[center_counts >= threshold].index.tolist()
        rare_centers = center_counts[center_counts < threshold].index.tolist()
        other_label = CENTER_GROUPING.get('other_label', 'CENTER_OTHER')

        print(f"   -> {len(major_centers)} centres majeurs conservés.")
        print(
            f"   -> {len(rare_centers)} centres rares vont être regroupés en '{other_label}'."
        )

        # Appliquer la transformation aux deux datasets (train et test)
        train_df[ID_COLUMNS["center"]] = train_df[ID_COLUMNS["center"]].apply(
            lambda x: x if x in major_centers else other_label
        )
        test_df[ID_COLUMNS["center"]] = test_df[ID_COLUMNS["center"]].apply(
            lambda x: x if x in major_centers else other_label
        )

        if external_imputation_payload is not None:
            for key in ("merged", "clinical"):
                df_payload = external_imputation_payload.get(key)
                if df_payload is not None and ID_COLUMNS["center"] in df_payload.columns:
                    df_payload[ID_COLUMNS["center"]] = df_payload[ID_COLUMNS["center"]].apply(
                        lambda x: x if x in major_centers else other_label
                    )
            external_imputation_df = external_imputation_payload.get("merged")


        print("\nEffectifs après regroupement (sur le train set):")
        print(train_df[ID_COLUMNS["center"]].value_counts())
    else:
        print("\n[PREP] Regroupement des centres rares désactivé.")

    keep_monocyte_indicator = EXPERIMENT.get("keep_monocyte_indicator", False)
    monocyte_indicator_train = None
    monocyte_indicator_test = None
    if keep_monocyte_indicator and "MONOCYTES" in train_df.columns:
        print("\n[PREP] Création de l'indicateur MONOCYTES_missing avant imputation…")

        def _apply_indicator(df: pd.DataFrame, label: str) -> pd.Series | None:
            if "MONOCYTES" not in df.columns:
                print(f"   -> Indicateur ignoré: colonne MONOCYTES absente ({label}).")
                return None
            mask = df["MONOCYTES"].isna().astype("int8")
            df["MONOCYTES_missing"] = mask
            return mask.copy()

        monocyte_indicator_train = _apply_indicator(train_df, "train")
        monocyte_indicator_test = _apply_indicator(test_df, "test")

    monocyte_mode = PREPROCESSING.get("monocyte_mode", "separate")
    use_supervised_monocyte = (
        monocyte_mode == "separate"
        and EXPERIMENT.get("use_monocyte_supervised", False)
        and "MONOCYTES" in train_df.columns
    )
    try:
        if use_supervised_monocyte:
            mono_cfg = PREPROCESSING.get("monocyte_imputer", {})
            train_df, test_df = supervised_monocyte_imputation(
                train_df,
                test_df,
                keep_indicator=EXPERIMENT.get("keep_monocyte_indicator", False),
                model_path=mono_cfg.get(
                    "model_path", os.path.join("models", "monocyte_imputer.joblib")
                ),
                extra_fit_df=external_imputation_df,
                include_test_rows=include_test_in_imputer,
                **{k: v for k, v in mono_cfg.items() if k not in {"model_path"}},
            )
        else:
            if monocyte_mode != "separate":
                print("[MONO] Mode 'joint' activé: MONOCYTES sera imputé via le pipeline global.")
            elif not EXPERIMENT.get("use_monocyte_supervised", False):
                print("[MONO] Imputation supervisée désactivée (paramètre EXPERIMENT).")
            elif "MONOCYTES" not in train_df.columns:
                print("[MONO] Colonne MONOCYTES absente. Aucune imputation dédiée.")
    except (
        Exception
    ) as e:  # noqa: BLE001 (intentional broad catch to keep pipeline running)
        print(f"[MONO] Imputation supervisée ignorée (erreur): {e}")

    if keep_monocyte_indicator:
        if monocyte_indicator_train is not None and "MONOCYTES_missing" in train_df.columns:
            train_df["MONOCYTES_missing"] = monocyte_indicator_train.astype("int8")
        if monocyte_indicator_test is not None and "MONOCYTES_missing" in test_df.columns:
            test_df["MONOCYTES_missing"] = monocyte_indicator_test.astype("int8")

    train_df, test_df = apply_early_continuous_imputation(
        train_df,
        test_df,
        extra_fit_df=external_imputation_df,
    )
    train_df = apply_float32_policy(
        train_df,
        candidates=float32_candidates,
        context="after early imputation (train)",
    )
    test_df = apply_float32_policy(
        test_df,
        candidates=float32_candidates,
        context="after early imputation (test)",
    )

    print("\n" + "=" * 50)
    print("ÉTAPE 2.5: FILTRAGE DES GÈNES MOLÉCULAIRES")
    print("=" * 50)

    important_genes_final_list = MolecularFeatureExtraction.get_frequent_genes(
        train_maf=molecular_train_clean, test_maf=molecular_test_clean
    )

    print("\n" + "=" * 50)
    print("ÉTAPE 3: FEATURE ENGINEERING")
    print("=" * 50)
    X_train_featured = run_feature_engineering(
        clinical_df=train_df,
        molecular_df=molecular_train_clean,
        important_genes=important_genes_final_list,
        data_manager=data_manager,
    )
    X_test_featured = run_feature_engineering(
        clinical_df=test_df,
        molecular_df=molecular_test_clean,
        important_genes=important_genes_final_list,
        data_manager=data_manager,
    )

    feature_auto_cast = FLOAT32_POLICY.get("auto_detect_feature_frames", False)
    X_external_featured: pd.DataFrame | None = None
    if external_imputation_payload is not None:
        print("[FUSION] Feature engineering sur Beat AML (imputation finale)...")
        X_external_featured = run_feature_engineering(
            clinical_df=external_imputation_payload["merged"],
            molecular_df=external_imputation_payload["molecular"],
            important_genes=important_genes_final_list,
            data_manager=data_manager,
        )
        X_external_featured = apply_float32_policy(
            X_external_featured,
            candidates=float32_candidates,
            auto_detect=feature_auto_cast,
            context="feature engineering (beat aml)",
        )

    auxiliary_imputer_cols = _inject_auxiliary_imputer_features(
        X_train_featured,
        X_test_featured,
        extra_dfs=[X_external_featured] if X_external_featured is not None else None,
    )

    X_train_featured = apply_float32_policy(
        X_train_featured,
        candidates=float32_candidates,
        auto_detect=feature_auto_cast,
        context="feature engineering (train)",
    )
    X_test_featured = apply_float32_policy(
        X_test_featured,
        candidates=float32_candidates,
        auto_detect=feature_auto_cast,
        context="feature engineering (test)",
    )

    print("\n" + "=" * 50)
    print("ÉTAPE 4: FINALISATION ET SÉPARATION")
    print("=" * 50)

    test_ids = X_test_featured[ID_COLUMNS["patient"]].copy()
    train_ids = X_train_featured[ID_COLUMNS["patient"]].copy()

    y_train_df = X_train_featured[[TARGET_COLUMNS["status"], TARGET_COLUMNS["time"]]].copy()


    X_train_to_process = X_train_featured.drop(
        columns=[TARGET_COLUMNS["status"], TARGET_COLUMNS["time"], ID_COLUMNS["patient"]], errors="ignore"
    )
    X_test_to_process = X_test_featured.drop(columns=[ID_COLUMNS["patient"]], errors="ignore")
    X_external_to_process: pd.DataFrame | None = None
    if X_external_featured is not None:
        X_external_to_process = X_external_featured.drop(
            columns=[TARGET_COLUMNS["status"], TARGET_COLUMNS["time"], ID_COLUMNS["patient"]],
            errors="ignore",
        )

    # Suppression explicite de colonnes
    if REDUNDANCY_POLICY.get("explicit_drop"):
        print(f"[PREP] Suppression explicite des colonnes: {REDUNDANCY_POLICY['explicit_drop']}")
        X_train_to_process = X_train_to_process.drop(columns=REDUNDANCY_POLICY["explicit_drop"], errors="ignore")
        X_test_to_process = X_test_to_process.drop(columns=REDUNDANCY_POLICY["explicit_drop"], errors="ignore")
        if X_external_to_process is not None:
            X_external_to_process = X_external_to_process.drop(
                columns=REDUNDANCY_POLICY["explicit_drop"], errors="ignore"
            )


    if not EXPERIMENT.get("use_center_ohe", False):
        datasets = [("train", X_train_to_process), ("test", X_test_to_process)]
        if X_external_to_process is not None:
            datasets.append(("beat_aml_impute", X_external_to_process))
        for df_name, df in datasets:
            if ID_COLUMNS["center"] in df.columns:
                print(
                    f"[PREP] Suppression de la colonne {ID_COLUMNS['center']} du jeu {df_name} (éviter one-hot dégénéré)."
                )
                df.drop(columns=[ID_COLUMNS["center"]], inplace=True)


    train_cols = X_train_to_process.columns
    X_test_to_process = X_test_to_process.reindex(columns=train_cols)
    if X_external_to_process is not None:
        X_external_to_process = X_external_to_process.reindex(columns=train_cols)


    print("\n" + "=" * 50)
    print("ÉTAPE 5: PRÉTRAITEMENT (IMPUTATION, SCALING, ENCODING)")
    print("=" * 50)

    preprocessor = get_preprocessing_pipeline(
        X_train_to_process, PREPROCESSING.get("imputer", "knn")
    )


    print("[PREPROCESSING] Entraînement du préprocesseur...")
    preprocessor_fit_df = X_train_to_process
    if include_test_in_imputer:
        preprocessor_fit_df = pd.concat([preprocessor_fit_df, X_test_to_process], ignore_index=True)
        print("[PREPROCESSING] -> Ajout des lignes de test lors du fit du préprocesseur.")
    if X_external_to_process is not None:
        preprocessor_fit_df = pd.concat([preprocessor_fit_df, X_external_to_process], ignore_index=True)
        print("[PREPROCESSING] -> Ajout des lignes Beat AML lors du fit du préprocesseur.")
    preprocessor.fit(preprocessor_fit_df)


    print("[PREPROCESSING] Transformation des données d'entraînement...")
    # La sortie est maintenant DIRECTEMENT un DataFrame avec les bons noms !
    X_train_processed_df = preprocessor.transform(X_train_to_process)

    print("[PREPROCESSING] Transformation des données de test...")
    X_test_processed_df = preprocessor.transform(X_test_to_process)

    if auxiliary_imputer_cols:
        _drop_auxiliary_imputer_columns(X_train_processed_df, auxiliary_imputer_cols)
        _drop_auxiliary_imputer_columns(X_test_processed_df, auxiliary_imputer_cols)
        print(
            f"[PREPROCESSING] Colonnes auxiliaires retirées après imputation: {auxiliary_imputer_cols}"
        )

    # Plus besoin de reconstruire les noms de colonnes manuellement !

    X_train_processed_df.insert(0, ID_COLUMNS["patient"], train_ids.values)
    X_test_processed_df.insert(0, ID_COLUMNS["patient"], test_ids.values)


    if EXPERIMENT.get("include_center_group_feature", True):
        try:
            center_map_train = train_df.set_index(ID_COLUMNS["patient"])[ID_COLUMNS["center"]].astype(str)
            center_map_test = test_df.set_index(ID_COLUMNS["patient"])[ID_COLUMNS["center"]].astype(str)
            X_train_processed_df.insert(
                1,
                "CENTER_GROUP",
                X_train_processed_df[ID_COLUMNS["patient"]]
                .map(center_map_train)
                .fillna("CENTER_OTHER")
                .values,
            )
            X_test_processed_df.insert(
                1,
                "CENTER_GROUP",
                X_test_processed_df[ID_COLUMNS["patient"]]
                .map(center_map_test)
                .fillna("CENTER_OTHER")
                .values,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[PREP] Impossible d'ajouter CENTER_GROUP (continuons sans): {e}")


    X_train_processed_df, X_test_processed_df = _apply_feature_subset_mode(
        X_train_processed_df, X_test_processed_df
    )


    if PREPROCESSING.get("drop_zero_variance", True):
        protected_columns = set(
            PREPROCESSING.get("zero_variance_protected_columns", []) or []
        )
        protected_prefixes = tuple(
            PREPROCESSING.get("zero_variance_protected_prefixes", []) or []
        )
        drop_cols = []
        for col in X_train_processed_df.columns:
            if col in [ID_COLUMNS["patient"], "CENTER_GROUP"]:
                continue
            if col in protected_columns:
                continue
            if protected_prefixes and col.startswith(protected_prefixes):
                continue
            nunique_train = X_train_processed_df[col].nunique(dropna=False)
            nunique_test = X_test_processed_df[col].nunique(dropna=False)
            if nunique_train <= 1 or nunique_test <= 1:
                drop_cols.append(col)

        if drop_cols:
            print(
                f"[PREPROCESSING] Suppression de {len(drop_cols)} colonnes à variance nulle: {drop_cols}"
            )
        X_train_processed_df.drop(columns=drop_cols, inplace=True, errors="ignore")
        X_test_processed_df.drop(columns=drop_cols, inplace=True, errors="ignore")
    else:
        print("[PREPROCESSING] Suppression des colonnes à variance nulle désactivée.")


    try:
        if EXPERIMENT.get("prune_feature", False):
            thr = float(EXPERIMENT.get("prune_feature_threshold", 0.90))
            print("\n" + "=" * 50)
            print("ÉTAPE 5.2: PRUNING DES FEATURES FORTEMENT CORRÉLÉES")
            print("=" * 50)

            X_train_processed_df, X_test_processed_df = (
                prune_highly_correlated_features_pair(
                    X_train_processed_df,
                    X_test_processed_df,
                    threshold=thr,
                    id_cols=("ID", "CENTER_GROUP"),
                )
            )
    except Exception as e:  # noqa: BLE001
        print(f"[PRUNING] Ignoré suite à une erreur non bloquante: {e}")

    # pruning des features rares
    X_train_processed_df, X_test_processed_df = prune_rare_binary_features(
        X_train_processed_df,
        X_test_processed_df,
        RARE_EVENT_PRUNING_THRESHOLD,
        ["ID", "CENTER_GROUP"],
    )

    processed_auto_cast = FLOAT32_POLICY.get("auto_detect_processed_frames", False)
    X_train_processed_df = apply_float32_policy(
        X_train_processed_df,
        auto_detect=processed_auto_cast,
        context="post-processing (train)",
    )
    X_test_processed_df = apply_float32_policy(
        X_test_processed_df,
        auto_detect=processed_auto_cast,
        context="post-processing (test)",
    )


    print("\n" + "=" * 50)
    print("ÉTAPE 6: SAUVEGARDE DES DATASETS TRAITÉS ET DU PRÉPROCESSEUR")
    print("=" * 50)

    # Sauvegarde des datasets qui seront lus par le script d'entraînement
    output_dir = DATA_PATHS["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    X_train_processed_df.to_csv(
        os.path.join(output_dir, "X_train_processed.csv"), index=False
    )
    X_test_processed_df.to_csv(
        os.path.join(output_dir, "X_test_processed.csv"), index=False
    )
    y_train_df.to_csv(os.path.join(output_dir, "y_train_processed.csv"), index=False)


    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))


    try:
        save_feature_list(tag, X_train_processed_df.columns.tolist())
    except Exception as e:  # noqa: BLE001
        print(f"[TRACE] Impossible d'enregistrer la liste des features: {e}")

    print("Pipeline de préparation des données terminée avec succès.")
    print(f"Fichiers finaux prêts dans le dossier '{output_dir}'")


if __name__ == "__main__":
    main()
