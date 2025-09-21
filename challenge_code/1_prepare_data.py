import os
import pandas as pd
from src.data.data_extraction.external_data_manager import ExternalDataManager
from src.modeling.pipeline_components import get_preprocessing_pipeline
import joblib
from src.config import (
    PREPROCESSING,
    EXPERIMENT,
    CLINICAL_RANGES,
    MOLECULAR_EXTERNAL_SCORES,
    RARE_EVENT_PRUNING_TRESHOLD,
    DATA_PATHS,
    ID_COLUMNS,
    TARGET_COLUMNS,
    CENTER_GROUPING,
    FEATURE_ENGINEERING_TOGGLES,
    REDUNDANCY_POLICY,
    MODEL_DIR,
)
from src.utils.experiment import (
    compute_tag,
    save_manifest,
    save_feature_list,
    ensure_experiment_dir,
    get_full_config_snapshot,
)

# --- 1. IMPORTATION DE VOTRE LOGIQUE MÉTIER ---
# Assurez-vous que le dossier 'src' est accessible depuis l'endroit où vous lancez ce script
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
from src.data.data_cleaning.imputer import supervised_monocyte_imputation
from src.data.features.pruning import (
    prune_highly_correlated_features_pair,
    prune_rare_binary_features,
)


def run_feature_engineering(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    important_genes: list,  # Accepte maintenant la liste pré-calculée
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
        # L'appel à la fonction moléculaire utilise maintenant la liste de gènes fixe
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


# --- 5. SCRIPT PRINCIPAL ---
def main():
    """Exécute la pipeline de préparation de données de A à Z."""
    data_manager = ExternalDataManager(
        cosmic_path=DATA_PATHS["cosmic_file"], oncokb_path=DATA_PATHS["oncokb_file"]
    )

    # --- Tag d'expérience et manifest ---
    cfg_slice = {"PREPROCESSING": PREPROCESSING, "EXPERIMENT": EXPERIMENT}
    tag = compute_tag(cfg_slice, prefix=EXPERIMENT.get("name"))
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
    }
    exp_dir = save_manifest(tag, full_config=cfg_slice, extra=extras)
    # Save the full config snapshot for complete traceability
    try:
        full_cfg = get_full_config_snapshot()
        base = ensure_experiment_dir(tag)
        with open(os.path.join(base, "config_full.json"), "w", encoding="utf-8") as f:
            import json as _json

            _json.dump(full_cfg, f, ensure_ascii=False, indent=2)
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

    # --- ÉTAPE 1 & 2: CHARGEMENT ET NETTOYAGE ---
    print("=" * 50)
    print("ÉTAPE 1 & 2: CHARGEMENT ET NETTOYAGE")
    print("=" * 50)
    clinical_train_raw = pd.read_csv(DATA_PATHS["input_clinical_train"])
    molecular_train_raw = pd.read_csv(DATA_PATHS["input_molecular_train"])
    target_train_raw = pd.read_csv(DATA_PATHS["input_target_train"])
    clinical_test_raw = pd.read_csv(DATA_PATHS["input_clinical_test"])
    molecular_test_raw = pd.read_csv(DATA_PATHS["input_molecular_test"])

    clinical_train_clean, molecular_train_clean, target_train_clean = (
        clean_and_validate_data(
            clinical_train_raw, molecular_train_raw, target_train_raw
        )
    )
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

        print(f"   -> {len(major_centers)} centres majeurs conservés.")
        print(
            f"   -> {len(rare_centers)} centres rares vont être regroupés en '{CENTER_GROUPING.get('other_label', 'CENTER_OTHER')}'."
        )

        # Appliquer la transformation aux deux datasets (train et test)
        train_df[ID_COLUMNS["center"]] = train_df[ID_COLUMNS["center"]].apply(
            lambda x: x if x in major_centers else CENTER_GROUPING.get('other_label', 'CENTER_OTHER')
        )
        test_df[ID_COLUMNS["center"]] = test_df[ID_COLUMNS["center"]].apply(
            lambda x: x if x in major_centers else CENTER_GROUPING.get('other_label', 'CENTER_OTHER')
        )

        # Afficher les nouveaux effectifs pour vérification
        print("\nEffectifs après regroupement (sur le train set):")
        print(train_df[ID_COLUMNS["center"]].value_counts())
    else:
        print("\n[PREP] Regroupement des centres rares désactivé.")
    # Imputation supervisée des monocytes avant le feature engineering pour réduire le drift
    try:
        if (
            EXPERIMENT.get("use_monocyte_supervised", False)
            and "MONOCYTES" in train_df.columns
        ):
            mono_cfg = PREPROCESSING.get("monocyte_imputer", {})
            train_df, test_df = supervised_monocyte_imputation(
                train_df,
                test_df,
                keep_indicator=EXPERIMENT.get("keep_monocyte_indicator", False),
                model_path=mono_cfg.get(
                    "model_path", os.path.join("models", "monocyte_imputer.joblib")
                ),
                **{k: v for k, v in mono_cfg.items() if k not in {"model_path"}},
            )
    except (
        Exception
    ) as e:  # noqa: BLE001 (intentional broad catch to keep pipeline running)
        print(f"[MONO] Imputation supervisée ignorée (erreur): {e}")
    # --- NOUVELLE ÉTAPE 2.5: DÉTERMINATION DE LA LISTE DE GÈNES ---
    print("\n" + "=" * 50)
    print("ÉTAPE 2.5: FILTRAGE DES GÈNES MOLÉCULAIRES")
    print("=" * 50)
    # On détermine la liste une seule fois en utilisant les données de train ET de test
    important_genes_final_list = MolecularFeatureExtraction.get_frequent_genes(
        train_maf=molecular_train_clean, test_maf=molecular_test_clean
    )
    # --- ÉTAPE 3: FEATURE ENGINEERING ---
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

    print("\n" + "=" * 50)
    print("ÉTAPE 4: FINALISATION ET SÉPARATION")
    print("=" * 50)

    test_ids = X_test_featured[ID_COLUMNS["patient"]].copy()
    train_ids = X_train_featured[ID_COLUMNS["patient"]].copy()

    y_train_df = X_train_featured[[TARGET_COLUMNS["status"], TARGET_COLUMNS["time"]]].copy()

    # Séparer les features de la target dans le set d'entraînement
    X_train_to_process = X_train_featured.drop(
        columns=[TARGET_COLUMNS["status"], TARGET_COLUMNS["time"], ID_COLUMNS["patient"]], errors="ignore"
    )
    X_test_to_process = X_test_featured.drop(columns=[ID_COLUMNS["patient"]], errors="ignore")

    # Suppression explicite de colonnes
    if REDUNDANCY_POLICY.get("explicit_drop"):
        print(f"[PREP] Suppression explicite des colonnes: {REDUNDANCY_POLICY['explicit_drop']}")
        X_train_to_process = X_train_to_process.drop(columns=REDUNDANCY_POLICY["explicit_drop"], errors="ignore")
        X_test_to_process = X_test_to_process.drop(columns=REDUNDANCY_POLICY["explicit_drop"], errors="ignore")

    # Retirer CENTER des features pour éviter un fort décalage train/test (test = CENTER_OTHER uniquement)
    if not EXPERIMENT.get("use_center_ohe", False):
        for df_name, df in [("train", X_train_to_process), ("test", X_test_to_process)]:
            if ID_COLUMNS["center"] in df.columns:
                print(
                    f"[PREP] Suppression de la colonne {ID_COLUMNS['center']} du jeu {df_name} (éviter one-hot dégénéré)."
                )
                df.drop(columns=[ID_COLUMNS["center"]], inplace=True)

    # Garantir la cohérence des colonnes entre train et test
    train_cols = X_train_to_process.columns
    X_test_to_process = X_test_to_process.reindex(columns=train_cols)

    # --- ÉTAPE 5: PRÉTRAITEMENT COMPLET ---
    print("\n" + "=" * 50)
    print("ÉTAPE 5: PRÉTRAITEMENT (IMPUTATION, SCALING, ENCODING)")
    print("=" * 50)

    preprocessor = get_preprocessing_pipeline(
        X_train_to_process, PREPROCESSING.get("imputer", "knn")
    )

    # Entraîner le préprocesseur
    print("[PREPROCESSING] Entraînement du préprocesseur...")
    preprocessor.fit(X_train_to_process)

    # Transformer les deux jeux de données
    print("[PREPROCESSING] Transformation des données d'entraînement...")
    # La sortie est maintenant DIRECTEMENT un DataFrame avec les bons noms !
    X_train_processed_df = preprocessor.transform(X_train_to_process)

    print("[PREPROCESSING] Transformation des données de test...")
    X_test_processed_df = preprocessor.transform(X_test_to_process)

    # Plus besoin de reconstruire les noms de colonnes manuellement !
    # Il suffit de réinsérer les IDs.
    X_train_processed_df.insert(0, ID_COLUMNS["patient"], train_ids.values)
    X_test_processed_df.insert(0, ID_COLUMNS["patient"], test_ids.values)

    # Ajouter un identifiant de groupe de centre pour la CV (non utilisé comme feature)
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

    # --- ÉTAPE 5.1: SUPPRESSION DES FEATURES À VARIANCE NULLE (sur train OU test) ---
    if PREPROCESSING.get("drop_zero_variance", True):
        drop_cols = []
        for col in X_train_processed_df.columns:
            if col in [ID_COLUMNS["patient"], "CENTER_GROUP"]:
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

    # --- ÉTAPE 5.2: PRUNING OPTIONNEL DES FEATURES FORTEMENT CORRÉLÉES ---
    try:
        if EXPERIMENT.get("prune_feature", False):
            thr = float(EXPERIMENT.get("prune_feature_threshold", 0.90))
            print("\n" + "=" * 50)
            print("ÉTAPE 5.2: PRUNING DES FEATURES FORTEMENT CORRÉLÉES")
            print("=" * 50)
            # Préserver ID et CENTER_GROUP
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
        RARE_EVENT_PRUNING_TRESHOLD,
        ["ID", "CENTER_GROUP"],
    )

    # --- ÉTAPE 6: SAUVEGARDE DES DONNÉES FINALES ET DES ARTEFACTS ---
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

    # Sauvegarde du préprocesseur ENTRAÎNÉ. C'est lui qui sera utilisé pour la prédiction finale.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

    # Enregistrer la liste des features finales pour la traçabilité
    try:
        save_feature_list(tag, X_train_processed_df.columns.tolist())
    except Exception as e:  # noqa: BLE001
        print(f"[TRACE] Impossible d'enregistrer la liste des features: {e}")

    print("Pipeline de préparation des données terminée avec succès.")
    print(f"Fichiers finaux prêts dans le dossier '{output_dir}'")


if __name__ == "__main__":
    main()
