import os
import pandas as pd
import numpy as np
from src.data.data_extraction.external_data_manager import ExternalDataManager
from src.modeling.pipeline_components import get_preprocessing_pipeline
import joblib

# --- 1. IMPORTATION DE VOTRE LOGIQUE MÉTIER ---
# Assurez-vous que le dossier 'src' est accessible depuis l'endroit où vous lancez ce script
from src.data.data_cleaning.cleaner import clean_and_validate_data
from src.data.features.feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
    MolecularFeatureExtraction,
    IntegratedFeatureEngineering,
)


# --- 2. IMPORTATION DES OUTILS DE MACHINE LEARNING ---
from sklearn.experimental import enable_iterative_imputer


# --- 4. FONCTION D'ORCHESTRATION DU FEATURE ENGINEERING ---
def run_feature_engineering(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    data_manager: ExternalDataManager,
) -> pd.DataFrame:
    """
    Exécute un pipeline de feature engineering de manière linéaire, robuste et simplifiée.
    """
    print("\n[FE] Démarrage du Feature Engineering (Pipeline Révisée)...")

    # --- Étape 1: Initialisation avec la cohorte clinique ---
    final_df = clinical_df.copy()
    final_df["ID"] = final_df["ID"].astype(str)
    molecular_df["ID"] = molecular_df["ID"].astype(str)

    # --- Étape 2: Enrichissement avec les features cliniques ---
    # La fonction ajoute directement les colonnes au dataframe. Plus besoin de merge.
    print("[FE] Création des features cliniques...")
    final_df = ClinicalFeatureEngineering.create_clinical_features(final_df)

    # --- Étape 3: Enrichissement avec les features cytogénétiques ---
    # La fonction extrait les features et les retourne alignées avec les IDs.
    print("[FE] Création des features cytogénétiques...")
    cyto_features = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
        final_df[["ID", "CYTOGENETICS"]].copy()  # On ne passe que ce qui est nécessaire
    )
    # Fusion sécurisée sur 'ID'
    final_df = pd.merge(final_df, cyto_features, on="ID", how="left")

    # --- Étape 4: Enrichissement avec TOUTES les features moléculaires en une seule fois ---
    print("[FE] Création des features moléculaires (risque et charge)...")
    # Regroupons la logique dans une fonction "master" pour l'efficacité
    all_molecular_features = MolecularFeatureExtraction.create_all_molecular_features(
        final_df[["ID"]], molecular_df, data_manager
    )
    if not all_molecular_features.empty:
        final_df = pd.merge(final_df, all_molecular_features, on="ID", how="left")

    # --- Étape 5: Nettoyage et remplissage final ---
    # Cette étape est toujours cruciale
    print("[FE] Nettoyage final post-fusion...")

    # Remplir les NaNs pour les patients sans données moléculaires
    mol_cols = [c for c in all_molecular_features.columns if c != "ID"]
    final_df[mol_cols] = final_df[mol_cols].fillna(0)

    # Supprimer les colonnes brutes qui ont été transformées
    final_df = final_df.drop(columns=["CYTOGENETICS"], errors="ignore")

    print(f"[FE] Feature Engineering terminé. Shape du dataframe : {final_df.shape}")
    missing_percentage = final_df.isnull().sum().sum() / final_df.size * 100
    print(f"[FE] Taux de valeurs manquantes résiduelles : {missing_percentage:.2f}%")

    return final_df


# --- 5. SCRIPT PRINCIPAL ---
def main():
    """Exécute la pipeline de préparation de données de A à Z."""
    oncokb_file_path = "datas/external/cancerGeneList.txt"
    cosmic_path = "datas/external/Cosmic_CancerGeneCensus_v102_GRCh38.tsv"
    data_manager = ExternalDataManager(
        cosmic_path=cosmic_path, oncokb_path=oncokb_file_path
    )

    # --- Configuration des chemins ---
    input_clinical_path = "datas/X_train/clinical_train.csv"
    input_molecular_path = "datas/X_train/molecular_train_filled.csv"
    input_target_path = "datas/target_train.csv"
    input_clinical_test_path = "datas/X_test/clinical_test.csv"
    input_molecular_test_path = "datas/X_test/molecular_test_filled.csv"
    output_dir = "datasets_processed"
    os.makedirs(output_dir, exist_ok=True)
    output_X_train_path = os.path.join(output_dir, "X_train_processed.csv")
    output_X_test_path = os.path.join(output_dir, "X_test_processed.csv")
    output_y_train_path = os.path.join(output_dir, "y_train_processed.csv")

    # --- ÉTAPE 1 & 2: CHARGEMENT ET NETTOYAGE ---
    print("=" * 50)
    print("ÉTAPE 1 & 2: CHARGEMENT ET NETTOYAGE")
    print("=" * 50)
    clinical_train_raw = pd.read_csv(input_clinical_path)
    molecular_train_raw = pd.read_csv(input_molecular_path)
    target_train_raw = pd.read_csv(input_target_path)
    clinical_test_raw = pd.read_csv(input_clinical_test_path)
    molecular_test_raw = pd.read_csv(input_molecular_test_path)

    clinical_train_clean, molecular_train_clean, target_train_clean = (
        clean_and_validate_data(
            clinical_train_raw, molecular_train_raw, target_train_raw
        )
    )
    fake_target_test = pd.DataFrame(
        {"ID": clinical_test_raw["ID"], "OS_YEARS": 1, "OS_STATUS": 0}
    )
    clinical_test_clean, molecular_test_clean, _ = clean_and_validate_data(
        clinical_test_raw, molecular_test_raw, fake_target_test
    )
    train_df = pd.merge(clinical_train_clean, target_train_clean, on="ID", how="inner")
    test_df = clinical_test_clean.copy()

    print("\n[PREP] Regroupement des centres rares...")
    threshold = 40

    # Apprendre les effectifs sur le jeu d'entraînement UNIQUEMENT
    center_counts = train_df["CENTER"].value_counts()
    major_centers = center_counts[center_counts >= threshold].index.tolist()
    rare_centers = center_counts[center_counts < threshold].index.tolist()

    print(f"   -> {len(major_centers)} centres majeurs conservés.")
    print(
        f"   -> {len(rare_centers)} centres rares vont être regroupés en 'CENTER_OTHER'."
    )

    # Appliquer la transformation aux deux datasets (train et test)
    # Pour le train set :
    train_df["CENTER"] = train_df["CENTER"].apply(
        lambda x: x if x in major_centers else "CENTER_OTHER"
    )

    # Pour le test set :
    test_df["CENTER"] = test_df["CENTER"].apply(
        lambda x: x if x in major_centers else "CENTER_OTHER"
    )

    # Afficher les nouveaux effectifs pour vérification
    print("\nEffectifs après regroupement (sur le train set):")
    print(train_df["CENTER"].value_counts())
    # --- ÉTAPE 3: FEATURE ENGINEERING ---
    print("\n" + "=" * 50)
    print("ÉTAPE 3: FEATURE ENGINEERING")
    print("=" * 50)
    X_train_featured = run_feature_engineering(
        train_df, molecular_train_clean, data_manager
    )
    X_test_featured = run_feature_engineering(
        test_df, molecular_test_clean, data_manager
    )

    print("\n" + "=" * 50)
    print("ÉTAPE 4: FINALISATION ET SÉPARATION")
    print("=" * 50)

    test_ids = X_test_featured["ID"].copy()
    train_ids = X_train_featured["ID"].copy()

    y_train_df = X_train_featured[["OS_STATUS", "OS_YEARS"]].copy()

    # Séparer les features de la target dans le set d'entraînement
    X_train_to_process = X_train_featured.drop(
        columns=["OS_STATUS", "OS_YEARS", "ID"], errors="ignore"
    )
    X_test_to_process = X_test_featured.drop(columns=["ID"], errors="ignore")

    # Retirer CENTER des features pour éviter un fort décalage train/test (test = CENTER_OTHER uniquement)
    for df_name, df in [("train", X_train_to_process), ("test", X_test_to_process)]:
        if "CENTER" in df.columns:
            print(
                f"[PREP] Suppression de la colonne CENTER du jeu {df_name} (éviter one-hot dégénéré)."
            )
            df.drop(columns=["CENTER"], inplace=True)

    # Garantir la cohérence des colonnes entre train et test
    train_cols = X_train_to_process.columns
    X_test_to_process = X_test_to_process.reindex(columns=train_cols)

    # --- ÉTAPE 5: PRÉTRAITEMENT COMPLET ---
    print("\n" + "=" * 50)
    print("ÉTAPE 5: PRÉTRAITEMENT (IMPUTATION, SCALING, ENCODING)")
    print("=" * 50)

    preprocessor = get_preprocessing_pipeline(X_train_to_process, "iterative")

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
    X_train_processed_df.insert(0, "ID", train_ids.values)
    X_test_processed_df.insert(0, "ID", test_ids.values)

    # --- ÉTAPE 5.1: SUPPRESSION DES FEATURES À VARIANCE NULLE (sur train OU test) ---
    drop_cols = []
    for col in X_train_processed_df.columns:
        if col == "ID":
            continue
        nunique_train = X_train_processed_df[col].nunique(dropna=False)
        nunique_test = X_test_processed_df[col].nunique(dropna=False)
        if nunique_train <= 1 or nunique_test <= 1:
            drop_cols.append(col)

    if drop_cols:
        print(
            f"[PREPROCESSING] Suppression de {len(drop_cols)} colonnes à variance nulle: {drop_cols[:10]}{'...' if len(drop_cols)>10 else ''}"
        )
        X_train_processed_df.drop(columns=drop_cols, inplace=True, errors="ignore")
        X_test_processed_df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # --- ÉTAPE 6: SAUVEGARDE DES DONNÉES FINALES ET DES ARTEFACTS ---
    print("\n" + "=" * 50)
    print("ÉTAPE 6: SAUVEGARDE DES DATASETS TRAITÉS ET DU PRÉPROCESSEUR")
    print("=" * 50)

    # Sauvegarde des datasets qui seront lus par le script d'entraînement
    output_dir = "datasets_processed"
    os.makedirs(output_dir, exist_ok=True)
    X_train_processed_df.to_csv(
        os.path.join(output_dir, "X_train_processed.csv"), index=False
    )
    X_test_processed_df.to_csv(
        os.path.join(output_dir, "X_test_processed.csv"), index=False
    )
    y_train_df.to_csv(os.path.join(output_dir, "y_train_processed.csv"), index=False)

    # Sauvegarde du préprocesseur ENTRAÎNÉ. C'est lui qui sera utilisé pour la prédiction finale.
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, os.path.join("models", "preprocessor.joblib"))

    print("Pipeline de préparation des données terminée avec succès.")
    print(f"Fichiers finaux prêts dans le dossier '{output_dir}'")


if __name__ == "__main__":
    main()
