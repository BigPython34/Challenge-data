# run_pipeline.py
# Ce script remplace clean_data.py, impute_data.py, et feature_engineering.py
# Il exécute la pipeline de préparation de données complète dans le bon ordre et en mémoire.

import os
import pandas as pd
import numpy as np

# --- 1. IMPORTATION DE VOTRE LOGIQUE MÉTIER ---
# Assurez-vous que le dossier 'src' est accessible depuis l'endroit où vous lancez ce script
from src.data.data_cleaning.cleaner import clean_and_validate_data
from src.data.features.feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
    MolecularFeatureExtraction,
    IntegratedFeatureEngineering,
)
from src.data.data_cleaning.imputer import AdvancedImputer

# --- 2. IMPORTATION DES OUTILS DE MACHINE LEARNING ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class ClipQuantiles(BaseEstimator, TransformerMixin):
    """Clippe chaque colonne selon des quantiles appris sur le train (par défaut 1e/99e).
    Fonctionne sur DataFrame pour conserver les noms de colonnes dans la pipeline.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        self.columns_ = Xdf.columns
        self.lower_bounds_ = Xdf.quantile(self.lower)
        self.upper_bounds_ = Xdf.quantile(self.upper)
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X, columns=self.columns_)
        Xclipped = Xdf.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return Xclipped.values


# --- 4. FONCTION D'ORCHESTRATION DU FEATURE ENGINEERING ---
def run_feature_engineering(
    clinical_df: pd.DataFrame, molecular_df: pd.DataFrame
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
        final_df[["ID"]], molecular_df
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
    X_train_featured = run_feature_engineering(train_df, molecular_train_clean)
    X_test_featured = run_feature_engineering(test_df, molecular_test_clean)

    # --- ÉTAPE 4: FINALISATION ET SÉPARATION ---
    print("\n" + "=" * 50)
    print("ÉTAPE 4: FINALISATION ET SÉPARATION")
    print("=" * 50)
    test_ids = X_test_featured["ID"].copy()
    train_ids = X_train_featured["ID"].copy()

    y_train_df = X_train_featured[["OS_STATUS", "OS_YEARS"]].copy()

    # On sépare les features numériques des features catégorielles
    numeric_features = X_train_featured.select_dtypes(include=np.number).columns.drop(
        ["OS_STATUS", "OS_YEARS"]
    )
    categorical_features = ["CENTER"]

    # Définir l'espace des features à partir du TRAIN uniquement (évite de "regarder" le test)
    X_train = X_train_featured[list(numeric_features) + categorical_features].copy()

    # S'assurer que le test possède bien les mêmes colonnes (ajoute les colonnes manquantes avec NaN)
    X_test_tmp = X_test_featured.copy()
    # Garantir la présence de la colonne catégorielle
    if "CENTER" not in X_test_tmp.columns:
        X_test_tmp["CENTER"] = "Unknown"
    # Réindexer les colonnes numériques du test sur celles du train
    X_test_num = X_test_tmp.reindex(columns=list(numeric_features))
    # Concaténer avec la/les colonnes catégorielles dans le bon ordre
    X_test = pd.concat([X_test_num, X_test_tmp[categorical_features]], axis=1)

    # Normaliser les centres non vus en train vers NaN pour être imputés en 'Unknown'
    train_centers = (
        set(X_train["CENTER"].dropna().unique())
        if "CENTER" in X_train.columns
        else set()
    )
    if "CENTER" in X_test.columns and len(train_centers) > 0:
        X_test.loc[~X_test["CENTER"].isin(train_centers), "CENTER"] = np.nan

    # --- ÉTAPE 4.5 : NETTOYAGE DES COLONNES VIDES ---
    print("\n[NETTOYAGE FINAL] Vérification des colonnes entièrement vides...")
    cols_to_drop = (
        X_train[numeric_features]
        .columns[X_train[numeric_features].isnull().all()]
        .tolist()
    )
    if cols_to_drop:
        print(
            f"[NETTOYAGE FINAL] Suppression de {len(cols_to_drop)} colonnes numériques vides."
        )
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop, errors="ignore")
        numeric_features = [col for col in numeric_features if col not in cols_to_drop]

    # --- ÉTAPE 5: IMPUTATION & PRÉTRAITEMENT COMPLET ---
    print("\n" + "=" * 50)
    print("ÉTAPE 5: IMPUTATION & PRÉTRAITEMENT COMPLET AVEC COLUMNTRANSFORMER")
    print("=" * 50)

    # Pipeline de prétraitement pour les données numériques
    numeric_transformer = Pipeline(
        steps=[
            ("clip", ClipQuantiles(lower=0.01, upper=0.99)),
            ("imputer", AdvancedImputer(strategy="knn")),
            ("scaler", RobustScaler()),
        ]
    )

    # Pipeline de prétraitement pour les données catégorielles
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Utiliser ColumnTransformer pour appliquer les bonnes étapes aux bonnes colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Entraîner le préprocesseur sur le jeu d'entraînement
    print("[PREPROCESSING] Entraînement du préprocesseur complet...")
    preprocessor.fit(X_train)

    # Transformer les deux jeux de données
    print("[PREPROCESSING] Transformation des données d'entraînement...")
    X_train_processed = preprocessor.transform(X_train)
    print("[PREPROCESSING] Transformation des données de test...")
    X_test_processed = preprocessor.transform(X_test)

    # --- Reconstruire les DataFrames ---
    # On récupère les noms des colonnes après le OneHotEncoding
    ohe_column_names = preprocessor.named_transformers_["cat"][
        "onehot"
    ].get_feature_names_out(categorical_features)
    final_column_names = list(numeric_features) + list(ohe_column_names)

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=final_column_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=final_column_names)

    # Ajouter ID en première colonne pour les deux
    X_train_processed_df.insert(0, "ID", train_ids.values)
    X_test_processed_df.insert(0, "ID", test_ids.values)

    # --- PRUNING: variance quasi nulle ---
    print("[PRUNING] Suppression des features à variance quasi nulle...")
    variances = X_train_processed_df.drop(columns=["ID"]).var(axis=0)
    keep_mask_var = variances > 1e-8
    kept_cols = ["ID"] + variances.index[keep_mask_var].tolist()
    dropped_var = [c for c in X_train_processed_df.columns if c not in kept_cols]
    if dropped_var:
        print(f"   -> {len(dropped_var)} colonnes supprimées (variance ~0)")
    X_train_processed_df = X_train_processed_df[kept_cols]
    X_test_processed_df = X_test_processed_df[kept_cols]

    # --- PRUNING: corrélation élevée ---
    print("[PRUNING] Suppression des features fortement corrélées...")
    corr_df = X_train_processed_df.drop(columns=["ID"]).corr().abs()
    upper = np.triu(np.ones(corr_df.shape), k=1).astype(bool)
    to_drop_corr = set()
    for i, col in enumerate(corr_df.columns):
        high_corr = corr_df.columns[(corr_df.values[:, i] > 0.995) & upper[:, i]]
        to_drop_corr.update(high_corr.tolist())
    if to_drop_corr:
        print(f"   -> {len(to_drop_corr)} colonnes supprimées (|r|>0.995)")
        keep_cols_corr = ["ID"] + [
            c
            for c in X_train_processed_df.columns
            if c not in to_drop_corr and c != "ID"
        ]
        X_train_processed_df = X_train_processed_df[keep_cols_corr]
        X_test_processed_df = X_test_processed_df[keep_cols_corr]

    # --- ÉTAPE 6: SAUVEGARDE DES DONNÉES FINALES ---
    print("\n" + "=" * 50)
    print("ÉTAPE 6: SAUVEGARDE DES DATASETS TRAITÉS")
    print("=" * 50)

    # Sauvegarde datasets
    X_train_processed_df.to_csv(output_X_train_path, index=False)
    X_test_processed_df.to_csv(output_X_test_path, index=False)
    y_train_df.to_csv(output_y_train_path, index=False)

    # Sauvegarde des artefacts pour la prédiction
    os.makedirs("models", exist_ok=True)
    try:
        import joblib

        joblib.dump(preprocessor, os.path.join("models", "preprocessor.joblib"))
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder le préprocesseur: {e}")

    final_cols_after_pruning = X_train_processed_df.columns.tolist()
    pd.Series(final_cols_after_pruning).to_csv(
        os.path.join("models", "final_feature_columns.csv"), index=False
    )

    print(
        f"Pipeline terminée. Les fichiers finaux sont sauvegardés dans le dossier '{output_dir}'"
    )


if __name__ == "__main__":
    main()
