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

# --- 2. IMPORTATION DES OUTILS DE MACHINE LEARNING ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class AdvancedImputer:
    """
    Imputeur avancé compatible scikit-learn, à utiliser APRÈS le feature engineering.
    """

    def __init__(self, strategy: str = "iterative", n_neighbors: int = 5):
        if strategy not in ["knn", "iterative"]:
            raise ValueError("La stratégie doit être 'knn' ou 'iterative'")
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.imputer_ = None
        self.trained_columns_ = None

    def fit(self, X: pd.DataFrame, y=None):
        print(
            f"\n[IMPUTATION] Entraînement de l'imputeur avec la stratégie '{self.strategy}'..."
        )
        self.trained_columns_ = X.columns.tolist()

        if self.strategy == "knn":
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        elif self.strategy == "iterative":
            estimator = RandomForestRegressor(
                n_estimators=10, random_state=42, n_jobs=-1
            )
            self.imputer_ = IterativeImputer(
                estimator=estimator,
                max_iter=10,
                random_state=42,
                initial_strategy="median",
                imputation_order="ascending",
            )
        self.imputer_.fit(X)
        print("[IMPUTATION] Imputeur entraîné avec succès.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None:
            raise RuntimeError(
                "L'imputeur doit être entraîné avant de transformer des données."
            )

        X_reordered = X[self.trained_columns_]
        print(
            f"[IMPUTATION] Transformation des données ({X_reordered.isna().sum().sum()} valeurs manquantes)..."
        )
        X_imputed_np = self.imputer_.transform(X_reordered)
        X_imputed_df = pd.DataFrame(
            X_imputed_np, columns=self.trained_columns_, index=X.index
        )
        print(
            f"[IMPUTATION] Transformation terminée. Nan restants : {X_imputed_df.isna().sum().sum()}"
        )
        return X_imputed_df


# --- 4. FONCTION D'ORCHESTRATION DU FEATURE ENGINEERING ---
def run_feature_engineering(
    clinical_df: pd.DataFrame, molecular_df: pd.DataFrame
) -> pd.DataFrame:
    """Exécute toutes les étapes de feature engineering de manière robuste avec des fusions explicites."""
    print("\n[FE] Démarrage du Feature Engineering...")

    # --- Étape 4.1: Création du DataFrame de base ---
    # On commence avec les données cliniques, qui définissent notre cohorte de patients.
    # On s'assure que 'ID' est une colonne de type string pour des fusions fiables.
    final_df = clinical_df.copy()
    final_df["ID"] = final_df["ID"].astype(str)

    # --- Étape 4.2: Ajout des features cliniques enrichies ---
    # On utilise une copie pour ne pas modifier l'original pendant la création de features.
    clinical_features = ClinicalFeatureEngineering.create_clinical_features(
        final_df.copy()
    )
    # On garde seulement les nouvelles colonnes créées + l'ID pour la fusion
    new_clinical_cols = [
        col for col in clinical_features.columns if col not in final_df.columns
    ] + ["ID"]
    final_df = pd.merge(
        final_df, clinical_features[new_clinical_cols], on="ID", how="left"
    )

    # --- Étape 4.3: Ajout des features cytogénétiques ---
    cyto_features = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(
        final_df.copy()
    )
    # cyto_features a un index numérique, on lui ajoute l'ID pour la fusion
    cyto_features["ID"] = final_df["ID"].values
    final_df = pd.merge(final_df, cyto_features, on="ID", how="left")

    # --- Étape 4.4: Ajout des features moléculaires ---
    # S'assurer que le df moléculaire a des ID en string
    molecular_df["ID"] = molecular_df["ID"].astype(str)

    # Risk features
    molecular_risk_features = (
        MolecularFeatureExtraction.extract_molecular_risk_features(
            final_df.copy(), molecular_df.copy()
        )
    )
    if not molecular_risk_features.empty:
        final_df = pd.merge(final_df, molecular_risk_features, on="ID", how="left")

    # Burden features
    burden_features = MolecularFeatureExtraction.create_molecular_burden_features(
        molecular_df.copy()
    )
    if not burden_features.empty:
        final_df = pd.merge(final_df, burden_features, on="ID", how="left")

    # --- Étape 4.5: Remplissage logique post-fusion ---
    # Un 'merge' à gauche crée des NaN si un patient n'a pas de données moléculaires.
    # C'est logique de les remplir par 0 (pas de mutation = 0, pas de VAF = 0).
    mol_cols = [
        col
        for col in final_df.columns
        if col.startswith(
            (
                "mut_",
                "vaf_",
                "CEBPA_",
                "TP53_",
                "pathway_",
                "eln_molecular_risk",
                "total_mutations",
            )
        )
    ]
    final_df[mol_cols] = final_df[mol_cols].fillna(0)

    print(f"[FE] Feature Engineering terminé. Shape du dataframe : {final_df.shape}")

    # Vérification du taux de remplissage
    missing_percentage = (
        final_df.isnull().sum().sum() / (final_df.shape[0] * final_df.shape[1]) * 100
    )
    print(f"[FE] Taux de valeurs manquantes après FE : {missing_percentage:.2f}%")

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

    y_train_df = X_train_featured[["OS_STATUS", "OS_YEARS"]].copy()

    # On sépare les features numériques des features catégorielles
    numeric_features = X_train_featured.select_dtypes(include=np.number).columns.drop(
        ["OS_STATUS", "OS_YEARS"]
    )
    categorical_features = [
        "CENTER"
    ]  # La seule feature catégorielle que l'on veut garder

    # On s'assure que les colonnes existent bien dans les deux sets
    numeric_features = list(numeric_features.intersection(X_test_featured.columns))

    X_train = X_train_featured[numeric_features + categorical_features]
    X_test = X_test_featured[numeric_features + categorical_features]

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
            ("imputer", AdvancedImputer(strategy="iterative")),
            ("scaler", StandardScaler()),
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

    # --- ÉTAPE 6: SAUVEGARDE DES DONNÉES FINALES ---
    print("\n" + "=" * 50)
    print("ÉTAPE 6: SAUVEGARDE DES DATASETS TRAITÉS")
    print("=" * 50)

    # Note: la sortie est maintenant un array numpy, on peut le sauvegarder avec np.save
    # ou le reconvertir en DataFrame si on reconstruit les noms des colonnes

    # On récupère les noms des colonnes après le OneHotEncoding
    ohe_column_names = preprocessor.named_transformers_["cat"][
        "onehot"
    ].get_feature_names_out(categorical_features)
    final_column_names = numeric_features + list(ohe_column_names)

    X_train_processed_df = pd.DataFrame(X_train_processed, columns=final_column_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=final_column_names)
    X_test_processed_df.insert(0, "ID", test_ids.values)
    X_train_processed_df.to_csv(output_X_train_path, index=False)
    X_test_processed_df.to_csv(output_X_test_path, index=False)
    y_train_df.to_csv(output_y_train_path, index=False)

    print(
        f"Pipeline terminée. Les fichiers finaux sont sauvegardés dans le dossier '{output_dir}'"
    )


if __name__ == "__main__":
    main()
