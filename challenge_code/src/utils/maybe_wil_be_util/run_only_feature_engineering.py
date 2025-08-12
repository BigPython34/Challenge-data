# Avant de lancer ce script, assurez-vous que vos fonctions ont été légèrement adaptées :
# - extract_cytogenetic_risk_features doit retourner un df avec la colonne 'ID'.
# - Les fonctions moléculaires pourraient être regroupées en une seule pour l'efficacité.
from src.data.features.feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
    MolecularFeatureExtraction,
    IntegratedFeatureEngineering,
)
import pandas as pd


def run_feature_engineering_revised(
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
    final_df = final_df.drop(columns=["CYTOGENETICS", "CENTER"], errors="ignore")

    print(f"[FE] Feature Engineering terminé. Shape du dataframe : {final_df.shape}")
    missing_percentage = final_df.isnull().sum().sum() / final_df.size * 100
    print(f"[FE] Taux de valeurs manquantes résiduelles : {missing_percentage:.2f}%")

    return final_df


if "__main__" == __name__:

    import pandas as pd

    # Chemins des fichiers
    clinical_path = "datas/X_train/clinical_train.csv"
    molecular_path = "datas/X_train/molecular_train_filled.csv"

    # Chargement des données
    clinical_df = pd.read_csv(clinical_path)
    molecular_df = pd.read_csv(molecular_path)

    # Exécution du feature engineering
    final_df = run_feature_engineering_revised(clinical_df, molecular_df)

    # Sauvegarde du DataFrame final
    final_df.to_csv("datasets_featured/X_train.csv", index=False)
