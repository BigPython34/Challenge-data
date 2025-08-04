# 5_ensemble_predictions.py

import pandas as pd
from scipy.stats import rankdata
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
import os


def create_and_evaluate_ensemble(rsf_preds_path, xgb_preds_path, target_path=None):
    """
    Crée un ensemble de rangs à partir de deux fichiers de prédiction et l'évalue.

    Parameters
    ----------
    rsf_preds_path : str
        Chemin vers le fichier CSV de prédictions du Random Survival Forest.
        Attend les colonnes 'ID' et 'risk_score' (où score = temps de survie).
    xgb_preds_path : str
        Chemin vers le fichier CSV de prédictions du Gradient Boosting.
        Attend les colonnes 'ID' et 'risk_score' (où score = risque).
    target_path : str, optional
        Chemin vers le fichier CSV des cibles réelles pour évaluation.
        Attend les colonnes 'OS_STATUS' et 'OS_YEARS'.
    """
    print("=== SCRIPT 5/5 : ENSEMBLING DE RANGS ET ÉVALUATION ===")

    # --- 1. Chargement des prédictions ---
    print("\n1. Chargement des fichiers de prédiction...")
    try:
        df_rsf = pd.read_csv(rsf_preds_path)
        df_xgb = pd.read_csv(xgb_preds_path)
        print(f"   Prédictions RSF chargées : {len(df_rsf)} lignes")
        print(f"   Prédictions XGB chargées : {len(df_xgb)} lignes")
    except FileNotFoundError as e:
        print(f"ERREUR : Fichier de prédiction introuvable : {e}")
        return

    # --- 2. Fusion et création des rangs ---
    print("\n2. Création de l'ensemble de rangs...")
    # Fusionner sur l'ID pour s'assurer de l'alignement
    df_merged = pd.merge(df_rsf, df_xgb, on="ID", suffixes=("_rsf", "_xgb"))

    # Pour RSF, un score plus HAUT = plus de survie (moins de risque). On inverse le score.
    df_merged["rank_rsf"] = rankdata(-df_merged["risk_score_rsf"], method="average")

    # Pour XGB, un score plus HAUT = plus de risque. On n'inverse pas.
    df_merged["rank_xgb"] = rankdata(df_merged["risk_score_xgb"], method="average")

    # Le score d'ensemble est la moyenne des rangs
    df_merged["ensemble_score"] = (df_merged["rank_rsf"] + df_merged["rank_xgb"]) / 2.0
    print("   Ensemble de rangs calculé avec succès.")

    # --- 3. Création du fichier de soumission final ---
    submission_df = df_merged[["ID", "ensemble_score"]].rename(
        columns={"ensemble_score": "risk_score"}
    )

    os.makedirs("submissions", exist_ok=True)
    submission_path = "submissions/submission_ensemble.csv"
    submission_df.to_csv(submission_path, index=False)
    print(
        f"\n3. Fichier de soumission pour l'ensemble sauvegardé dans : {submission_path}"
    )
    print("   Statistiques du score d'ensemble :")
    print(submission_df["risk_score"].describe())

    # --- 4. Évaluation (si les cibles sont fournies) ---
    if target_path and os.path.exists(target_path):
        print("\n4. Évaluation du score de l'ensemble...")
        df_target = pd.read_csv(target_path)

        # Préparer les données pour le C-Index
        y_true_sksurv = Surv.from_arrays(
            event=df_target["OS_STATUS"], time=df_target["OS_YEARS"]
        )
        # Pour IPCW, on a besoin des données d'entraînement pour estimer la censure
        y_train_df = pd.read_csv("datasets_processed/y_train_processed.csv")
        y_train_sksurv = Surv.from_arrays(
            event=y_train_df["OS_STATUS"], time=y_train_df["OS_YEARS"]
        )

        # L'évaluation se fait sur le score de risque. Pour le C-Index, un score plus HAUT doit signifier un risque PLUS ÉLEVÉ.
        # Notre 'ensemble_score' (moyenne de rangs) respecte déjà cela.
        ensemble_c_index, _, _, _, _ = concordance_index_ipcw(
            y_train_sksurv, y_true_sksurv, submission_df["risk_score"]
        )
        print("\n" + "=" * 60)
        print(f"   SCORE C-INDEX IPCW DE L'ENSEMBLE : {ensemble_c_index:.5f}")
        print("=" * 60)
    else:
        print("\n4. Aucune évaluation effectuée (fichier cible non fourni).")


if __name__ == "__main__":
    # Configurez ici les chemins vers vos fichiers de prédictions
    xgb_submission_file = "submissions/submission_gradient_boosting_n_estimators800_learning_rate0.0...ples_leaf30_min_samples_split40_20250804_232557_20250804_233402.csv"  # <-- METTRE LE VRAI NOM DE VOTRE FICHIER RSF
    rsf_submission_file = "submissions/submission_rsf_20250804_233113.csv"  # <-- METTRE LE VRAI NOM DE VOTRE FICHIER XGB

    # Si vous évaluez sur votre jeu de validation, fournissez le fichier cible
    # Laissez à None si vous générez pour le test final du challenge
    validation_target_file = "datasets_processed/y_validation_set.csv"  # <-- Créez ce fichier si vous évaluez sur un split

    create_and_evaluate_ensemble(
        rsf_submission_file, xgb_submission_file, validation_target_file
    )
