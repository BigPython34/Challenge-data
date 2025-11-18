import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import seaborn as sns
import os
from sklearn.inspection import permutation_importance


from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================



INPUT_DIR = (
    "datasets_processed"
)

X_TRAIN_FILENAME = "X_train_processed.csv"
Y_TRAIN_FILENAME = "y_train_processed.csv"


OUTPUT_DIR = "reports/feature_importance"
# --------------------------------------------------



ID_COLUMN = "ID"
GROUP_COLUMN = "CENTER_GROUP"

# Noms des colonnes cibles dans y_train
EVENT_STATUS_COL = "OS_STATUS"
EVENT_TIME_COL = "OS_YEARS"


MODELS_TO_RUN = ["rsf"]


N_TOP_FEATURES = 150


# ==============================================================================
# 2. FONCTIONS UTILITAIRES
# ==============================================================================


def plot_feature_importance(importances: pd.Series, model_name: str, output_dir: str):
    """Génère et sauvegarde un graphique en barres de l'importance des features."""

    top_features = importances.head(N_TOP_FEATURES)

    plt.figure(figsize=(12, N_TOP_FEATURES * 0.4))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")

    plt.title(
        f"Top {N_TOP_FEATURES} Features les plus importantes - Modèle {model_name.upper()}",
        fontsize=16,
    )
    plt.xlabel("Importance (Permutation)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Sauvegarde du graphique
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def plot_cox_coefficients(coefficients: pd.Series, output_dir: str):
    """Génère un graphique spécifique pour les coefficients du modèle de Cox."""

    pos_coefs = (
        coefficients[coefficients > 0]
        .sort_values(ascending=False)
        .head(N_TOP_FEATURES // 2)
    )
    neg_coefs = (
        coefficients[coefficients < 0]
        .sort_values(ascending=True)
        .head(N_TOP_FEATURES // 2)
    )


    plot_data = pd.concat([pos_coefs, neg_coefs]).sort_values(ascending=False)

    colors = ["red" if c > 0 else "blue" for c in plot_data.values]

    plt.figure(figsize=(14, N_TOP_FEATURES * 0.4))
    sns.barplot(x=plot_data.values, y=plot_data.index, palette=colors)

    plt.title(f"Features les plus importantes - Modèle de Cox", fontsize=16)
    plt.xlabel("Coefficient (Log Hazard Ratio)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "feature_importance_cox.png")
    plt.savefig(save_path, dpi=300)
    print(f"Graphique sauvegardé : {save_path}")
    plt.close()  # Ferme la figure


# ==============================================================================
# 3. SCRIPT PRINCIPAL
# ==============================================================================


def main():

    print("=" * 60)
    print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("=" * 60)
    try:
        X = pd.read_csv(os.path.join(INPUT_DIR, X_TRAIN_FILENAME))
        y_df = pd.read_csv(os.path.join(INPUT_DIR, Y_TRAIN_FILENAME))
    except FileNotFoundError as e:
        print(f"\nERREUR: Fichier non trouvé. {e}")
        return
    print(f"Données chargées. Shape de X_train : {X.shape}")
    print("\n" + "=" * 60)
    print("ÉTAPE 2: PRÉPARATION DES DONNÉES")
    print("=" * 60)
    feature_cols = [col for col in X.columns if col not in [ID_COLUMN, GROUP_COLUMN]]
    X_features = X[feature_cols]
    y_structured = Surv.from_dataframe(EVENT_STATUS_COL, EVENT_TIME_COL, y_df)
    print(f"Préparation terminée. {X_features.shape[1]} features prêtes.")


    for model_name in MODELS_TO_RUN:
        print("\n" + "=" * 60)
        print(f"ÉTAPE 3: ANALYSE AVEC LE MODÈLE {model_name.upper()}")
        print("=" * 60)

        model = None  # Initialisation
        if model_name == "rsf":
            model = RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=15,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42,
            )
        elif model_name == "gbsa":
            model = GradientBoostingSurvivalAnalysis(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
            )
        elif model_name == "cox":
            X_cox = X_features.loc[:, X_features.std() > 0]
            model = CoxPHSurvivalAnalysis(alpha=0.1)

        if model is None:
            print(f"Modèle '{model_name}' non reconnu.")
            continue

        print(f"Entraînement du modèle {model_name.upper()}...")
        X_fit = X_cox if model_name == "cox" else X_features
        model.fit(X_fit, y_structured)


        if model_name == "rsf":
            print("Calcul de l'importance par permutation pour RSF... (peut être long)")
            perm_result = permutation_importance(
                model,
                X_features,
                y_structured,
                n_repeats=5,
                random_state=42,
                n_jobs=2,
            )
            importances = perm_result.importances_mean
            importance_series = pd.Series(importances, index=X_features.columns)

        elif model_name == "gbsa":
            importances = model.feature_importances_
            importance_series = pd.Series(importances, index=X_features.columns)

        elif model_name == "cox":
            importance_series = pd.Series(model.coef_, index=X_cox.columns)
            sorted_importances = importance_series.reindex(
                importance_series.abs().sort_values(ascending=False).index
            )
            print(f"\n--- TOP FEATURES (par magnitude) POUR LE MODÈLE COX ---")
            print(sorted_importances.head(N_TOP_FEATURES))
            plot_cox_coefficients(importance_series, OUTPUT_DIR)
            continue

        # Affichage et plotting pour RSF et GBSA
        sorted_importances = importance_series.sort_values(ascending=False)
        print(
            f"\n--- TOP {N_TOP_FEATURES} FEATURES POUR LE MODÈLE {model_name.upper()} ---"
        )
        print(sorted_importances.head(N_TOP_FEATURES))
        plot_feature_importance(sorted_importances, model_name, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analyse terminée.")
    print("=" * 60)


if __name__ == "__main__":
    main()
