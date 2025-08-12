# Fichier: src/visualization/visualize.py
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os


def plot_cv_results(cv_results: Dict[str, List[float]]):
    """
    Crée un boxplot pour comparer la distribution des scores de validation croisée
    pour chaque modèle. C'est bien mieux qu'un simple bar chart des moyennes.
    """
    # Préparer les données pour le plotting
    df_results = pd.DataFrame(cv_results)
    sorted_models = df_results.mean().sort_values(ascending=False).index

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_results[sorted_models], orient="h", palette="viridis")

    plt.title(
        "Distribution des Scores IPCW C-index par Modèle (Validation Croisée)",
        fontsize=16,
    )
    plt.xlabel("IPCW C-index (plus c'est élevé, mieux c'est)", fontsize=12)
    plt.ylabel("Modèle", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    # Sauvegarder le graphique
    plt.savefig("reports/model_comparison_cv.png", dpi=300)
    print(
        "✓ Graphique de comparaison des modèles sauvegardé dans 'reports/model_comparison_cv.png'"
    )
    plt.close()


def plot_feature_importances(
    model, feature_names: List[str], model_name: str, top_n: int = 25
):
    """
    Affiche et sauvegarde l'importance des features pour un modèle donné.
    Gère les modèles à base d'arbres (feature_importances_) et les modèles linéaires (coef_).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Top {top_n} Feature Importances - {model_name} (MDI)"
    elif hasattr(model, "coef_"):
        # Pour les modèles linéaires comme Cox, on prend la valeur absolue des coefficients
        importances = np.abs(model.coef_)
        title = f"Top {top_n} Coefficient Magnitudes - {model_name}"
    else:
        print(
            f"   [AVERTISSEMENT] L'importance des features n'est pas disponible pour le modèle {model_name}."
        )
        return

    # Créer un DataFrame pour un tri facile
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        "importance", ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, top_n * 0.4))  # Ajuster la hauteur dynamiquement
    sns.barplot(
        x="importance", y="feature", data=feature_importance_df, palette="rocket"
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Importance / Magnitude", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    # Sauvegarder le graphique
    plt.savefig(
        f"reports/feature_importance_{model_name}.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"✓ Graphique d'importance des features sauvegardé dans 'reports/feature_importance_{model_name}.png'"
    )
    plt.close()


def generate_post_training_report(
    cv_results: Dict, final_model, feature_names: List[str], model_name: str
):
    """
    Fonction principale qui orchestre la création de tous les graphiques de diagnostic.
    """
    print("\n" + "=" * 80)
    print("  GÉNÉRATION DU RAPPORT DE VISUALISATION POST-ENTRAÎNEMENT")
    print("=" * 80)

    # Créer le dossier de rapports s'il n'existe pas
    os.makedirs("reports", exist_ok=True)

    # 1. Graphique de comparaison des performances de la validation croisée
    print("\n[1/2] Génération du boxplot de comparaison des modèles...")
    plot_cv_results(cv_results)

    # 2. Graphique d'importance des features pour le MEILLEUR modèle final
    print(
        f"\n[2/2] Génération du graphique d'importance pour le meilleur modèle : {model_name}..."
    )
    plot_feature_importances(final_model, feature_names, model_name)

    print("\n" + "=" * 80)
    print("  Rapport de visualisation terminé.")
    print("=" * 80)
