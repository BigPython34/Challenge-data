# Visualization functions (SHAP, importances, etc.)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import shap


def plot_feature_importances(
    importances, feature_names, top_n=20, title="Feature Importances"
):
    """Display feature importances"""
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(
        range(len(indices[:top_n])),
        importances[indices[:top_n]],
        color="b",
        align="center",
    )
    plt.xticks(
        range(len(indices[:top_n])),
        [feature_names[i] for i in indices[:top_n]],
        rotation=90,
    )
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 8)):
    """Display correlation matrix"""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    print(corr_matrix)
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_shap_analysis(model, X_train, X_test, max_samples=100):
    """Analyse SHAP pour l'explainabilite du modele"""
    # Creer un background dataset
    background = X_train.sample(min(max_samples, len(X_train)))

    # Definir une fonction de prediction
    def predict_fn(x):
        return model.predict(x)

    # Creer l'explainer et calculer les valeurs SHAP
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test.iloc[:max_samples])

    # Visualisations
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test.iloc[:max_samples], show=False)
    plt.tight_layout()
    plt.show()

    return shap_values


def plot_model_comparison(results, metric="test_cindex"):
    """Compare les performances des modeles"""
    models = list(results.keys())
    scores = [results[model][metric] for model in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=["skyblue", "lightcoral", "lightgreen"])
    plt.title(f"Comparaison des modeles - {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45)

    # Ajouter les valeurs sur les barres
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.001,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def plot_predictions_distribution(submissions):
    """Affiche la distribution des predictions pour chaque modele"""
    fig, axes = plt.subplots(1, len(submissions), figsize=(15, 5))
    if len(submissions) == 1:
        axes = [axes]

    for idx, (name, info) in enumerate(submissions.items()):
        submission_df = info["submission_df"]
        axes[idx].hist(submission_df["risk_score"], bins=30, alpha=0.7, color=f"C{idx}")
        axes[idx].set_title(f"Distribution - {name}")
        axes[idx].set_xlabel("Risk Score")
        axes[idx].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def create_visualization_report(models, results, X_train, feature_names=None):
    """Cree un rapport complet de visualisations"""
    print("=== GENERATION DU RAPPORT DE VISUALISATION ===")

    # 1. Comparaison des modeles
    plot_model_comparison(results)

    # 2. Feature importances pour chaque modele
    if feature_names is None:
        feature_names = X_train.columns

    for name, model_info in models.items():
        model = model_info["model"]
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                plot_feature_importances(
                    importances,
                    feature_names,
                    title=f"Feature Importances - {name}",
                )
        except (NotImplementedError, AttributeError):
            print(f"Feature importances non disponibles pour le modele {name}")
            continue

    print("Rapport de visualisation termine!")
