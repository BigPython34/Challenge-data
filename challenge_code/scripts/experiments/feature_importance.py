import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# --- Import from your project structure ---
from src.modeling.train import load_training_dataset_csv, get_survival_models

def plot_feature_importances_from_gb(model, feature_names: List[str], model_name: str, top_n: int = 30):
    """
    Génère et sauvegarde un graphique d'importance des features à partir d'un
    modèle GradientBoostingSurvivalAnalysis entraîné.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"❌ ERREUR: Le modèle '{model_name}' ne possède pas l'attribut 'feature_importances_'.")
        return

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    print("\nTop features (importance décroissante):")
    print(feature_importance_df.head(top_n).to_string(index=False))

    feature_importance_df = feature_importance_df.head(top_n)

    plt.figure(figsize=(12, top_n * 0.45))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='rocket')
    
    plt.title(f'Top {top_n} Features les Plus Importantes - {model_name}', fontsize=16)
    plt.xlabel('Importance (Gain moyen)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    save_path = f"reports/feature_importance_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Graphique d'importance des features sauvegardé dans : '{save_path}'")
    plt.close()


def main():
    """
    Script principal pour entraîner un modèle Gradient Boosting et générer son
    graphique d'importance des features.
    """
    print("=" * 80)
    print("  GÉNÉRATION DU RAPPORT D'IMPORTANCE (BASÉ SUR GRADIENT BOOSTING)")
    print("=" * 80)
    
    # --- CONFIGUREZ CE CHEMIN ---
    TRAIN_DATA_PATH = "datasets_processed/X_train_processed.csv"
    # -----------------------------

    print(f"\n[1/3] Chargement des données depuis : '{TRAIN_DATA_PATH}'...")
    try:
        X_train, y_train = load_training_dataset_csv(
            X_train_path=TRAIN_DATA_PATH,
            y_train_path="datasets_processed/y_train_processed.csv"
        )
        if 'ID' in X_train.columns:
            X_train = X_train.drop(columns=['ID'])
        feature_names = X_train.columns.tolist()
        print(f"✓ {len(X_train)} échantillons et {len(feature_names)} features chargés.")
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier de données non trouvé.")
        return

    print("\n[2/3] Entraînement du modèle Gradient Boosting pour l'analyse...")
    # Utiliser les bons paramètres experts que nous avons définis
    gb_model = get_survival_models()['GradientBoosting']
    
    try:
        gb_model.fit(X_train, y_train)
        print("✓ Modèle Gradient Boosting entraîné.")
    except Exception as e:
        print(f"❌ ERREUR lors de l'entraînement du modèle : {e}")
        return

    print("\n[3/3] Génération du graphique d'importance...")
    plot_feature_importances_from_gb(
        model=gb_model,
        feature_names=feature_names,
        model_name="GradientBoosting",
        top_n=203
    )
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()