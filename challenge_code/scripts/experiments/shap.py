import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap # Assurez-vous d'avoir installé shap : pip install shap

# Fichier: shap.py (uniquement la fonction à changer)

def generate_shap_report(model, X_train: pd.DataFrame, model_name: str, n_background: int = 50, n_explain: int = 100):
    """
    Génère un rapport d'importance des features en utilisant SHAP.
    Utilise un sous-échantillon pour des calculs rapides.
    """
    print("   -> Préparation de l'analyse SHAP (cela peut prendre quelques minutes)...")
    
    # 1. Créer le jeu de données "de fond" (background)
    # --- CORRECTION ICI ---
    # On utilise directement la méthode .sample() de pandas, qui est la bonne pratique.
    if len(X_train) > n_background:
        background_data = X_train.sample(n=n_background, random_state=42)
    else:
        background_data = X_train

    # 2. Créer l'objet Explainer
    try:
        # On passe directement le DataFrame pandas, KernelExplainer sait le gérer.
        explainer = shap.KernelExplainer(model.predict, background_data)
    except Exception as e:
        print(f"❌ ERREUR lors de la création de l'explainer SHAP : {e}")
        return

    # 3. Sélectionner les données à expliquer
    if len(X_train) > n_explain:
        data_to_explain = X_train.sample(n=n_explain, random_state=42)
    else:
        data_to_explain = X_train

    # 4. Calculer les valeurs SHAP
    print(f"   -> Calcul des valeurs SHAP sur {len(data_to_explain)} échantillons...")
    try:
        shap_values = explainer.shap_values(data_to_explain)
    except Exception as e:
        print(f"❌ ERREUR lors du calcul des valeurs SHAP : {e}")
        return
        
    # 5. Générer et sauvegarder le graphique
    print("   -> Génération du graphique SHAP summary plot...")
    
    plt.figure()
    # On utilise plot_type="bar" pour obtenir un classement global des features
    shap.summary_plot(shap_values, data_to_explain, plot_type="bar", show=False)
    
    plt.title(f'Importance Globale des Features (SHAP) - {model_name}', fontsize=16)
    plt.xlabel('Impact moyen sur la prédiction du modèle (valeur SHAP absolue)', fontsize=10)
    plt.tight_layout()
    
    os.makedirs("reports", exist_ok=True)
    save_path = f"reports/shap_importance_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Graphique d'importance SHAP sauvegardé dans : '{save_path}'")
    plt.close()


def main():
    """
    Script principal pour charger un modèle et générer son rapport d'importance SHAP.
    """
    print("=" * 80)
    print("  GÉNÉRATION DU RAPPORT D'IMPORTANCE DES FEATURES (AVEC SHAP)")
    print("=" * 80)
    
    # --- CONFIGUREZ CES CHEMINS ---
    MODEL_PATH = "models/model_RSF.joblib" # Adaptez ce nom de fichier
    TRAIN_DATA_PATH = "datasets_processed/X_train_processed.csv"
    # -----------------------------

    print(f"\n[1/3] Chargement du modèle depuis : '{MODEL_PATH}'...")
    try:
        model = joblib.load(MODEL_PATH)
        model_name = os.path.basename(MODEL_PATH).replace('.joblib', '')
        print("✓ Modèle chargé.")
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier modèle non trouvé. Vérifiez le chemin : '{MODEL_PATH}'")
        return

    print(f"\n[2/3] Chargement des données d'entraînement pour l'analyse SHAP...")
    try:
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        if 'ID' in df_train.columns:
            df_train = df_train.drop(columns=['ID'])
        print(f"✓ {len(df_train)} échantillons chargés.")
    except FileNotFoundError:
        print(f"❌ ERREUR: Fichier de données d'entraînement non trouvé.")
        return

    print("\n[3/3] Lancement de la génération du rapport SHAP...")
    # NOTE: n_background et n_explain sont bas pour un test rapide.
    # Augmentez-les pour une analyse plus précise (ex: 100 et 200).
    generate_shap_report(
        model=model,
        X_train=df_train,
        model_name=model_name,
        n_background=50,
        n_explain=100
    )
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()