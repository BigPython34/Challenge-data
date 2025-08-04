#!/usr/bin/env python3
"""
Script 3/3 : Prédictions finales
Charge le modèle entraîné et applique les prédictions aux données de test
"""
import os
import pickle
import joblib
import pandas as pd
import argparse


def predict_and_submit(selected_model=None):
    """Applique le modèle entraîné aux données de test enrichies et génère la soumission."""
    print("=== SCRIPT 3/3 : PREDICTIONS FINALES ===")
    print("Objectif : Générer les prédictions sur les données de test enrichies")
    print("=" * 60)

    # 1. Chargement du modèle et des features
    print("\n1. Chargement du modèle...")

    if selected_model:
        # Modèle spécifique demandé
        model_file = f"models/{selected_model}.pkl"
        if not os.path.exists(model_file):
            print(f"ERREUR : Modèle {selected_model} introuvable.")
            return None

        print(f"   Chargement du modèle spécifique : {selected_model}")
        best_model = joblib.load(model_file)

        # Charger les features depuis le package principal
        with open("models/model_package.pkl", "rb") as f:
            model_package = pickle.load(f)
        features = model_package["features"]
        best_model_name = selected_model
    else:
        # Utiliser le meilleur modèle par défaut
        print("   Chargement du meilleur modèle par défaut...")
        with open("models/model_package.pkl", "rb") as f:
            model_package = pickle.load(f)

        best_model = model_package["best_model"]["model"]
        features = model_package["features"]
        best_model_name = model_package["best_model_name"]

    print(f"   Modèle chargé : {best_model_name}")
    print(f"   Nombre de features : {len(features)}")

    # 2. Chargement des données de test enrichies
    print("\n2. Chargement des données de test...")

    enriched_test_path = "datasets_processed/X_test_processed.csv"
    if not os.path.exists(enriched_test_path):
        print(f"ERREUR : Dataset enrichi de test introuvable : {enriched_test_path}")
        print("Veuillez d'abord exécuter le script de préparation des données.")
        return None

    df_test_enriched = pd.read_csv(enriched_test_path)
    print(f"   Données de test chargées : {df_test_enriched.shape}")

    # 3. Préparation des features
    print("\n3. Préparation des features...")

    # Ajouter les features manquantes avec des zéros
    missing_features = []
    for feature in features:
        if feature not in df_test_enriched.columns:
            df_test_enriched[feature] = 0.0
            missing_features.append(feature)

    if missing_features:
        print(
            f"   Features manquantes ajoutées (remplies avec 0) : {len(missing_features)}"
        )

    # Sélectionner exactement les mêmes features que l'entraînement
    X_test_final = df_test_enriched[features].fillna(0)
    print(f"   Features préparées : {X_test_final.shape}")

    # 4. Génération des prédictions
    print("\n4. Génération des prédictions...")

    try:
        predictions = best_model.predict(X_test_final)
        print(f"   Prédictions générées pour {len(predictions)} échantillons")
    except Exception as e:
        print(f"ERREUR lors des prédictions : {e}")
        return None

    # 5. Création du DataFrame de soumission
    submission_df = pd.DataFrame(
        {"ID": df_test_enriched["ID"], "risk_score": predictions}
    )

    # 6. Sauvegarde des résultats
    print("\n5. Sauvegarde des résultats...")

    os.makedirs("submissions", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Gérer les noms de modèles trop longs pour Windows
    model_name_for_file = best_model_name
    if len(model_name_for_file) > 100:
        model_name_for_file = (
            model_name_for_file[:50] + "..." + model_name_for_file[-47:]
        )

    submission_filename = f"submission_{model_name_for_file}_{timestamp}.csv"
    submission_path = f"submissions/{submission_filename}"

    # Sauvegarder le fichier de soumission
    submission_df.to_csv(submission_path, index=False)
    submission_df.to_csv("submissions/latest_submission.csv", index=False)

    print(f"   Fichier de soumission : {submission_path}")
    print(f"   Copie latest : submissions/latest_submission.csv")

    # 7. Statistiques finales
    print("\n6. Statistiques des prédictions :")
    print(submission_df["risk_score"].describe())

    print("\n" + "=" * 60)
    print("SCRIPT 3/3 TERMINÉ AVEC SUCCÈS !")
    print(f"Fichier de soumission : {submission_path}")
    print("=" * 60)

    return {
        "predictions": submission_df,
        "submission_file": submission_path,
        "model_used": best_model_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génère les prédictions avec un modèle spécifique"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nom du modèle à utiliser (sans extension .pkl). "
        "Ex: cox_alpha1.0_20250723_120432. "
        "Si non spécifié, utilise le meilleur modèle.",
    )

    args = parser.parse_args()

    if args.model:
        print(f"Utilisation du modèle spécifique : {args.model}")
    else:
        print("Utilisation du meilleur modèle par défaut")

    predict_and_submit(selected_model=args.model)
