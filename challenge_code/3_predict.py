#!/usr/bin/env python3
"""
Script 3/3 : Prédictions finales
Charge la pipeline de prédiction complète et l'applique aux données de test.
"""
import os
import joblib
import pandas as pd

# IMPORTANT : Pour que joblib puisse charger les classes personnalisées,
# il faut qu'elles soient définies ou importées ici.
# Le mieux est de les mettre dans un fichier .py et de les importer.


def predict_and_submit():
    print("=== SCRIPT 3/3 : PREDICTIONS FINALES (Pipeline) ===")
    print("=" * 60)

    # --- 1. CHARGEMENT DE LA PIPELINE DE PRÉDICTION COMPLÈTE ---
    print("\n1. Chargement de la pipeline de prédiction...")
    pipeline_path = os.path.join("models", "final_prediction_pipeline.joblib")
    try:
        prediction_pipeline = joblib.load(pipeline_path)
    except FileNotFoundError:
        print(f"ERREUR : Pipeline de prédiction non trouvée à '{pipeline_path}'")
        print("Veuillez d'abord exécuter le script d'entraînement (2_train_model.py).")
        return
    except Exception as e:
        print(f"ERREUR lors du chargement de la pipeline : {e}")
        return

    # --- 2. CHARGEMENT ET PRÉPARATION DES DONNÉES DE TEST ---
    # Nous avons besoin des données de test AVANT le prétraitement (juste après le feature engineering)
    print("\n2. Chargement des données de test (post-feature engineering)...")
    X_test_featured_path = "datasets_featured/X_test_featured.csv"  # Le fichier sauvegardé à l'étape 4.25 de 1_prepare_data.py
    try:
        X_test_featured = pd.read_csv(X_test_featured_path)
        ids = X_test_featured["ID"].copy()
    except FileNotFoundError:
        print(
            f"ERREUR : Fichier de features de test non trouvé à '{X_test_featured_path}'"
        )
        return

    # --- 3. GÉNÉRATION DES PRÉDICTIONS ---
    print("\n3. Génération des prédictions...")
    # La magie opère ici : la pipeline gère tout (imputation, scaling, prédiction)
    predictions = prediction_pipeline.predict(X_test_featured)
    print(f"   Prédictions générées pour {len(predictions)} échantillons.")

    # --- 4. CRÉATION ET SAUVEGARDE DE LA SOUMISSION ---
    print("\n4. Sauvegarde des résultats...")
    submission_df = pd.DataFrame({"ID": ids, "risk_score": predictions})

    os.makedirs("submissions", exist_ok=True)
    submission_path = (
        f"submissions/submission_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    submission_df.to_csv(submission_path, index=False)
    submission_df.to_csv("submissions/latest_submission.csv", index=False)

    print(f"   Fichier de soumission sauvegardé : {submission_path}")
    print("\n" + "=" * 60)
    print("SCRIPT 3/3 TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)


if __name__ == "__main__":
    predict_and_submit()
