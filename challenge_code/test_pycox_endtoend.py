#!/usr/bin/env python3
"""
Script de test pour entraîner et tester PyCox DeepSurv en une seule session
"""
import pickle
import pandas as pd
import numpy as np
from src.data.prepare import prepare_enriched_dataset, prepare_test_dataset
from src.modeling.train import train_pycox_deepsurv_model
from src.config import PYCOX_DEEPSURV_PARAMS


def test_pycox_end_to_end():
    """Test end-to-end de PyCox DeepSurv sans sérialisation problématique"""
    print("=== TEST END-TO-END PYCOX DEEPSURV ===")
    print("Entraînement et test en une seule session")
    print("=" * 50)

    # 1. Charger le dataset
    print("\n1. Chargement des données...")
    dataset_path = "datasets/training_dataset.pkl"
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    raw_data = dataset["raw_data"]
    imputer = dataset["imputer"]
    features = dataset["features"]

    print(f"   Données chargées : {X_train.shape[0]} train, {X_test.shape[0]} test")

    # 2. Entraîner un modèle PyCox DeepSurv avec des paramètres rapides
    print("\n🧠 2. Entraînement PyCox DeepSurv (test rapide)...")
    try:
        # Paramètres rapides pour le test
        pycox_model = train_pycox_deepsurv_model(
            X_train, y_train, X_test, y_test, **PYCOX_DEEPSURV_PARAMS
        )
        print("   Modèle PyCox DeepSurv entraîné avec succès")
        print(f"   Type du modèle : {type(pycox_model)}")
    except Exception as e:
        print(f"   Erreur lors de l'entraînement : {e}")
        return None

    # 3. Préparer les données de test pour prédiction
    print("\n🔧 3. Préparation des données de test...")
    try:
        df_test_enriched = prepare_enriched_dataset(
            raw_data["clinical_test"],
            raw_data["molecular_test"],
            None,
            imputer=imputer,
            advanced_imputation_method="iterative_ensemble",
            is_training=False,
        )

        df_enriched = dataset["df_enriched"]
        center_columns_train = [
            col for col in df_enriched.columns if col.startswith("center_")
        ]

        X_test_final = prepare_test_dataset(
            df_test_enriched, features, center_columns_train
        )
        print(f"   Données de test préparées : {X_test_final.shape}")
    except Exception as e:
        print(f"   Erreur lors de la préparation : {e}")
        return None

    # 4. Test des prédictions directement (sans sérialisation)
    print("\n🔮 4. Test des prédictions PyCox DeepSurv...")
    try:
        # Tester la prédiction directe
        predictions = pycox_model.predict(X_test_final)

        print(f"   Prédictions générées : {len(predictions)} échantillons")
        print(f"   Type des prédictions : {type(predictions)}")
        f"   Shape : {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}"

        # Statistiques des prédictions
        if hasattr(predictions, "min"):
            print(f"   Min: {predictions.min():.3f}")
            print(f"   Max: {predictions.max():.3f}")
            print(f"   Moyenne: {predictions.mean():.3f}")
        else:
            # Si c'est un array numpy
            pred_array = np.array(predictions)
            print(f"   Min: {pred_array.min():.3f}")
            print(f"   Max: {pred_array.max():.3f}")
            print(f"   Moyenne: {pred_array.mean():.3f}")

    except Exception as e:
        print(f"   Erreur lors des prédictions : {e}")
        return None

    # 5. Créer un fichier de soumission de test
    print("\n💾 5. Sauvegarde du test...")
    try:
        submission_df = pd.DataFrame(
            {"ID": raw_data["clinical_test"]["ID"], "risk_score": predictions}
        )

        import os

        os.makedirs("submissions", exist_ok=True)
        test_submission_path = "submissions/test_pycox_deepsurv_endtoend.csv"
        submission_df.to_csv(test_submission_path, index=False)
        print(f"   Test de soumission PyCox sauvé : {test_submission_path}")

        # Résumé rapide
        print(f"\nRésumé du test PyCox DeepSurv :")
        print(f"   • Nombre de prédictions : {len(submission_df)}")
        print(f"   • Min: {submission_df['risk_score'].min():.3f}")
        print(f"   • Max: {submission_df['risk_score'].max():.3f}")
        print(f"   • Moyenne: {submission_df['risk_score'].mean():.3f}")

    except Exception as e:
        print(f"   Erreur lors de la sauvegarde : {e}")
        return None

    print("\n" + "=" * 50)
    print("TEST END-TO-END PYCOX DEEPSURV RÉUSSI !")
    print("Le support PyCox DeepSurv fonctionne correctement")
    print("Prédictions générées sans sérialisation problématique")
    print("=" * 50)

    return {
        "model": pycox_model,
        "predictions": submission_df,
        "submission_file": test_submission_path,
    }


if __name__ == "__main__":
    test_pycox_end_to_end()
