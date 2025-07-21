#!/usr/bin/env python3
"""
Script 3/3 : Prédictions finales
Charge le modèle entraîné et applique les prédictions aux données de test
"""
import pickle
import os
import joblib
import pandas as pd
from src.data.prepare import prepare_enriched_dataset, prepare_test_dataset


def predict_and_submit():
    """Applique le modèle entraîné aux données de test et génère les soumissions"""
    print("=== SCRIPT 3/3 : PRÉDICTIONS FINALES ===")
    print("Objectif : Générer les prédictions sur les données de test")
    print("=" * 60)

    # 1. Vérification et chargement du modèle
    print("\n🤖 1. Chargement du modèle entraîné...")
    model_package_path = "trained_models/model_package.pkl"

    if not os.path.exists(model_package_path):
        print("❌ ERREUR : Modèle entraîné introuvable !")
        print(f"   Fichier attendu : {model_package_path}")
        print("   ➡️  Veuillez d'abord exécuter : python 2_train_models.py")
        return None

    try:
        with open(model_package_path, "rb") as f:
            model_package = pickle.load(f)

        best_model_dict = model_package["best_model"]
        best_model_name = model_package["best_model_name"]
        features = model_package["features"]
        imputer = model_package["imputer"]

        # Extraire le modèle du dictionnaire
        best_model = best_model_dict["model"]

        print("   ✅ Modèle chargé avec succès")
        print(f"   🏆 Modèle : {best_model_name}")
        print(f"   📊 Features : {len(features)}")

    except FileNotFoundError:
        print("❌ ERREUR : Fichier modèle introuvable")
        return None
    except Exception as e:
        print(f"❌ ERREUR lors du chargement du modèle : {e}")
        return None

    # 2. Chargement des données originales
    print("\n📂 2. Chargement des données de test...")
    dataset_path = "datasets/training_dataset.pkl"

    if not os.path.exists(dataset_path):
        print("❌ ERREUR : Dataset original introuvable !")
        print("   ➡️  Veuillez d'abord exécuter : python 1_prepare_data.py")
        return None

    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        raw_data = dataset["raw_data"]
        print("   ✅ Données de test chargées")

    except Exception as e:
        print(f"❌ ERREUR lors du chargement des données : {e}")
        return None

    # 3. Préparation des données de test
    print("\n🔧 3. Préparation des données de test...")

    try:
        # Préparer les données de test avec le même preprocessing
        df_test_enriched = prepare_enriched_dataset(
            raw_data["clinical_test"],
            raw_data["molecular_test"],
            None,  # pas de target pour test
            imputer=imputer,
        )[
            0
        ]  # On récupère seulement le dataframe, pas l'imputer

        # Obtenir les colonnes center depuis les données d'entraînement
        df_enriched = dataset["df_enriched"]
        center_columns_train = [
            col for col in df_enriched.columns if col.startswith("center_")
        ]

        # Préparer les features de test
        X_test_final = prepare_test_dataset(
            df_test_enriched, features, center_columns_train
        )

        print(f"   ✅ Données de test préparées : {X_test_final.shape}")
        print(f"   📊 {X_test_final.shape[0]} échantillons à prédire")

    except Exception as e:
        print(f"❌ ERREUR lors de la préparation des données de test : {e}")
        return None

    # 4. Génération des prédictions
    print("\n🔮 4. Génération des prédictions...")

    try:
        # Prédictions avec le meilleur modèle directement
        predictions = best_model.predict(X_test_final)

        # Créer le DataFrame de soumission
        submission_df = pd.DataFrame(
            {"ID": raw_data["clinical_test"]["ID"], "risk_score": predictions}
        )

        print(f"   ✅ Prédictions générées : {len(submission_df)} échantillons")

    except Exception as e:
        print(f"❌ ERREUR lors des prédictions : {e}")
        return None

    # 5. Sauvegarde des résultats
    print("\n💾 5. Sauvegarde des prédictions...")

    # Créer le répertoire de résultats
    os.makedirs("submissions", exist_ok=True)

    try:
        # Sauvegarder les prédictions
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        submission_filename = f"submission_{best_model_name}_{timestamp}.csv"
        submission_path = f"submissions/{submission_filename}"

        submission_df.to_csv(submission_path, index=False)
        print(f"   ✅ Prédictions sauvegardées : {submission_path}")

        # Sauvegarder aussi une version "latest" pour faciliter l'utilisation
        latest_path = "submissions/latest_submission.csv"
        submission_df.to_csv(latest_path, index=False)
        print(f"   ✅ Copie sauvegardée : {latest_path}")

        # Créer un résumé des prédictions
        summary_path = f"submissions/summary_{timestamp}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=== RÉSUMÉ DES PRÉDICTIONS ===\n")
            f.write(f"Date de génération : {pd.Timestamp.now().isoformat()}\n")
            f.write(f"Modèle utilisé : {best_model_name}\n")
            f.write(f"Nombre de prédictions : {len(submission_df)}\n")
            f.write(f"Fichier de soumission : {submission_filename}\n")
            f.write("\nDistribution des prédictions :\n")
            f.write(str(submission_df["risk_score"].describe()))
            f.write("\n\nAperçu des prédictions :\n")
            f.write(str(submission_df.head(10)))

        print(f"   ✅ Résumé sauvegardé : {summary_path}")

    except Exception as e:
        print(f"❌ ERREUR lors de la sauvegarde : {e}")
        return None

    # 6. Génération du résumé final
    print("\n📊 6. Génération du résumé final...")

    try:
        # Créer un résumé simple
        final_summary = f"""=== RÉSUMÉ FINAL DES PRÉDICTIONS ===
Date de génération : {pd.Timestamp.now().isoformat()}
Modèle utilisé : {best_model_name}
Nombre de prédictions : {len(submission_df)}
Fichier de soumission : {submission_filename}

Statistiques des prédictions :
{submission_df['risk_score'].describe()}

Aperçu des données :
{submission_df.head(10)}
"""

        final_summary_path = "submissions/final_summary.txt"
        with open(final_summary_path, "w", encoding="utf-8") as f:
            f.write(final_summary)

        print(f"   ✅ Résumé final : {final_summary_path}")

    except Exception as e:
        print(f"   ⚠️  Avertissement résumé final : {e}")

    # 7. Statistiques finales
    print("\n📈 7. Statistiques des prédictions :")
    print(f"   • Nombre total de prédictions : {len(submission_df)}")
    print(f"   • Colonnes : {list(submission_df.columns)}")
    print(f"   • Statistiques des risk_scores :")
    print(f"     - Min: {submission_df['risk_score'].min():.3f}")
    print(f"     - Max: {submission_df['risk_score'].max():.3f}")
    print(f"     - Moyenne: {submission_df['risk_score'].mean():.3f}")
    print(f"     - Médiane: {submission_df['risk_score'].median():.3f}")

    print("\n" + "=" * 60)
    print("🎉 SCRIPT 3/3 TERMINÉ AVEC SUCCÈS !")
    print("✅ Prédictions finales générées")
    print(f"📁 Fichier de soumission : {submission_path}")
    print("🏁 PIPELINE COMPLET TERMINÉ !")
    print("=" * 60)

    return {
        "predictions": submission_df,
        "submission_file": submission_path,
        "model_used": best_model_name,
    }


if __name__ == "__main__":
    predict_and_submit()
