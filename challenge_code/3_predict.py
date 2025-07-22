#!/usr/bin/env python3
"""
Script 3/3 : Prédictions finales
Charge le modèle entraîné et applique les prédictions aux données de test
"""
import pickle
import os
import joblib
import pandas as pd
from src.data.prepare import prepare_enriched_dataset

# Import pour PyCox DeepSurv si disponible
try:
    from src.modeling.train import PyCoxWrapper

    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False


def predict_and_submit():
    """Applique le modèle entraîné aux données de test et génère les soumissions"""
    print("=== SCRIPT 3/3 : PREDICTIONS FINALES ===")
    print("Objectif : Generer les predictions sur les donnees de test")
    print("=" * 60)

    # 1. Vérification et chargement du modèle et de l'imputer
    print("\n 1. Chargement du modele entraine et de l'imputer...")

    model_package_path = "models/model_package.pkl"

    if not os.path.exists(model_package_path):
        print("ERREUR : Modele entraine introuvable !")
        print(f"   Fichier attendu : {model_package_path}")
        print("   Veuillez d'abord executer : python 2_train_models.py")
        return None

    try:
        # Charger le package modèle complet avec l'imputer
        with open(model_package_path, "rb") as f:
            model_package = pickle.load(f)

        best_model_dict = model_package["best_model"]
        best_model_name = model_package["best_model_name"]
        features = model_package["features"]
        imputer = model_package["imputer"]  # Utiliser l'imputer du modèle

        # Extraire le modèle du dictionnaire
        best_model = best_model_dict["model"]

        print("   Modele charge avec succes")
        print(f"   Modele : {best_model_name}")
        print(f"   Features : {len(features)}")
        print("   Imputer d'entrainement charge")

    except FileNotFoundError:
        print("ERREUR : Fichier modele introuvable")
        return None
    except Exception as e:
        print(f"ERREUR lors du chargement du modele : {e}")
        return None

    # 2. Chargement des données de test
    print("\n 2. Chargement des donnees de test...")

    # Import des fonctions de chargement
    from src.data.load import load_clinical_data, load_molecular_data

    try:
        # Charger directement les données de test
        clinical_test = load_clinical_data(train=False)
        molecular_test = load_molecular_data(train=False)

        print(f"   Donnees cliniques de test : {clinical_test.shape}")
        print(f"   Donnees moleculaires de test : {molecular_test.shape}")

    except Exception as e:
        print(f"ERREUR lors du chargement des donnees de test : {e}")
        print("   Verifiez que les fichiers de test existent dans datas/X_test/")
        return None

    # 3. Préparation des données de test
    print("\n 3. Preparation des donnees de test...")

    try:
        # Préparer les données de test avec l'imputer exact utilisé pendant l'entraînement
        df_test_enriched = prepare_enriched_dataset(
            clinical_test,
            molecular_test,
            None,  # pas de target pour test
            imputer=imputer,  # Utiliser le vrai imputer d'entraînement
            advanced_imputation_method="medical",
            is_training=False,
        )

        # Sauvegarde du dataset enrichi de test au format CSV
        enriched_test_csv = "datasets/enriched_test.csv"
        df_test_enriched.to_csv(enriched_test_csv, index=False)
        print(f"   Dataset enrichi de test exporte : {enriched_test_csv}")

        # Préparer les features de test - utiliser la même logique que l'entraînement
        # Exclure les colonnes de métadonnées
        exclude_cols = ["ID", "OS_YEARS", "OS_STATUS", "CENTER", "CYTOGENETICS"]
        available_features = [
            col for col in df_test_enriched.columns if col not in exclude_cols
        ]

        # Sélectionner seulement les features utilisées pendant l'entraînement
        missing_features = []
        for feature in features:
            if feature not in df_test_enriched.columns:
                # Ajouter les features manquantes avec des zéros
                df_test_enriched[feature] = 0.0
                missing_features.append(feature)

        if missing_features:
            print(
                f"   Features manquantes ajoutées (remplies avec 0) : {len(missing_features)}"
            )

        # Sélectionner exactement les mêmes features que l'entraînement
        X_test_final = df_test_enriched[features]

        # Vérification finale des valeurs manquantes
        nan_count = X_test_final.isnull().sum().sum()
        if nan_count > 0:
            print(
                f"   ATTENTION : {nan_count} valeurs NaN détectées, remplacement par 0"
            )
            X_test_final = X_test_final.fillna(0)

        print(f"   Donnees de test preparees : {X_test_final.shape}")
        print(f"   {X_test_final.shape[0]} echantillons a predire")

    except Exception as e:
        print(f"ERREUR lors de la preparation des donnees de test : {e}")
        return None

    # 4. Génération des prédictions
    print("\n 4. Generation des predictions...")

    try:
        # Prédictions avec le meilleur modèle directement
        # Vérifier si c'est un modèle PyCox (wrapper spécial)
        if hasattr(best_model, "__class__") and "PyCoxWrapper" in str(type(best_model)):
            print("   Modele PyCox DeepSurv detecte - predictions specialisees")
            predictions = best_model.predict(X_test_final)
        else:
            # Modèles scikit-survival standards
            predictions = best_model.predict(X_test_final)

        # Créer le DataFrame de soumission
        submission_df = pd.DataFrame(
            {"ID": clinical_test["ID"], "risk_score": predictions}
        )

        print(f"   Predictions generees : {len(submission_df)} echantillons")
        print(f"   Type du modele : {type(best_model).__name__}")

    except Exception as e:
        print(f"ERREUR lors des predictions : {e}")
        print(f"   Type de modele : {type(best_model)}")
        return None

    # 5. Sauvegarde des résultats
    print("\n 5. Sauvegarde des predictions...")

    # Créer le répertoire de résultats
    os.makedirs("submissions", exist_ok=True)

    try:
        # Sauvegarder les prédictions
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        submission_filename = f"submission_{best_model_name}_{timestamp}.csv"
        submission_path = f"submissions/{submission_filename}"

        submission_df.to_csv(submission_path, index=False)
        print(f"   Predictions sauvegardees : {submission_path}")

        # Sauvegarder aussi une version "latest" pour faciliter l'utilisation
        latest_path = "submissions/latest_submission.csv"
        submission_df.to_csv(latest_path, index=False)
        print(f"   Copie sauvegardee : {latest_path}")

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

        print(f"   Resume sauvegarde : {summary_path}")

    except Exception as e:
        print(f"ERREUR lors de la sauvegarde : {e}")
        return None

    # 6. Génération du résumé final
    print("\n 6. Generation du resume final...")

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

        print(f"   Resume final : {final_summary_path}")

    except Exception as e:
        print(f"   Avertissement resume final : {e}")

    # 7. Statistiques finales
    # 7. Statistiques finales
    print("\n 7. Statistiques des predictions :")
    print(f"   • Nombre total de predictions : {len(submission_df)}")
    print(f"   • Colonnes : {list(submission_df.columns)}")
    print("   • Statistiques des risk_scores :")
    print(f"     - Min: {submission_df['risk_score'].min():.3f}")
    print(f"     - Max: {submission_df['risk_score'].max():.3f}")
    print(f"     - Moyenne: {submission_df['risk_score'].mean():.3f}")
    print(f"     - Mediane: {submission_df['risk_score'].median():.3f}")

    print("\n" + "=" * 60)
    print("SCRIPT 3/3 TERMINE AVEC SUCCES !")
    print("Predictions finales generees")
    print(f"Fichier de soumission : {submission_path}")
    print("PIPELINE COMPLET TERMINE !")
    print("=" * 60)

    return {
        "predictions": submission_df,
        "submission_file": submission_path,
        "model_used": best_model_name,
    }


if __name__ == "__main__":
    predict_and_submit()
