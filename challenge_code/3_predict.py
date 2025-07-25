#!/usr/bin/env python3
"""
Script 3/3 : Prédictions finales
Charge le modèle entraîné et applique les prédictions aux données de test
"""
import pickle
import os
import joblib
import pandas as pd
import numpy as np
import argparse
from src.data.prepare import prepare_enriched_dataset

# Import pour PyCox DeepSurv si disponible
from src.modeling.train import PyCoxWrapper


def predict_and_submit(selected_model=None):
    """Applique le modèle entraîné aux données de test et génère les soumissions

    Parameters:
    -----------
    selected_model : str, optional
        Nom du modèle spécifique à utiliser. Si None, utilise le meilleur modèle.
        Format attendu: nom du fichier sans extension (ex: 'cox_alpha1.0_20250723_120432')
    """
    print("=== SCRIPT 3/3 : PREDICTIONS FINALES ===")
    print("Objectif : Generer les predictions sur les donnees de test")
    print("=" * 60)

    # 1. Verification and loading of model and imputer
    print("\n 1. Chargement du modele entraine et de l'imputer...")

    # Determine which model to use
    if selected_model:
        # Specific model requested
        model_file = f"{selected_model}.pkl"
        model_path = f"models/{model_file}"

        if os.path.exists(model_path):
            print(f"   Utilisation du modèle spécifique : {selected_model}")
            try:
                # Individual models are saved with joblib, not pickle
                individual_model = joblib.load(model_path)

                # For individual models, we must load the imputer separately
                model_package_path = "models/model_package.pkl"
                if os.path.exists(model_package_path):
                    with open(model_package_path, "rb") as f:
                        model_package = pickle.load(f)
                    imputer = model_package["imputer"]
                    features = model_package["features"]
                else:
                    print(
                        "ATTENTION : Package modèle complet introuvable, utilisation d'imputation basique"
                    )
                    imputer = {"columns_imputed": []}
                    # Load features from the enriched training dataset
                    train_enriched_path = "datasets/enriched_train.csv"
                    if os.path.exists(train_enriched_path):
                        train_df = pd.read_csv(train_enriched_path)
                        exclude_cols = [
                            "ID",
                            "OS_YEARS",
                            "OS_STATUS",
                            "CENTER",
                            "CYTOGENETICS",
                        ]
                        features = [
                            col for col in train_df.columns if col not in exclude_cols
                        ]
                    else:
                        features = []

                best_model = individual_model
                best_model_name = selected_model
                best_model_dict = {"model": individual_model}

            except Exception as e:
                print(
                    f"ERREUR lors du chargement du modèle spécifique {selected_model}: {e}"
                )
                print(f"   Type d'erreur: {type(e).__name__}")
                print(f"   Taille du fichier: {os.path.getsize(model_path)} bytes")

                # Essayer de lire les premiers bytes pour diagnostiquer
                try:
                    with open(model_path, "rb") as f:
                        first_bytes = f.read(20)
                    print(f"   Premiers bytes: {first_bytes}")
                except:
                    print("   Impossible de lire les premiers bytes")

                print(
                    "   Le fichier modèle semble corrompu ou dans un format incompatible"
                )
                print("Retour au modèle par défaut...")
                selected_model = None
        else:
            print(
                f"ERREUR : Modèle spécifique {selected_model} introuvable dans {model_path}"
            )
            print("Modèles disponibles :")
            if os.path.exists("models"):
                for file in os.listdir("models"):
                    if file.endswith(".pkl") and file != "model_package.pkl":
                        print(f"   - {file[:-4]}")  # Enlever l'extension .pkl
            print("Retour au modèle par défaut...")
            selected_model = None

    if not selected_model:
        # Use the default model (best model)
        model_package_path = "models/model_package.pkl"

        if not os.path.exists(model_package_path):
            print("ERREUR : Modele entraine introuvable !")
            print(f"   Fichier attendu : {model_package_path}")
            print("   Veuillez d'abord executer : python 2_train_models.py")
            return None

        try:
            # Load the complete model package with imputer
            with open(model_package_path, "rb") as f:
                model_package = pickle.load(f)

            best_model_dict = model_package["best_model"]
            best_model_name = model_package["best_model_name"]
            features = model_package["features"]
            imputer = model_package["imputer"]  # Use the model's imputer

            # Extract the model from the dictionary
            best_model = best_model_dict["model"]

        except FileNotFoundError:
            print("ERREUR : Fichier modele introuvable")
            return None
        except Exception as e:
            print(f"ERREUR lors du chargement du modele : {e}")
            return None

    print("   Modele charge avec succes")
    print(f"   Modele : {best_model_name}")
    print(f"   Features : {len(features)}")
    print("   Imputer d'entrainement charge")

    # 2. Loading test data
    print("\n 2. Chargement des donnees de test...")

    # Import des fonctions de chargement
    from src.data.load import load_clinical_data, load_molecular_data

    try:
        # Load test data directly
        clinical_test = load_clinical_data(train=False)
        molecular_test = load_molecular_data(train=False)

        print(f"   Donnees cliniques de test : {clinical_test.shape}")
        print(f"   Donnees moleculaires de test : {molecular_test.shape}")

    except Exception as e:
        print(f"ERREUR lors du chargement des donnees de test : {e}")
        print("   Verifiez que les fichiers de test existent dans datas/X_test/")
        return None

    # 3. Test data preparation
    print("\n 3. Preparation des donnees de test...")

    try:
        # Utiliser le nouveau systeme de preprocessing
        # Verifier si le fichier enrichi de test existe deja
        enriched_test_path = "datasets/enriched_test.csv"

        if os.path.exists(enriched_test_path):
            print(f"   Chargement du dataset de test pre-enrichi: {enriched_test_path}")
            df_test_enriched = pd.read_csv(enriched_test_path)
        else:
            print(
                "   Dataset de test non trouve, preparation a partir des donnees brutes..."
            )
            # Charger la configuration de preprocessing depuis les metadonnees
            training_dataset_path = "datasets/training_dataset.pkl"
            config = {}

            if os.path.exists(training_dataset_path):
                try:
                    with open(training_dataset_path, "rb") as f:
                        training_data = pickle.load(f)
                        config = training_data.get("config", {})
                except:
                    pass

            if not config:
                # Configuration par defaut si non disponible
                config = {
                    "clinical": ["CYTOGENETICSv3"],
                    "molecular": ["GENE"],
                    "additional": [
                        ["cadd", "phred"],
                        ["dbnsfp", "polyphen2", "hdiv", "score"],
                        ["clinvar", "clinical_significance"],
                        ["gnomad_exome", "af", "af"],
                    ],
                    "outliers": {"threshold": 0.05, "multiplier": 1.5},
                    "feature_engineering": {"logs": ["DEPTH", "VAF"]},
                    "aggregations": {
                        "molecular": {
                            "GENE": {
                                "method": "one_hot",
                                "min_count": 5,
                                "rare_label": "gene_other",
                            },
                            "REF": {"method": "count_bases"},
                            "ALT": {"method": "count_bases"},
                        }
                    },
                }

            # Utiliser les fonctions de preprocessing securisees
            from src.utils.preprocess_safe import (
                safe_parse_cytogenetics_v3,
                safe_add_myvariant_data,
                safe_create_one_hot,
                safe_count_bases_per_id,
                safe_log_transform,
                safe_process_outliers,
            )

            # Preprocessing clinique
            clinical_processed = clinical_test.copy()
            if "CYTOGENETICSv3" in config.get("clinical", []):
                clinical_processed = safe_parse_cytogenetics_v3(
                    clinical_processed, "CYTOGENETICS"
                )

            # Nettoyage clinique
            numeric_cols = clinical_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if clinical_processed[col].isna().sum() > 0:
                    median_val = clinical_processed[col].median()
                    clinical_processed[col].fillna(median_val, inplace=True)

            categorical_cols = clinical_processed.select_dtypes(
                include=["object"]
            ).columns
            for col in categorical_cols:
                if clinical_processed[col].isna().sum() > 0:
                    clinical_processed[col].fillna("unknown", inplace=True)

            # Preprocessing moleculaire
            molecular_processed = molecular_test.copy()

            # MyVariant
            additional_fields = config.get("additional", [])
            if additional_fields:
                molecular_processed = safe_add_myvariant_data(
                    molecular_processed, additional_fields
                )

            # Feature engineering sur les genes
            gene_features = None
            if (
                "GENE" in config.get("molecular", [])
                and "GENE" in molecular_processed.columns
            ):
                gene_config = (
                    config.get("aggregations", {}).get("molecular", {}).get("GENE", {})
                )
                if gene_config.get("method") == "one_hot":
                    gene_features = safe_create_one_hot(
                        molecular_processed,
                        "ID",
                        "GENE",
                        gene_config.get("min_count", 5),
                        gene_config.get("rare_label", "gene_other"),
                    )

            # Comptage des bases
            ref_features = None
            alt_features = None
            if "REF" in molecular_processed.columns:
                ref_features = safe_count_bases_per_id(molecular_processed, "ID", "REF")
            if "ALT" in molecular_processed.columns:
                alt_features = safe_count_bases_per_id(molecular_processed, "ID", "ALT")

            # Transformations log
            log_cols = config.get("feature_engineering", {}).get("logs", [])
            for col in log_cols:
                if col in molecular_processed.columns:
                    molecular_processed[f"{col}_log"] = safe_log_transform(
                        molecular_processed, col
                    )

            # Traitement des outliers
            if "outliers" in config:
                molecular_processed = safe_process_outliers(
                    molecular_processed, **config["outliers"]
                )

            # Agregation au niveau patient
            numeric_cols = molecular_processed.select_dtypes(
                include=[np.number]
            ).columns
            numeric_cols = [c for c in numeric_cols if c != "ID"]

            if len(numeric_cols) > 0:
                molecular_agg = (
                    molecular_processed.groupby("ID")[numeric_cols]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                molecular_agg.columns = ["ID"] + [
                    f"{col}_{stat}" for col, stat in molecular_agg.columns[1:]
                ]
                molecular_agg = molecular_agg.fillna(0)
            else:
                molecular_agg = (
                    molecular_processed[["ID"]].drop_duplicates().reset_index(drop=True)
                )

            # Merger avec les features additionnelles
            if gene_features is not None:
                molecular_agg = molecular_agg.merge(gene_features, on="ID", how="left")
            if ref_features is not None:
                molecular_agg = molecular_agg.merge(ref_features, on="ID", how="left")
            if alt_features is not None:
                molecular_agg = molecular_agg.merge(alt_features, on="ID", how="left")

            molecular_agg = molecular_agg.fillna(0)

            # Merge final
            df_test_enriched = clinical_processed.merge(
                molecular_agg, on="ID", how="inner"
            )
            df_test_enriched = df_test_enriched.fillna(0)

            # Sauvegarder
            df_test_enriched.to_csv(enriched_test_path, index=False)
            print(f"   Dataset de test enrichi sauvegarde: {enriched_test_path}")

        # Prepare test features - use the same logic as training
        # Exclude metadata columns
        exclude_cols = ["ID", "OS_YEARS", "OS_STATUS", "CENTER", "CYTOGENETICS"]
        available_features = [
            col for col in df_test_enriched.columns if col not in exclude_cols
        ]

        # Select only the features used during training
        missing_features = []
        for feature in features:
            if feature not in df_test_enriched.columns:
                # Add missing features with zeros
                df_test_enriched[feature] = 0.0
                missing_features.append(feature)

        if missing_features:
            print(
                f"   Features manquantes ajoutées (remplies avec 0) : {len(missing_features)}"
            )

        # Select exactly the same features as training
        X_test_final = df_test_enriched[features]

        # Final check for missing values
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
        # Utiliser les IDs du dataset enrichi qui correspondent aux predictions
        submission_df = pd.DataFrame(
            {"ID": df_test_enriched["ID"], "risk_score": predictions}
        )

        # Pour les IDs cliniques qui n'ont pas de donnees moleculaires,
        # ajouter des predictions par defaut (moyenne des autres predictions)
        all_clinical_ids = set(clinical_test["ID"])
        predicted_ids = set(submission_df["ID"])
        missing_ids = all_clinical_ids - predicted_ids

        if missing_ids:
            print(f"   {len(missing_ids)} IDs cliniques sans donnees moleculaires")
            default_risk = submission_df["risk_score"].mean()

            missing_df = pd.DataFrame(
                {
                    "ID": list(missing_ids),
                    "risk_score": [default_risk] * len(missing_ids),
                }
            )

            submission_df = pd.concat([submission_df, missing_df], ignore_index=True)
            submission_df = submission_df.sort_values("ID").reset_index(drop=True)

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

        # Tronquer le nom du modèle s'il est trop long pour éviter les erreurs Windows
        model_name_for_file = best_model_name
        if len(model_name_for_file) > 100:  # Limiter à 100 caractères
            model_name_for_file = (
                model_name_for_file[:50] + "..." + model_name_for_file[-47:]
            )

        submission_filename = f"submission_{model_name_for_file}_{timestamp}.csv"
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
            f.write(
                f"Modèle utilisé : {best_model_name}\n"
            )  # Nom complet dans le résumé
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
    # Configuration des arguments en ligne de commande
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
