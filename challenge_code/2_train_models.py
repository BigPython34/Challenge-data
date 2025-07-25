#!/usr/bin/env python3
"""
Script 2/3 : Entraînement des modèles
Charge le dataset préparé, entraîne les modèles et sauvegarde le meilleur
"""
import pickle
import os
import joblib
import pandas as pd
from src.modeling.train import train_and_save_all_models
from src.modeling.evaluate import compare_models
from src.utils.helpers import set_seed
from src.visualization.plots import create_visualization_report


def train_and_save_models():
    """Entraîne tous les modèles et sauvegarde le meilleur"""
    print("=== SCRIPT 2/3 : ENTRAINEMENT DES MODELES ===")
    print("Objectif : Entrainer et sauvegarder les modeles")
    print("=" * 60)

    # 1. Verification and loading of dataset
    print("\n 1. Chargement du dataset prepare...")
    dataset_path = "datasets/training_dataset.pkl"

    if not os.path.exists(dataset_path):
        print("ERREUR : Dataset prepare introuvable !")
        print(f"   Fichier attendu : {dataset_path}")
        print("   Veuillez d'abord executer : python 1_prepare_data.py")
        return None

    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        print("   Dataset charge avec succes")
        print(
            f"   {dataset['metadata']['n_samples_train']} echantillons d'entrainement"
        )
        print(f"   {dataset['metadata']['n_features']} features")
    except Exception as e:
        print(f"ERREUR lors du chargement : {e}")
        return None

    # Extract the data
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    features = dataset["features"]

    # 2. Configuration
    set_seed()

    # 3. Model training
    print("\n 2. Entrainement des modeles...")
    print("   Entrainement en cours (cela peut prendre plusieurs minutes)...")

    try:
        models = train_and_save_all_models(X_train, y_train)
        print(f"   {len(models)} modeles entraines avec succes")
    except Exception as e:
        print(f"ERREUR lors de l'entrainement : {e}")
        return None

    # 4. Evaluation and selection of best model
    print("\n 3. Evaluation des modeles...")

    try:
        results, best_model_name = compare_models(
            models, X_train, y_train, X_test, y_test
        )
        print("   Evaluation terminee")
        print(f"   Meilleur modele : {best_model_name}")

        # Afficher les performances
        print("\n   Performances des modeles :")
        for model_name, metrics in results.items():
            if "test_accuracy" in metrics:
                print(f"      • {model_name}: {metrics['test_accuracy']:.3f}")
    except Exception as e:
        print(f"ERREUR lors de l'evaluation : {e}")
        return None

    # 5. Saving complete model package
    print("\n 4. Sauvegarde du package modele...")

    # Create the models directory
    os.makedirs("models", exist_ok=True)

    # Complete model package
    model_package = {
        "best_model": models[best_model_name],
        "best_model_name": best_model_name,
        "all_models": models,
        "evaluation_results": results,
        "features": features,
        "imputer": dataset["metadata"].get("imputer", None),  # Get imputer if available
        "training_metadata": {
            "training_date": pd.Timestamp.now().isoformat(),
            "n_training_samples": len(X_train),
            "n_features": len(features),
            "best_model_accuracy": results[best_model_name].get("test_accuracy", "N/A"),
            "feature_names": features,
        },
    }

    # Sauvegarder le package complet
    package_path = "models/model_package.pkl"
    with open(package_path, "wb") as f:
        pickle.dump(model_package, f)

    print(f"   Package modele sauvegarde : {package_path}")

    # Also save the best model alone (for quick use)
    best_model_path = "models/best_model.joblib"
    joblib.dump(models[best_model_name], best_model_path)
    print(f"   Meilleur modele sauvegarde : {best_model_path}")

    # Save readable metadata
    metadata_path = "models/model_info.txt"
    with open(metadata_path, "w") as f:
        f.write("=== INFORMATIONS DU MODELE ENTRAINE ===\n")
        f.write(
            f"Date d'entrainement : {model_package['training_metadata']['training_date']}\n"
        )
        f.write(f"Meilleur modele : {best_model_name}\n")
        f.write(f"Precision : {results[best_model_name].get('test_accuracy', 'N/A')}\n")
        f.write(f"Echantillons d'entrainement : {len(X_train)}\n")
        f.write(f"Nombre de features : {len(features)}\n")
        f.write("\nPerformances de tous les modeles :\n")
        for model_name, metrics in results.items():
            if "test_accuracy" in metrics:
                f.write(f"  • {model_name}: {metrics['test_accuracy']:.3f}\n")

    print(f"   Metadonnees sauvegardees : {metadata_path}")
    print(
        f"   Taille du package : {os.path.getsize(package_path) / 1024 / 1024:.2f} MB"
    )

    # 6. Generation of visualization report
    print("\n 5. Generation du rapport de visualisation...")
    try:
        create_visualization_report(models, results, X_test, y_test)
        print("   Rapport de visualisation genere")
    except Exception as e:
        print(f"   Avertissement rapport visualisation : {e}")

    print("\n" + "=" * 60)
    print("SCRIPT 2/3 TERMINE AVEC SUCCES !")
    print(f"Meilleur modele : {best_model_name}")
    print("Modeles prets pour les predictions")
    print("Prochaine etape : python 3_predict.py")
    print("=" * 60)

    return model_package


if __name__ == "__main__":
    train_and_save_models()
