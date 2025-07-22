#!/usr/bin/env python3
"""
Script pour corriger l'imputer manquant dans le model_package
"""
import pickle
import os


def fix_imputer_in_model_package():
    """Ajoute l'imputer manquant au model package"""

    # Charger les données
    dataset_path = "datasets/training_dataset.pkl"
    model_package_path = "models/model_package.pkl"

    if not os.path.exists(dataset_path):
        print("ERREUR : Dataset d'entraînement introuvable")
        return False

    if not os.path.exists(model_package_path):
        print("ERREUR : Model package introuvable")
        return False

    try:
        # Charger l'imputer depuis le dataset
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        imputer = dataset["metadata"]["imputer"]
        print(f"Imputer chargé: {imputer}")

        # Charger le model package
        with open(model_package_path, "rb") as f:
            model_package = pickle.load(f)

        # Ajouter l'imputer
        model_package["imputer"] = imputer

        # Sauvegarder le package corrigé
        with open(model_package_path, "wb") as f:
            pickle.dump(model_package, f)

        print("Model package mis à jour avec l'imputer")
        return True

    except Exception as e:
        print(f"ERREUR : {e}")
        return False


if __name__ == "__main__":
    fix_imputer_in_model_package()
