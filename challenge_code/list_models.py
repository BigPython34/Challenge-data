#!/usr/bin/env python3
"""
Script utilitaire pour lister les modèles disponibles
"""
import os


def list_available_models():
    """Liste tous les modèles disponibles dans le répertoire models/"""
    models_dir = "models"

    if not os.path.exists(models_dir):
        print("Le répertoire 'models' n'existe pas encore.")
        print("Exécutez d'abord : python 2_train_models.py")
        return

    print("=== MODÈLES DISPONIBLES ===")
    print("Répertoire :", os.path.abspath(models_dir))
    print()

    # Lister tous les fichiers .pkl
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

    if not model_files:
        print("Aucun modèle trouvé dans le répertoire.")
        print("Exécutez d'abord : python 2_train_models.py")
        return

    print("Modèles individuels :")
    individual_models = [f for f in model_files if f != "model_package.pkl"]

    if individual_models:
        for i, model_file in enumerate(individual_models, 1):
            model_name = model_file[:-4]  # Enlever l'extension .pkl
            print(f"  {i:2d}. {model_name}")

        print(f"\nPour utiliser un modèle spécifique :")
        print(f"  python 3_predict.py --model <nom_du_modèle>")
        print(f"  ou")
        print(f"  python main.py --step-3 --model <nom_du_modèle>")

        print(f"\nExemple :")
        first_model = individual_models[0][:-4]
        print(f"  python 3_predict.py --model {first_model}")

    if "model_package.pkl" in model_files:
        print(f"\nModèle par défaut (meilleur modèle) :")
        print(f"  model_package.pkl")
        print(f"  Utilisé quand aucun modèle spécifique n'est spécifié")

    print()


if __name__ == "__main__":
    list_available_models()
