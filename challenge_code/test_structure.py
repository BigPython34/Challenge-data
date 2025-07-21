# Script de test rapide pour vérifier que tous les modules fonctionnent
import sys
import os

# Ajouter le répertoire du projet au path temporairement pour les tests
sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """Test des imports des modules"""
    try:
        from src.config import SEED, IMPORTANT_GENES

        print("✓ Config module - OK")

        from src.data.load import load_all_data

        print("✓ Data load module - OK")

        from src.data.prepare import clean_target_data

        print("✓ Data prepare module - OK")

        from src.data.features import extract_advanced_cytogenetic_features

        print("✓ Data features module - OK")

        from src.modeling.train import train_cox_model

        print("✓ Modeling train module - OK")

        from src.modeling.evaluate import evaluate_model_cindex

        print("✓ Modeling evaluate module - OK")

        from src.modeling.predict import predict_and_save_submission

        print("✓ Modeling predict module - OK")

        from src.utils.helpers import set_seed

        print("✓ Utils helpers module - OK")

        from src.visualization.plots import plot_feature_importances

        print("✓ Visualization plots module - OK")

        print("\n🎉 Tous les modules sont importables correctement!")
        return True

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False


def test_config():
    """Test de la configuration"""
    try:
        from src.config import SEED, IMPORTANT_GENES, DATA_DIR

        print(f"✓ SEED: {SEED}")
        print(f"✓ Nombre de gènes importants: {len(IMPORTANT_GENES)}")
        print(f"✓ Répertoire des données: {DATA_DIR}")
        return True
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False


def main():
    """Test principal"""
    print("=== TEST DE LA STRUCTURE MODULAIRE ===\n")

    print("1. Test des imports...")
    imports_ok = test_imports()

    print("\n2. Test de la configuration...")
    config_ok = test_config()

    if imports_ok and config_ok:
        print("\n✅ Tous les tests passent! La structure modulaire est prête.")
        print("\nPour lancer le pipeline complet:")
        print("  python main.py")
        print("\nPour utiliser les notebooks:")
        print("  jupyter notebook notebooks/modular_workflow.ipynb")
    else:
        print("\n❌ Des erreurs ont été détectées. Vérifiez la configuration.")


if __name__ == "__main__":
    main()
