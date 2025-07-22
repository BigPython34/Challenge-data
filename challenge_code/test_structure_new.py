#!/usr/bin/env python3
"""
Script de test complet pour vérifier la structure et les fonctionnalités du projet
Teste les imports, les fichiers de données, et les scripts principaux
"""
import sys
import os

# Ajouter le répertoire du projet au path temporairement pour les tests
sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """Test des imports des modules"""
    print("🔍 Test des imports des modules...")

    try:
        from src.config import SEED, IMPORTANT_GENES, MODEL_DIR

        print("  ✅ Config module - OK")

        from src.data.load import load_all_data

        print("  ✅ Data load module - OK")

        from src.data.prepare import clean_target_data

        print("  ✅ Data prepare module - OK")

        from src.data.features import extract_advanced_cytogenetic_features

        print("  ✅ Data features module - OK")

        from src.modeling.train import train_cox_model

        print("  ✅ Modeling train module - OK")

        from src.modeling.evaluate import evaluate_model_cindex

        print("  ✅ Modeling evaluate module - OK")

        from src.modeling.predict import predict_and_save_submission

        print("  ✅ Modeling predict module - OK")

        from src.utils.helpers import set_seed

        print("  ✅ Utils helpers module - OK")

        from src.visualization.plots import plot_feature_importances

        print("  ✅ Visualization plots module - OK")

        print("\n🎉 Tous les modules sont importables correctement!")
        return True

    except ImportError as e:
        print(f"\n❌ Erreur d'import: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Erreur inattendue lors des imports: {e}")
        return False


def test_data_files():
    """Test de la présence des fichiers de données"""
    print("\n📂 Test des fichiers de données...")

    required_files = [
        "datas/X_train/clinical_train.csv",
        "datas/X_train/molecular_train.csv",
        "datas/X_test/clinical_test.csv",
        "datas/X_test/molecular_test.csv",
        "datas/target_train.csv",
    ]

    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} - {size:,} bytes")
        else:
            print(f"  ❌ {file_path} - MANQUANT")
            all_present = False

    return all_present


def test_scripts():
    """Test de la présence des scripts principaux"""
    print("\n🐍 Test des scripts principaux...")

    required_scripts = [
        "1_prepare_data.py",
        "2_train_models.py",
        "3_predict.py",
        "main.py",
    ]

    all_present = True
    for script in required_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} - MANQUANT")
            all_present = False

    return all_present


def test_directories():
    """Test de la structure des dossiers"""
    print("\n📁 Test de la structure des dossiers...")

    required_dirs = [
        "src",
        "src/data",
        "src/modeling",
        "src/utils",
        "src/visualization",
        "datas",
        "datas/X_train",
        "datas/X_test",
    ]

    all_present = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - MANQUANT")
            all_present = False

    # Test des dossiers de sortie (créés automatiquement)
    output_dirs = ["models", "submissions", "datasets"]
    print("\n  📂 Dossiers de sortie (créés automatiquement):")
    for directory in output_dirs:
        if os.path.exists(directory):
            contents = len(os.listdir(directory))
            print(f"    📂 {directory}/ - {contents} fichier(s)")
        else:
            print(f"    📂 {directory}/ - sera créé automatiquement")

    return all_present


def test_configuration():
    """Test de la configuration"""
    print("\n⚙️ Test de la configuration...")

    try:
        from src.config import SEED, IMPORTANT_GENES, MODEL_DIR, TAU

        print(f"  ✅ Seed: {SEED}")
        print(f"  ✅ Tau (années): {TAU}")
        print(f"  ✅ Gènes importants: {len(IMPORTANT_GENES)} gènes")
        print(f"  ✅ Dossier des modèles: {MODEL_DIR}")

        return True
    except Exception as e:
        print(f"  ❌ Erreur de configuration: {e}")
        return False


def main():
    """Fonction principale du test"""
    print("🧪 TEST DE STRUCTURE DU PROJET")
    print("=" * 50)

    tests = [
        ("Imports des modules", test_imports),
        ("Fichiers de données", test_data_files),
        ("Scripts principaux", test_scripts),
        ("Structure des dossiers", test_directories),
        ("Configuration", test_configuration),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name.upper()}")
        print("-" * len(test_name))
        result = test_func()
        results.append((test_name, result))

    # Résumé final
    print("\n" + "=" * 50)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1

    total = len(results)
    print(f"\n📊 Score: {passed}/{total} tests réussis")

    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✨ Le projet est prêt à être exécuté avec: python main.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) ont échoué")
        print("🔧 Vérifiez les erreurs ci-dessus avant d'exécuter le pipeline")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
