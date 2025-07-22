#!/usr/bin/env python3
"""
Script de nettoyage du projet - Supprime tous les fichiers non essentiels
"""

import os
import shutil
from pathlib import Path


def cleanup_project():
    """Nettoyer le projet en supprimant tous les fichiers non essentiels"""

    print(" === NETTOYAGE DU PROJET ===")

    # Fichiers à supprimer (chemins relatifs depuis challenge_code/)
    files_to_remove = [
        # Duplicats et versions obsolètes
        "src/data/features_new.py",
        "src/data/features_backup.py",
        "src/data/prepare_new.py",
        "src/data/prepare_backup.py",
        "1_prepare_data.py.backup",
        "1_prepare_data_corrupted.py",
        "1_prepare_data_fixed.py",
        # Scripts de test/développement
        "test_structure.py",
        "test_structure_new.py",
        "test_new_pipeline.py",
        "test_compatibility.py",
        "test_pycox_endtoend.py",
        "demo_medical_pipeline.py",
        # Scripts utilitaires non essentiels
        "install_deps.py",
        "extract_unique_genes.py",
        "check_model.py",
        "migrate_models.py",
        "project_report.py",
        "train_models.py",
        # Notebooks et documentation de développement
        "ADVANCED_IMPROVEMENTS_REPORT.md",
        "ADVANCED_IMPUTATION_REPORT.md",
        "PYCOX_INTEGRATION_REPORT.md",
        "PREPROCESSING_IMPROVEMENT_REPORT.md",
        "GUIDE_UTILISATEUR.md",
    ]

    removed_count = 0
    error_count = 0

    for file_path in files_to_remove:
        full_path = Path(file_path)

        try:
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                    print(f"✅ Supprimé: {file_path}")
                    removed_count += 1
                elif full_path.is_dir():
                    shutil.rmtree(full_path)
                    print(f"✅ Supprimé (dossier): {file_path}")
                    removed_count += 1
            else:
                print(f"⏭️  Déjà absent: {file_path}")
        except Exception as e:
            print(f"❌ Erreur pour {file_path}: {e}")
            error_count += 1

    # Supprimer le dossier notebooks s'il est vide
    notebooks_dir = Path("notebooks")
    if notebooks_dir.exists() and not any(notebooks_dir.iterdir()):
        notebooks_dir.rmdir()
        print(f"✅ Supprimé dossier vide: notebooks/")
        removed_count += 1

    print(f"\n📊 === RÉSUMÉ DU NETTOYAGE ===")
    print(f"✅ Fichiers supprimés: {removed_count}")
    print(f"❌ Erreurs: {error_count}")
    print(f"🎯 Projet nettoyé et optimisé!")

    # Afficher la structure finale
    print(f"\n📁 === STRUCTURE FINALE ===")
    print("Fichiers essentiels conservés:")
    essential_files = [
        "1_prepare_data.py",
        "2_train_models.py",
        "3_predict.py",
        "requirements.txt",
        "README.md",
        "src/config.py",
        "src/data/features.py",
        "src/data/load.py",
        "src/data/prepare.py",
        "src/modeling/train.py",
        "src/modeling/evaluate.py",
        "src/modeling/predict.py",
        "src/utils/helpers.py",
        "src/visualization/plots.py",
    ]

    for file_path in essential_files:
        status = "✅" if Path(file_path).exists() else "❌"
        print(f"  {status} {file_path}")


if __name__ == "__main__":
    cleanup_project()
