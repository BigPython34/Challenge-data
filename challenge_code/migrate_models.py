#!/usr/bin/env python3
"""
Script de migration pour unifier l'utilisation des modèles
Supprime trained_models/ et migrate tout vers models/
"""
import os
import shutil
from pathlib import Path


def migrate_models():
    """Migre les modèles de trained_models vers models et supprime trained_models"""
    print("🔄 Migration des modèles en cours...")

    trained_models_dir = Path("trained_models")
    models_dir = Path("models")

    # Créer le dossier models s'il n'existe pas
    models_dir.mkdir(exist_ok=True)

    if trained_models_dir.exists():
        print(f"📁 Dossier {trained_models_dir} trouvé")

        # Copier les fichiers utiles vers models/
        for file_path in trained_models_dir.iterdir():
            if file_path.is_file() and file_path.name != ".gitkeep":
                destination = models_dir / file_path.name
                print(f"📄 Migration {file_path.name} → models/")
                shutil.copy2(file_path, destination)

        # Supprimer le dossier trained_models
        print(f"🗑️  Suppression du dossier {trained_models_dir}")
        shutil.rmtree(trained_models_dir)

        print("✅ Migration terminée avec succès")
    else:
        print(f"ℹ️  Dossier {trained_models_dir} non trouvé, rien à migrer")


def update_gitignore():
    """Met à jour le .gitignore pour supprimer les références à trained_models"""
    gitignore_path = Path("..") / ".gitignore"

    if not gitignore_path.exists():
        print("⚠️  .gitignore non trouvé")
        return

    print("📝 Mise à jour du .gitignore...")

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Supprimer les lignes spécifiques à trained_models
    lines_to_remove = [
        "**/trained_models/",
        "challenge_code/trained_models/",
        "challenge_code/trained_models/*.pkl",
        "challenge_code/trained_models/*.joblib",
    ]

    modified = False
    for line_to_remove in lines_to_remove:
        if line_to_remove in content:
            content = content.replace(line_to_remove + "\n", "")
            modified = True
            print(f"🗑️  Supprimé: {line_to_remove}")

    if modified:
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ .gitignore mis à jour")
    else:
        print("ℹ️  Aucune modification nécessaire dans .gitignore")


if __name__ == "__main__":
    print("🚀 Démarrage de la migration des modèles")
    print("=" * 50)

    migrate_models()
    update_gitignore()

    print("=" * 50)
    print("🎉 Migration terminée !")
    print("\n💡 Prochaines étapes:")
    print("   1. Vérifiez que tout fonctionne avec: python test_structure.py")
    print("   2. Committez les changements si tout est OK")
