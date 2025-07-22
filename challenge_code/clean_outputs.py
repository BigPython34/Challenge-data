#!/usr/bin/env python3
"""
Script pour nettoyer les fichiers de sortie générés par le pipeline
Supprime modèles, résultats, datasets enrichis et autres fichiers temporaires
"""
import os
import shutil
from pathlib import Path


def clean_directory(directory, pattern="*", extensions_only=False):
    """Nettoie un répertoire selon un pattern"""
    if not os.path.exists(directory):
        print(f"📁 Le répertoire {directory} n'existe pas")
        return 0

    if extensions_only:
        # Pour les extensions, chercher récursivement
        files = []
        for ext in pattern if isinstance(pattern, list) else [pattern]:
            files.extend(Path(directory).rglob(f"*{ext}"))
    else:
        files = list(Path(directory).glob(pattern))

    if not files:
        print(f"📁 Aucun fichier à nettoyer dans {directory}")
        return 0

    print(f"🧹 Nettoyage de {len(files)} fichiers dans {directory}...")
    cleaned = 0
    for file in files:
        try:
            if file.is_file():
                os.remove(file)
                print(f"  ✅ Supprimé: {file.name}")
                cleaned += 1
            elif file.is_dir() and file.name != ".gitkeep":
                shutil.rmtree(file)
                print(f"  📂 Dossier supprimé: {file.name}")
                cleaned += 1
        except (OSError, PermissionError) as e:
            print(f"  ❌ Erreur lors de la suppression de {file}: {e}")

    return cleaned


def clean_empty_directories():
    """Supprime les dossiers vides (sauf ceux avec .gitkeep)"""
    directories_to_check = ["models", "submissions", "datasets"]

    for directory in directories_to_check:
        if os.path.exists(directory):
            try:
                # Ne supprimer que si le dossier est vide ou ne contient que .gitkeep
                contents = os.listdir(directory)
                if not contents or (len(contents) == 1 and ".gitkeep" in contents):
                    print(f"📁 Dossier {directory} conservé (vide ou avec .gitkeep)")
                else:
                    print(f"📁 Dossier {directory} contient {len(contents)} éléments")
            except (OSError, PermissionError) as e:
                print(f"❌ Erreur lors de la vérification de {directory}: {e}")


def main():
    """Nettoie tous les fichiers de sortie"""
    print("🧹 NETTOYAGE DES FICHIERS DE SORTIE")
    print("=" * 50)

    total_cleaned = 0

    # Nettoyer les modèles
    print("\n🤖 Nettoyage des modèles...")
    total_cleaned += clean_directory(
        "models", ["*.pkl", "*.joblib", "*.h5"], extensions_only=True
    )

    # Nettoyer les soumissions
    print("\n📤 Nettoyage des soumissions...")
    total_cleaned += clean_directory("submissions", "*.csv")
    total_cleaned += clean_directory("submissions", "*.txt")

    # Nettoyer les datasets enrichis
    print("\n📊 Nettoyage des datasets...")
    total_cleaned += clean_directory(
        "datasets", ["*.csv", "*.pkl"], extensions_only=True
    )

    # Nettoyer les résultats/figures (si présents)
    print("\n📈 Nettoyage des visualisations...")
    total_cleaned += clean_directory(
        ".", ["*.png", "*.jpg", "*.pdf"], extensions_only=True
    )

    # Nettoyer les fichiers cache Python
    print("\n🐍 Nettoyage des caches Python...")
    total_cleaned += clean_directory(".", "__pycache__")
    total_cleaned += clean_directory("src", "__pycache__")

    # Vérifier les dossiers vides
    print("\n📁 Vérification des dossiers...")
    clean_empty_directories()

    print("=" * 50)
    print(f"✅ NETTOYAGE TERMINÉ - {total_cleaned} éléments supprimés")
    print("\n💡 Le projet est maintenant dans un état propre")
    print("   Vous pouvez relancer le pipeline avec: python main.py")


if __name__ == "__main__":
    main()
