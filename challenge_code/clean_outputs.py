#!/usr/bin/env python3
"""
Script pour nettoyer les fichiers de sortie générés par les tests
"""
import os
import glob


def clean_directory(directory, pattern):
    """Nettoie un répertoire selon un pattern"""
    if not os.path.exists(directory):
        print(f"Le répertoire {directory} n'existe pas")
        return

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        print(f"Aucun fichier à nettoyer dans {directory}")
        return

    print(f"Nettoyage de {len(files)} fichiers dans {directory}...")
    for file in files:
        try:
            os.remove(file)
            print(f"  ✓ Supprimé: {os.path.basename(file)}")
        except Exception as e:
            print(f"  ✗ Erreur lors de la suppression de {file}: {e}")


def main():
    """Nettoie tous les fichiers de sortie"""
    print("=== NETTOYAGE DES FICHIERS DE SORTIE ===")

    # Nettoyer les modèles
    clean_directory("models", "*.pkl")

    # Nettoyer les résultats
    clean_directory("results", "*.csv")

    # Nettoyer les graphiques (si présents)
    clean_directory("results", "*.png")
    clean_directory("results", "*.jpg")
    clean_directory("results", "*.pdf")

    print("=== NETTOYAGE TERMINÉ ===")


if __name__ == "__main__":
    main()
