#!/usr/bin/env python3
"""
Script de démarrage rapide pour le projet Challenge Data
Guide l'utilisateur à travers les étapes d'installation et d'exécution
"""
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Affiche la bannière du projet"""
    print("=" * 60)
    print("    CHALLENGE DATA - LEUKEMIA RISK PREDICTION")
    print("=" * 60)
    print("    Prédiction du risque de leucémie avec ML")
    print("    7 modèles de survie avancés")
    print("    Pipeline automatisé complet")
    print("=" * 60)


def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERREUR] Python 3.8+ requis")
        print(f"   Version actuelle: {version.major}.{version.minor}")
        return False

    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_data_files():
    """Vérifie la présence des fichiers de données"""
    data_files = [
        "datas/X_train/clinical_train.csv",
        "datas/X_train/molecular_train.csv",
        "datas/X_test/clinical_test.csv",
        "datas/X_test/molecular_test.csv",
        "datas/target_train.csv",
    ]

    missing = []
    for file_path in data_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        print("[ERREUR] Fichiers de données manquants:")
        for file_path in missing:
            print(f"   • {file_path}")
        return False

    print("[OK] Tous les fichiers de données sont présents")
    return True


def run_command(command, description):
    """Exécute une commande et affiche le résultat"""
    print(f"\n{description}...")
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        print(f" {description} - Succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} - Erreur:")
        print(f"   {e.stderr}")
        return False


def main():
    """Fonction principale du script de démarrage"""
    print_banner()

    print("\nVÉRIFICATIONS PRÉLIMINAIRES")
    print("-" * 40)

    # Vérifications de base
    checks = [
        ("Version Python", check_python_version),
        ("Fichiers de données", check_data_files),
    ]

    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            print(f"\nVérification échouée: {check_name}")
            print("Corrigez les erreurs avant de continuer")
            return 1

    print("\nToutes les vérifications passées!")

    # Menu interactif
    while True:
        print("\nQUE VOULEZ-VOUS FAIRE ?")
        print("=" * 40)
        print("1. Installer les dépendances")
        print("2. Tester la structure du projet")
        print("3. Lancer le pipeline complet")
        print("4. Étapes individuelles:")
        print("   a. Préparer les données")
        print("   b. Entraîner les modèles")
        print("   c. Générer les prédictions")
        print("5. Nettoyer les sorties")
        print("6. Quitter")

        choice = input("\nVotre choix (1-6): ").strip()

        if choice == "1":
            print("\nINSTALLATION DES DÉPENDANCES")
            print("-" * 40)
            success = run_command(
                [sys.executable, "install_deps.py"], "Installation des dépendances"
            )
            if not success:
                print("Essayez manuellement: pip install -r requirements.txt")

        elif choice == "2":
            print("\nTEST DE LA STRUCTURE")
            print("-" * 40)
            run_command(
                [sys.executable, "test_structure.py"], "Test de la structure du projet"
            )

        elif choice == "3":
            print("\nPIPELINE COMPLET")
            print("-" * 40)
            print("Cette opération peut prendre 10-30 minutes")
            confirm = input("Continuer ? (o/N): ").strip().lower()
            if confirm in ["o", "oui", "y", "yes"]:
                run_command(
                    [sys.executable, "main.py"], "Exécution du pipeline complet"
                )
            else:
                print("Opération annulée")

        elif choice == "4":
            print("\nÉTAPES INDIVIDUELLES")
            print("-" * 40)
            step_choice = input("Choisir (a/b/c): ").strip().lower()

            if step_choice == "a":
                run_command(
                    [sys.executable, "1_prepare_data.py"], "Préparation des données"
                )
            elif step_choice == "b":
                run_command(
                    [sys.executable, "2_train_models.py"], "Entraînement des modèles"
                )
            elif step_choice == "c":
                run_command(
                    [sys.executable, "3_predict.py"], "Génération des prédictions"
                )
            else:
                print("Choix invalide")

        elif choice == "5":
            print("\nNETTOYAGE")
            print("-" * 40)
            confirm = (
                input("Supprimer tous les fichiers générés ? (o/N): ").strip().lower()
            )
            if confirm in ["o", "oui", "y", "yes"]:
                run_command(
                    [sys.executable, "clean_outputs.py"], "Nettoyage des sorties"
                )
            else:
                print("Opération annulée")

        elif choice == "6":
            print("\nAu revoir!")
            break

        else:
            print("Choix invalide, veuillez recommencer")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur")
        sys.exit(0)
