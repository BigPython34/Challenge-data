#!/usr/bin/env python3
"""
Script principal - Orchestrateur du pipeline de ML en 3 étapes
Coordonne l'exécution séquentielle des 3 scripts du pipeline
"""
import os
import sys
import subprocess
import time
import argparse


def run_script(script_name, step_number, total_steps, selected_model=None):
    """Exécute un script Python et gère les erreurs"""
    print(f"\n{'='*70}")
    print(f" EXÉCUTION SCRIPT {step_number}/{total_steps} : {script_name}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Exécuter le script
        cmd = [sys.executable, script_name]
        if selected_model:
            cmd += ["--model", selected_model]
        subprocess.run(
            cmd,
            capture_output=False,
            check=True,
            cwd=os.getcwd(),
        )

        end_time = time.time()
        duration = end_time - start_time
        print(f"\n Script {script_name} terminé avec succès")
        print(f"  Durée d'exécution : {duration:.2f} secondes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n ERREUR dans {script_name}")
        print(f"Code de retour : {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n ERREUR : Script {script_name} introuvable")
        return False
    except OSError as e:
        print(f"\n ERREUR système dans {script_name} : {e}")
        return False


def check_prerequisites():
    """Vérifie que tous les scripts sont présents"""
    scripts = ["1_prepare_data.py", "2_train_models.py", "3_predict.py"]
    missing_scripts = []

    for script in scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)

    if missing_scripts:
        print(" ERREUR : Scripts manquants :")
        for script in missing_scripts:
            print(f"    {script}")
        return False

    return True


def main(selected_model=None):
    """Pipeline principal complet en 3 étapes"""
    print(" PIPELINE COMPLET DE MACHINE LEARNING ")
    print("=" * 70)
    print("Ce pipeline exécute séquentiellement 3 scripts :")
    print("    1_prepare_data.py   Préparation et sauvegarde du dataset")
    print("    2_train_models.py   Entraînement et sauvegarde des modèles")
    print("    3_predict.py        Génération des prédictions finales")
    print("=" * 70)

    # Vérifier les prérequis
    if not check_prerequisites():
        return False

    start_time_total = time.time()

    # Étape 1 : Préparation des données
    if not run_script("1_prepare_data.py", 1, 3):
        print("\n ARRÊT DU PIPELINE - Échec étape 1")
        return False

    # Étape 2 : Entraînement des modèles
    if not run_script("2_train_models.py", 2, 3, selected_model):
        print("\n ARRÊT DU PIPELINE - Échec étape 2")
        return False

    # Étape 3 : Prédictions finales
    if not run_script("3_predict.py", 3, 3, selected_model):
        print("\n ARRÊT DU PIPELINE - Échec étape 3")
        return False

    # Résumé final
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total

    print("\n" + "" * 35)
    print(" PIPELINE COMPLET TERMINÉ AVEC SUCCÈS ! ")
    print("" * 35)
    print(
        f"  Durée totale : {total_duration:.2f} secondes ({total_duration/60:.1f} minutes)"
    )
    print("\n Fichiers générés :")
    print("   • datasets/training_dataset.pkl      → Dataset préparé")
    print("   • models/model_package.pkl   → Modèles entraînés")
    print("   • submissions/latest_submission.csv  → Prédictions finales")
    print("\n Votre modèle est prêt pour la soumission ! ")
    print("=" * 70)

    return True


def run_step_1():
    """Exécute seulement l'étape 1 : Préparation des données"""
    print("=== EXÉCUTION ÉTAPE 1 UNIQUEMENT ===")
    return run_script("1_prepare_data.py", 1, 1)


def run_step_2(selected_model=None):
    """Exécute seulement l'étape 2 : Entraînement des modèles"""
    print("=== EXÉCUTION ÉTAPE 2 UNIQUEMENT ===")
    return run_script("2_train_models.py", 1, 1, selected_model)


def run_step_3(selected_model=None):
    """Exécute seulement l'étape 3 : Prédictions finales"""
    print("=== EXÉCUTION ÉTAPE 3 UNIQUEMENT ===")
    return run_script("3_predict.py", 1, 1, selected_model)


def run_from_step(step_number, selected_model=None):
    """Exécute le pipeline à partir d'une étape donnée"""
    scripts = ["1_prepare_data.py", "2_train_models.py", "3_predict.py"]

    if step_number < 1 or step_number > 3:
        print(f" ERREUR : Numéro d'étape invalide : {step_number}")
        return False

    print(f"=== EXÉCUTION À PARTIR DE L'ÉTAPE {step_number} ===")

    start_time = time.time()

    for i in range(step_number - 1, 3):
        if i == 1:
            ok = run_script(scripts[i], i + 1, 3, selected_model)
        elif i == 2:
            ok = run_script(scripts[i], i + 1, 3, selected_model)
        else:
            ok = run_script(scripts[i], i + 1, 3)
        if not ok:
            print(f"\n ARRÊT - Échec étape {i + 1}")
            return False

    end_time = time.time()
    duration = end_time - start_time

    print("\n EXÉCUTION TERMINÉE AVEC SUCCÈS !")
    print(f"  Durée : {duration:.2f} secondes")
    return True


if __name__ == "__main__":
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Pipeline de ML pour la LMA")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nom du modèle à utiliser pour l'entraînement/prédiction",
    )
    parser.add_argument(
        "--step-1",
        action="store_true",
        help="Exécuter seulement l'étape 1 (préparation)",
    )
    parser.add_argument(
        "--step-2",
        action="store_true",
        help="Exécuter seulement l'étape 2 (entraînement)",
    )
    parser.add_argument(
        "--step-3",
        action="store_true",
        help="Exécuter seulement l'étape 3 (prédictions)",
    )
    parser.add_argument(
        "--from-step-2", action="store_true", help="Exécuter à partir de l'étape 2"
    )
    parser.add_argument(
        "--from-step-3", action="store_true", help="Exécuter à partir de l'étape 3"
    )

    args = parser.parse_args()

    # Exécution selon les arguments
    if args.step_1:
        run_step_1()
    elif args.step_2:
        run_step_2(args.model)
    elif args.step_3:
        run_step_3(args.model)
    elif args.from_step_2:
        run_from_step(2, args.model)
    elif args.from_step_3:
        run_from_step(3, args.model)
    else:
        # Pipeline complet par défaut
        main(args.model)
