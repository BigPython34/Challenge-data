#!/usr/bin/env python3
"""
Script principal - Orchestrateur du pipeline de ML en 3 étapes
Coordonne l'exécution séquentielle des 3 scripts du pipeline
"""
import os
import sys
import subprocess
import time


def run_script(script_name, step_number, total_steps):
    """Exécute un script Python et gère les erreurs"""
    print(f"\n{'='*70}")
    print(f"🚀 EXÉCUTION SCRIPT {step_number}/{total_steps} : {script_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Exécuter le script
        subprocess.run([sys.executable, script_name], 
                      capture_output=False, 
                      check=True,
                      cwd=os.getcwd())
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Script {script_name} terminé avec succès")
        print(f"⏱️  Durée d'exécution : {duration:.2f} secondes")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERREUR dans {script_name}")
        print(f"Code de retour : {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERREUR : Script {script_name} introuvable")
        return False
    except OSError as e:
        print(f"\n❌ ERREUR système dans {script_name} : {e}")
        return False


def check_prerequisites():
    """Vérifie que tous les scripts sont présents"""
    scripts = ["1_prepare_data.py", "2_train_models.py", "3_predict.py"]
    missing_scripts = []
    
    for script in scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print("❌ ERREUR : Scripts manquants :")
        for script in missing_scripts:
            print(f"   • {script}")
        return False
    
    return True


def main():
    """Pipeline principal complet en 3 étapes"""
    print("🔥 PIPELINE COMPLET DE MACHINE LEARNING 🔥")
    print("=" * 70)
    print("Ce pipeline exécute séquentiellement 3 scripts :")
    print("  1️⃣  1_prepare_data.py  → Préparation et sauvegarde du dataset")
    print("  2️⃣  2_train_models.py  → Entraînement et sauvegarde des modèles") 
    print("  3️⃣  3_predict.py       → Génération des prédictions finales")
    print("=" * 70)
    
    # Vérifier les prérequis
    if not check_prerequisites():
        return False
    
    start_time_total = time.time()
    
    # Étape 1 : Préparation des données
    if not run_script("1_prepare_data.py", 1, 3):
        print("\n💥 ARRÊT DU PIPELINE - Échec étape 1")
        return False
    
    # Étape 2 : Entraînement des modèles
    if not run_script("2_train_models.py", 2, 3):
        print("\n💥 ARRÊT DU PIPELINE - Échec étape 2")
        return False
    
    # Étape 3 : Prédictions finales
    if not run_script("3_predict.py", 3, 3):
        print("\n💥 ARRÊT DU PIPELINE - Échec étape 3")
        return False
    
    # Résumé final
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    
    print("\n" + "🎉" * 35)
    print("� PIPELINE COMPLET TERMINÉ AVEC SUCCÈS ! 🏆")
    print("🎉" * 35)
    print(f"⏱️  Durée totale : {total_duration:.2f} secondes ({total_duration/60:.1f} minutes)")
    print("\n📁 Fichiers générés :")
    print("   • datasets/training_dataset.pkl      → Dataset préparé")
    print("   • models/model_package.pkl   → Modèles entraînés")
    print("   • submissions/latest_submission.csv  → Prédictions finales")
    print("\n✨ Votre modèle est prêt pour la soumission ! ✨")
    print("=" * 70)
    
    return True


def run_step_1():
    """Exécute seulement l'étape 1 : Préparation des données"""
    print("=== EXÉCUTION ÉTAPE 1 UNIQUEMENT ===")
    return run_script("1_prepare_data.py", 1, 1)


def run_step_2():
    """Exécute seulement l'étape 2 : Entraînement des modèles"""
    print("=== EXÉCUTION ÉTAPE 2 UNIQUEMENT ===")
    return run_script("2_train_models.py", 1, 1)


def run_step_3():
    """Exécute seulement l'étape 3 : Prédictions finales"""
    print("=== EXÉCUTION ÉTAPE 3 UNIQUEMENT ===")
    return run_script("3_predict.py", 1, 1)


def run_from_step(step_number):
    """Exécute le pipeline à partir d'une étape donnée"""
    scripts = ["1_prepare_data.py", "2_train_models.py", "3_predict.py"]
    
    if step_number < 1 or step_number > 3:
        print(f"❌ ERREUR : Numéro d'étape invalide : {step_number}")
        return False
    
    print(f"=== EXÉCUTION À PARTIR DE L'ÉTAPE {step_number} ===")
    
    start_time = time.time()
    
    for i in range(step_number - 1, 3):
        if not run_script(scripts[i], i + 1, 3):
            print(f"\n💥 ARRÊT - Échec étape {i + 1}")
            return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n🎉 EXÉCUTION TERMINÉE AVEC SUCCÈS !")
    print(f"⏱️  Durée : {duration:.2f} secondes")
    return True


if __name__ == "__main__":
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) > 1:
        if sys.argv[1] == "--step-1":
            run_step_1()
        elif sys.argv[1] == "--step-2":
            run_step_2()
        elif sys.argv[1] == "--step-3":
            run_step_3()
        elif sys.argv[1].startswith("--from-step-"):
            step_num = int(sys.argv[1].split("-")[-1])
            run_from_step(step_num)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python main.py                  # Pipeline complet (3 étapes)")
            print("  python main.py --step-1         # Étape 1 seulement (préparation)")
            print("  python main.py --step-2         # Étape 2 seulement (entraînement)")
            print("  python main.py --step-3         # Étape 3 seulement (prédictions)")
            print("  python main.py --from-step-2    # À partir de l'étape 2")
            print("  python main.py --from-step-3    # À partir de l'étape 3")
            print("  python main.py --help           # Afficher cette aide")
        else:
            print(f"Option inconnue: {sys.argv[1]}")
            print("Utilisez --help pour voir les options disponibles")
    else:
        # Pipeline complet par défaut
        main()
