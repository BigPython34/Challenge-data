#!/usr/bin/env python3
"""
Script d'optimisation RSF avec paramètres personnalisables

Usage:
    python rsf_optimization_advanced.py

Permet de:
- Configurer le nombre de trials et folds
- Choisir le mode de reprise
- Personnaliser les paramètres d'optimisation
"""
import pickle
import sys
import os

sys.path.append(".")


def get_user_config():
    """Obtenir la configuration de l'utilisateur"""
    print("\n🔧 CONFIGURATION PERSONNALISÉE")
    print("-" * 40)

    # Nombre de trials
    while True:
        try:
            n_trials = input("Nombre de trials (ex: 300) [300]: ").strip()
            n_trials = int(n_trials) if n_trials else 300
            if n_trials > 0:
                break
            print("❌ Le nombre de trials doit être positif")
        except ValueError:
            print("❌ Veuillez entrer un nombre entier")

    # Nombre de folds
    while True:
        try:
            n_splits = input("Nombre de folds CV (ex: 5) [5]: ").strip()
            n_splits = int(n_splits) if n_splits else 5
            if 2 <= n_splits <= 10:
                break
            print("❌ Le nombre de folds doit être entre 2 et 10")
        except ValueError:
            print("❌ Veuillez entrer un nombre entier")

    # Mode de reprise
    print("\nMode de reprise:")
    print("1. Automatique (recommandé)")
    print("2. Nouvelle optimisation (ignore les existantes)")

    while True:
        choice = input("Choix [1]: ").strip()
        if not choice or choice == "1":
            auto_resume = True
            break
        elif choice == "2":
            auto_resume = False
            break
        else:
            print("❌ Choix invalide")

    return n_trials, n_splits, auto_resume


def main():
    print("=" * 60)
    print("🚀 OPTIMISATION RSF AVANCÉE 🚀")
    print("=" * 60)

    # 1. Charger le dataset
    print("\n1. Chargement du dataset...")
    try:
        with open("datasets/training_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        print(f"✅ Dataset chargé: {X_train.shape}")
    except Exception as e:
        print(f"❌ Erreur chargement dataset: {e}")
        return

    # 2. Importer la fonction d'optimisation
    try:
        from src.modeling.optimize_hyperparameters import (
            resume_or_start_rsf_optimization,
            find_latest_study,
        )

        print("✅ Module d'optimisation importé")
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        return

    # 3. Vérifier les optimisations existantes
    latest_study = find_latest_study("rsf")
    if latest_study:
        print(f"\n📁 Optimisation existante trouvée: {os.path.basename(latest_study)}")

        # Analyser l'étude existante
        try:
            import joblib

            study_data = joblib.load(latest_study)

            if isinstance(study_data, dict) and "study" in study_data:
                study = study_data["study"]
            else:
                study = study_data

            print(f"   📊 Trials complétés: {len(study.trials)}")
            print(f"   🏆 Meilleur score: {study.best_value:.4f}")

        except Exception as e:
            print(f"   ❌ Erreur lecture étude: {e}")
    else:
        print(f"\n📁 Aucune optimisation existante détectée")

    # 4. Configuration utilisateur
    n_trials, n_splits, auto_resume = get_user_config()

    print(f"\n✅ Configuration finale:")
    print(f"   📊 Trials: {n_trials}")
    print(f"   🔄 CV Folds: {n_splits}")
    print(f"   🤖 Reprise auto: {'Oui' if auto_resume else 'Non'}")

    # 5. Confirmation finale
    print(f"\n🚀 Prêt à lancer l'optimisation...")
    response = input("Continuer ? (o/n) [o]: ").lower().strip()

    if response and response not in ["o", "oui", "y", "yes"]:
        print("❌ Annulé")
        return

    # 6. Lancer l'optimisation
    print(f"\n🔥 Démarrage de l'optimisation...")
    print("-" * 60)

    try:
        best_params, best_score, csv_path = resume_or_start_rsf_optimization(
            X_train,
            y_train,
            n_trials=n_trials,
            n_splits=n_splits,
            auto_resume=auto_resume,
        )

        # 7. Résultats finaux
        print("-" * 60)
        print("🎉 OPTIMISATION TERMINÉE !")
        print(f"🏆 Meilleur score: {best_score:.4f}")
        print("🔧 Meilleurs paramètres:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")

        if csv_path:
            print(f"📊 Résultats: {os.path.basename(csv_path)}")

    except KeyboardInterrupt:
        print(f"\n⏸️  Optimisation interrompue")
        print("💡 Relancez le script pour reprendre")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
