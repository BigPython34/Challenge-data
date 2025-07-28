#!/usr/bin/env python3
"""
Script principal d'optimisation RSF avec reprise automatique

Usage:
    python start_rsf_optimization.py

Le script:
- Reprend automatiquement l'optimisation la plus récente si elle existe
- Démarre une nouvelle optimisation sinon
- Sauvegarde l'étude pour permettre la reprise en cas d'interruption
"""
import pickle
import sys
import os

sys.path.append(".")


def main():
    print("=" * 60)
    print("🔥 OPTIMISATION RANDOM SURVIVAL FOREST 🔥")
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
        )

        print("✅ Module d'optimisation importé")
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        return

    # 3. Configuration par défaut (modifiable ici)
    N_TRIALS = 300  # Nombre total de trials souhaités
    N_SPLITS = 5  # Nombre de folds pour la validation croisée

    print(f"\n2. Configuration:")
    print(f"   📊 Trials: {N_TRIALS}")
    print(f"   🔄 CV Folds: {N_SPLITS}")
    print(f"   🤖 Reprise automatique: Activée")

    # 4. Demander confirmation
    print(f"\n3. Prêt à lancer l'optimisation...")
    response = input("Continuer ? (o/n) [o]: ").lower().strip()

    if response and response not in ["o", "oui", "y", "yes"]:
        print("❌ Annulé")
        return

    # 5. Lancer l'optimisation avec reprise automatique
    print(f"\n🚀 Démarrage de l'optimisation...")
    print("-" * 60)

    try:
        best_params, best_score, csv_path = resume_or_start_rsf_optimization(
            X_train,
            y_train,
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            auto_resume=True,  # Reprise automatique
        )

        # 6. Afficher les résultats finaux
        print("-" * 60)
        print("🎉 OPTIMISATION TERMINÉE !")
        print(f"🏆 Meilleur score: {best_score:.4f}")
        print("🔧 Meilleurs paramètres:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")

        if csv_path:
            print(f"📊 Résultats détaillés: {os.path.basename(csv_path)}")

        print("\n💡 Pour reprendre une optimisation interrompue:")
        print("   python start_rsf_optimization.py")

    except KeyboardInterrupt:
        print(f"\n⏸️  Optimisation interrompue par l'utilisateur")
        print("💡 Pour reprendre là où vous vous êtes arrêté:")
        print("   python start_rsf_optimization.py")

    except Exception as e:
        print(f"\n❌ Erreur durant l'optimisation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
