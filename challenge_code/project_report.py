#!/usr/bin/env python3
"""
Script de génération de rapport sur l'état du projet
Génère un résumé complet des modèles, performances et fichiers
"""
import os
import pickle
import joblib
from datetime import datetime
from pathlib import Path
import pandas as pd


def get_file_info(file_path):
    """Obtient les informations sur un fichier"""
    if not os.path.exists(file_path):
        return None

    stat = os.stat(file_path)
    return {
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
    }


def analyze_models():
    """Analyse les modèles entraînés"""
    models_dir = Path("models")
    if not models_dir.exists():
        return {"count": 0, "models": []}

    models = []
    total_size = 0

    for model_file in models_dir.glob("*.pkl"):
        info = get_file_info(model_file)
        if info:
            total_size += info["size"]
            models.append(
                {
                    "name": model_file.name,
                    "size_mb": info["size_mb"],
                    "modified": info["modified"],
                }
            )

    for model_file in models_dir.glob("*.joblib"):
        info = get_file_info(model_file)
        if info:
            total_size += info["size"]
            models.append(
                {
                    "name": model_file.name,
                    "size_mb": info["size_mb"],
                    "modified": info["modified"],
                }
            )

    return {
        "count": len(models),
        "models": sorted(models, key=lambda x: x["modified"], reverse=True),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


def analyze_datasets():
    """Analyse les datasets enrichis"""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return {"count": 0, "datasets": []}

    datasets = []
    total_size = 0

    for dataset_file in datasets_dir.glob("*.csv"):
        info = get_file_info(dataset_file)
        if info:
            total_size += info["size"]
            # Essayer de lire quelques infos sur le CSV
            try:
                df = pd.read_csv(dataset_file, nrows=0)  # Juste les headers
                num_cols = len(df.columns)
            except:
                num_cols = "?"

            datasets.append(
                {
                    "name": dataset_file.name,
                    "size_mb": info["size_mb"],
                    "columns": num_cols,
                    "modified": info["modified"],
                }
            )

    for dataset_file in datasets_dir.glob("*.pkl"):
        info = get_file_info(dataset_file)
        if info:
            total_size += info["size"]
            datasets.append(
                {
                    "name": dataset_file.name,
                    "size_mb": info["size_mb"],
                    "columns": "?",  # Pas de colonnes pour les fichiers pkl
                    "modified": info["modified"],
                }
            )

    return {
        "count": len(datasets),
        "datasets": sorted(datasets, key=lambda x: x["modified"], reverse=True),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


def analyze_submissions():
    """Analyse les fichiers de soumission"""
    submissions_dir = Path("submissions")
    if not submissions_dir.exists():
        return {"count": 0, "submissions": []}

    submissions = []
    total_size = 0

    for sub_file in submissions_dir.glob("*.csv"):
        info = get_file_info(sub_file)
        if info:
            total_size += info["size"]

            # Essayer de compter les lignes
            try:
                df = pd.read_csv(sub_file)
                num_rows = len(df)
            except:
                num_rows = "?"

            submissions.append(
                {
                    "name": sub_file.name,
                    "size_mb": info["size_mb"],
                    "rows": num_rows,
                    "modified": info["modified"],
                }
            )

    return {
        "count": len(submissions),
        "submissions": sorted(submissions, key=lambda x: x["modified"], reverse=True)[
            :10
        ],  # Top 10
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


def get_model_performance():
    """Tente de lire les performances des modèles"""
    try:
        info_file = Path("models/model_info.txt")
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                content = f.read()
                return content
    except:
        pass

    return "Aucune information de performance disponible"


def generate_report():
    """Génère un rapport complet"""
    print("📊 RAPPORT D'ÉTAT DU PROJET")
    print("=" * 60)
    print(f"📅 Généré le: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")

    # Analyse des modèles
    print("\n🤖 MODÈLES ENTRAÎNÉS")
    print("-" * 30)
    models_info = analyze_models()
    print(f"📊 Nombre de modèles: {models_info['count']}")
    print(f"💾 Taille totale: {models_info['total_size_mb']} MB")

    if models_info["models"]:
        print("\n📋 Modèles récents:")
        for model in models_info["models"][:5]:  # Top 5
            print(
                f"  • {model['name']} - {model['size_mb']} MB - {model['modified'].strftime('%d/%m %H:%M')}"
            )

    # Analyse des datasets
    print("\n📊 DATASETS ENRICHIS")
    print("-" * 30)
    datasets_info = analyze_datasets()
    print(f"📊 Nombre de datasets: {datasets_info['count']}")
    print(f"💾 Taille totale: {datasets_info['total_size_mb']} MB")

    if datasets_info["datasets"]:
        print("\n📋 Datasets disponibles:")
        for dataset in datasets_info["datasets"]:
            cols_info = (
                f" - {dataset['columns']} colonnes"
                if dataset.get("columns") != "?"
                else ""
            )
            print(f"  • {dataset['name']} - {dataset['size_mb']} MB{cols_info}")

    # Analyse des soumissions
    print("\n📤 SOUMISSIONS")
    print("-" * 30)
    submissions_info = analyze_submissions()
    print(f"📊 Nombre de soumissions: {submissions_info['count']}")
    print(f"💾 Taille totale: {submissions_info['total_size_mb']} MB")

    if submissions_info["submissions"]:
        print("\n📋 Soumissions récentes:")
        for sub in submissions_info["submissions"][:3]:  # Top 3
            rows_info = f" - {sub['rows']} lignes" if sub.get("rows") != "?" else ""
            print(
                f"  • {sub['name']}{rows_info} - {sub['modified'].strftime('%d/%m %H:%M')}"
            )

    # Performance des modèles
    print("\n🎯 PERFORMANCES")
    print("-" * 30)
    performance = get_model_performance()
    print(performance[:500] + "..." if len(performance) > 500 else performance)

    # Statistiques globales
    total_size = (
        models_info["total_size_mb"]
        + datasets_info["total_size_mb"]
        + submissions_info["total_size_mb"]
    )
    total_files = (
        models_info["count"] + datasets_info["count"] + submissions_info["count"]
    )

    print("\n💫 RÉSUMÉ GLOBAL")
    print("-" * 30)
    print(f"📁 Total de fichiers générés: {total_files}")
    print(f"💾 Espace disque utilisé: {total_size:.2f} MB")
    print(
        f"🚀 Projet prêt: {'✅ Oui' if models_info['count'] > 0 else '❌ Non (aucun modèle)'}"
    )

    # Recommandations
    print("\n💡 RECOMMANDATIONS")
    print("-" * 30)
    if models_info["count"] == 0:
        print("• Lancez 'python main.py' pour commencer l'entraînement")
    elif total_size > 1000:
        print("• Considérez 'python clean_outputs.py' pour libérer de l'espace")
    else:
        print("• Le projet est opérationnel et optimisé")

    print(
        f"\n📈 Dernière activité: {max([m.get('modified', datetime.min) for m in models_info['models']] + [datetime.min]).strftime('%d/%m/%Y à %H:%M')}"
    )


def main():
    """Fonction principale"""
    try:
        generate_report()
    except Exception as e:
        print(f"❌ Erreur lors de la génération du rapport: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code)
