# Prédiction sur de nouveaux jeux de données
import pandas as pd
import os
from datetime import datetime

from ..config import RESULTS_DIR


def predict_and_save_submission(model, X_eval, df_eval_enriched, filename_suffix=""):
    """Fait des prédictions et sauvegarde le fichier de soumission"""
    # Prédictions
    predictions = model.predict(X_eval)

    # Créer le DataFrame de soumission
    submission_df = pd.DataFrame(
        {"ID": df_eval_enriched["ID"], "risk_score": predictions}
    )

    # Nom du fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename_suffix:
        filename = f"submission_{filename_suffix}_{timestamp}.csv"
    else:
        filename = f"submission_{timestamp}.csv"

    # Sauvegarde
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    submission_df.to_csv(filepath, index=False)

    print(f"Fichier de soumission sauvegardé: {filepath}")
    return filepath, submission_df


def predict_with_best_model(models, best_model_name, X_eval, df_eval_enriched):
    """Fait des prédictions avec le meilleur modèle"""
    best_model = models[best_model_name]["model"]

    print(f"Prédictions avec le meilleur modèle: {best_model_name}")
    filepath, submission_df = predict_and_save_submission(
        best_model, X_eval, df_eval_enriched, filename_suffix=f"best_{best_model_name}"
    )

    return filepath, submission_df


def predict_with_all_models(models, X_eval, df_eval_enriched):
    """Fait des prédictions avec tous les modèles"""
    submissions = {}

    for name, model_info in models.items():
        model = model_info["model"]
        print(f"Prédictions avec le modèle {name}...")

        filepath, submission_df = predict_and_save_submission(
            model, X_eval, df_eval_enriched, filename_suffix=name
        )

        submissions[name] = {"filepath": filepath, "submission_df": submission_df}

    return submissions
