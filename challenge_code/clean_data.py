import os
import pandas as pd
from src.data.data_cleaning.cleaner import clean_and_validate_data

# Définir les chemins des fichiers d'entrée et de sortie
input_clinical_path = "datas/X_train/clinical_train.csv"
input_molecular_path = "datas/X_train/molecular_train.csv"
input_target_path = "datas/target_train.csv"

output_clinical_path = "datasets/cleaned_clinical.csv"
output_molecular_path = "datasets/cleaned_molecular.csv"
output_target_path = "datasets/cleaned_target.csv"

input_clinical_test_path = "datas/X_test/clinical_test.csv"
input_target_test_path = "datas/X_test/fake_target_test.csv"
input_molecular_test_path = "datas/X_test/molecular_test.csv"


output_clinical_test_path = "datasets/cleaned_clinical_test.csv"
output_molecular_test_path = "datasets/cleaned_molecular_test.csv"
output_target_test_path = "datasets/cleaned_target_test.csv"


def main():
    print("=== DÉBUT DU NETTOYAGE DES DONNÉES ===")

    # Charger les données brutes
    clinical_df = pd.read_csv(input_clinical_path)
    molecular_df = pd.read_csv(input_molecular_path)
    target_df = pd.read_csv(input_target_path)
    # Nettoyer et valider les données
    cleaned_clinical, cleaned_molecular, cleaned_target = clean_and_validate_data(
        clinical_df, molecular_df, target_df
    )

    # Sauvegarder les données nettoyées
    os.makedirs(os.path.dirname(output_clinical_path), exist_ok=True)
    cleaned_clinical.to_csv(output_clinical_path, index=False)
    cleaned_molecular.to_csv(output_molecular_path, index=False)
    cleaned_target.to_csv(output_target_path, index=False)

    # Charger les données brutes
    clinical_test_df = pd.read_csv(input_clinical_test_path)
    molecular_test_df = pd.read_csv(input_molecular_test_path)
    target_test_df = pd.read_csv(input_target_test_path)
    # Nettoyer et valider les données
    cleaned_clinical_test, cleaned_molecular_test, cleaned_target_test = (
        clean_and_validate_data(clinical_test_df, molecular_test_df, target_test_df)
    )

    # Sauvegarder les données nettoyées
    os.makedirs(os.path.dirname(output_clinical_test_path), exist_ok=True)
    cleaned_clinical_test.to_csv(output_clinical_test_path, index=False)
    cleaned_molecular_test.to_csv(output_molecular_test_path, index=False)
    cleaned_target_test.to_csv(output_target_test_path, index=False)

    print("=== NETTOYAGE TERMINÉ ===")
    print(
        f"Données nettoyées sauvegardées dans {os.path.dirname(output_clinical_path)}"
    )


if __name__ == "__main__":
    main()
