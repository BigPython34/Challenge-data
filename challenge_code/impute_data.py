import os
import pandas as pd
from src.data.data_cleaning.imputer import ClinicalImputer, ImputationStrategy

# Définir les chemins des fichiers d'entrée et de sortie
input_clinical_path = "datasets/cleaned_clinical.csv"
output_imputed_clinical_path = "datasets/imputed_clinical.csv"

# Définir les chemins des fichiers d'entrée et de sortie pour les données moléculaires
input_molecular_path = "datasets/cleaned_molecular.csv"
output_imputed_molecular_path = "datasets/imputed_molecular.csv"

# Définir les chemins des fichiers d'entrée et de sortie
input_clinical_test_path = "datasets/cleaned_clinical_test.csv"
output_imputed_clinical_test_path = "datasets/imputed_clinical_test.csv"

# Définir les chemins des fichiers d'entrée et de sortie pour les données moléculaires
input_molecular_test_path = "datasets/cleaned_molecular_test.csv"
output_imputed_molecular__testpath = "datasets/imputed_molecular_test.csv"


def main():
    print("=== DÉBUT DE L'IMPUTATION DES DONNÉES ===")

    # Charger les données nettoyées
    clinical_df = pd.read_csv(input_clinical_path)

    # Effectuer l'imputation intelligente
    imputed_clinical, imputation_metadata = (
        ClinicalImputer.intelligent_clinical_imputation(
            clinical_df, strategy=ImputationStrategy.MEDICAL_INFORMED
        )
    )

    # Sauvegarder les données imputées
    os.makedirs(os.path.dirname(output_imputed_clinical_path), exist_ok=True)
    imputed_clinical.to_csv(output_imputed_clinical_path, index=False)

    print("=== IMPUTATION TERMINÉE ===")
    print(f"Données imputées sauvegardées dans {output_imputed_clinical_path}")
    print("Métadonnées d'imputation:")
    print(imputation_metadata)

    # Charger les données moléculaires nettoyées
    molecular_df = pd.read_csv(input_molecular_path)

    # Effectuer l'imputation standard pour les données moléculaires
    imputed_molecular, molecular_metadata = (
        ClinicalImputer.intelligent_clinical_imputation(
            molecular_df, strategy=ImputationStrategy.ITERATIVE
        )
    )

    # Sauvegarder les données moléculaires imputées
    os.makedirs(os.path.dirname(output_imputed_molecular_path), exist_ok=True)
    imputed_molecular.to_csv(output_imputed_molecular_path, index=False)

    print("=== IMPUTATION DES DONNÉES MOLÉCULAIRES TERMINÉE ===")
    print(
        f"Données moléculaires imputées sauvegardées dans {output_imputed_molecular_path}"
    )
    print("Métadonnées d'imputation moléculaire:")
    print(molecular_metadata)

    # Imputation des données de test
    print("=== DÉBUT DE L'IMPUTATION DES DONNÉES DE TEST ===")

    # Charger les données cliniques de test nettoyées
    clinical_test_df = pd.read_csv(input_clinical_test_path)

    # Effectuer l'imputation intelligente pour les données cliniques de test
    imputed_clinical_test, clinical_test_metadata = (
        ClinicalImputer.intelligent_clinical_imputation(
            clinical_test_df, strategy=ImputationStrategy.MEDICAL_INFORMED
        )
    )

    # Sauvegarder les données cliniques de test imputées
    os.makedirs(os.path.dirname(output_imputed_clinical_test_path), exist_ok=True)
    imputed_clinical_test.to_csv(output_imputed_clinical_test_path, index=False)

    print("=== IMPUTATION DES DONNÉES CLINIQUES DE TEST TERMINÉE ===")
    print(
        f"Données cliniques de test imputées sauvegardées dans {output_imputed_clinical_test_path}"
    )
    print("Métadonnées d'imputation cliniques de test:")
    print(clinical_test_metadata)

    # Charger les données moléculaires de test nettoyées
    molecular_test_df = pd.read_csv(input_molecular_test_path)

    # Effectuer l'imputation standard pour les données moléculaires de test
    imputed_molecular_test, molecular_test_metadata = (
        ClinicalImputer.intelligent_clinical_imputation(
            molecular_test_df, strategy=ImputationStrategy.ITERATIVE
        )
    )

    # Sauvegarder les données moléculaires de test imputées
    os.makedirs(os.path.dirname(output_imputed_molecular__testpath), exist_ok=True)
    imputed_molecular_test.to_csv(output_imputed_molecular__testpath, index=False)

    print("=== IMPUTATION DES DONNÉES MOLÉCULAIRES DE TEST TERMINÉE ===")
    print(
        f"Données moléculaires de test imputées sauvegardées dans {output_imputed_molecular__testpath}"
    )
    print("Métadonnées d'imputation moléculaires de test:")
    print(molecular_test_metadata)


# Appeler la fonction d'imputation des données de test dans main
if __name__ == "__main__":
    main()
