import pandas as pd
import numpy as np
from src.data.features.feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
    MolecularFeatureExtraction,
    IntegratedFeatureEngineering,
)
import os


def perform_feature_engineering(clinical_data_path, molecular_data_path, output_path):
    """
    Perform feature engineering on clinical and molecular data.

    Parameters
    ----------
    clinical_data_path : str
        Path to the cleaned and imputed clinical data file.
    molecular_data_path : str
        Path to the cleaned and imputed molecular data file.
    output_path : str
        Path to save the engineered features.
    """
    # Load clinical and molecular data
    clinical_df = pd.read_csv(clinical_data_path)
    molecular_df = pd.read_csv(molecular_data_path)

    # Perform clinical feature engineering
    clinical_features = ClinicalFeatureEngineering.create_clinical_features(clinical_df)

    # Perform one-hot encoding for CENTER variable
    clinical_features = ClinicalFeatureEngineering._create_center_one_hot_encoding(
        clinical_features
    )

    # Perform molecular feature extraction
    molecular_features = MolecularFeatureExtraction.extract_molecular_risk_features(
        molecular_df, molecular_df
    )

    # Perform cytogenetic feature extraction
    cytogenetic_features = (
        CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(clinical_df)
    )

    # Combine all features
    final_features = IntegratedFeatureEngineering.combine_all_features(
        clinical_features, molecular_features, molecular_features, cytogenetic_features
    )
    final_features = final_features.fillna(0)
    # Save the engineered features
    final_features.to_csv(output_path, index=False)


# Appeler la fonction d'ingénierie des caractéristiques des données de test dans main
if __name__ == "__main__":
    perform_feature_engineering(
        "datasets/imputed_clinical.csv",
        "datasets/imputed_molecular.csv",
        "datasets/final_training_dataset.csv",
    )

    # Feature engineering pour les données de test
    perform_feature_engineering(
        "datasets/imputed_clinical_test.csv",
        "datasets/imputed_molecular_test.csv",
        "datasets/engineered_features_test.csv",
    )
