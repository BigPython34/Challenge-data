import pandas as pd
import numpy as np


def select_uncorrelated_features(df, correlation_threshold=1, variance_threshold=0.00):
    """
    Select features that are not highly correlated and have sufficient variance.

    Parameters:
    - df: DataFrame containing features.
    - correlation_threshold: Threshold above which features are considered highly correlated.
    - variance_threshold: Minimum variance required for a feature to be selected.

    Returns:
    - DataFrame with selected features.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate variance for each feature
    variances = numeric_df.var()
    low_variance_features = variances[variances < variance_threshold].index

    # Drop low variance features
    numeric_df = numeric_df.drop(columns=low_variance_features)

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Identify highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    correlated_features = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > correlation_threshold)
    ]

    # Drop highly correlated features
    numeric_df = numeric_df.drop(columns=correlated_features)

    return numeric_df


if __name__ == "__main__":
    # Example usage
    input_path = "datasets/final_training_dataset.csv"
    output_path = "datasets/selected_features.csv"

    # Load dataset
    data = pd.read_csv(input_path)

    # Select uncorrelated features
    selected_features = select_uncorrelated_features(data)

    # Save selected features
    selected_features.to_csv(output_path, index=False)
    print(f"Selected features saved to {output_path}")
