#!/usr/bin/env python3
"""
Script 2/3: Model Training
Loads the prepared dataset, trains models, and saves the best one.
"""
from src.modeling.train import (
    load_training_dataset_csv,
    train_and_save_all_models,
    evaluate_and_save_best_model,
    generate_visualization_report,
)
from sklearn.model_selection import train_test_split


def main():
    print("=== SCRIPT 2/3: MODEL TRAINING ===")
    print("Objective: Train and save models")
    print("=" * 60)

    # 1. Load the prepared dataset
    print("\n1. Loading the prepared dataset...")
    X_train_path = "datasets/selected_features.csv"
    y_train_path = "datasets/cleaned_target.csv"

    try:
        X, y = load_training_dataset_csv(X_train_path, y_train_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Split data into training and test sets
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.005, random_state=42
    )

    # 2. Train models
    print("\n2. Training models...")
    print("Training in progress (this may take several minutes)...")

    try:
        models = train_and_save_all_models(X_train, y_train)
        print(f"{len(models)} models successfully trained")
    except Exception as e:
        print(f"ERROR during training: {e}")
        return

    # 3. Evaluate and save the best model
    print("\n3. Evaluating models...")
    try:
        model_package = evaluate_and_save_best_model(
            models, X_train, y_train, X_test, y_test, X_train.columns.tolist()
        )
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        return

    # 4. Generate visualization report
    print("\n4. Generating visualization report...")
    try:
        generate_visualization_report(
            models, model_package["evaluation_results"], None, None
        )
    except Exception as e:
        print(f"Warning: Visualization report generation failed: {e}")

    print("\n" + "=" * 60)
    print("SCRIPT 2/3 COMPLETED SUCCESSFULLY!")
    print(f"Best model: {model_package['best_model_name']}")
    print("Models ready for predictions")
    print("Next step: python 3_predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
