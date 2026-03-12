import pandas as pd
import os


def report_missing_values(file_path):
    df = pd.read_csv(file_path)
    total_rows = len(df)
    missing_per_column = df.isnull().sum()
    report = []
    for col in df.columns:
        missing = missing_per_column[col]
        report.append(f"{col}: {missing} missing / {total_rows} total")
    print(f"\nReport for {os.path.basename(file_path)}:")
    print("\n".join(report))


if __name__ == "__main__":
    files = [
        "datas/X_test/clinical_test.csv",
        "datas/X_test/molecular_test_filled.csv",
        "datas/X_train/clinical_train.csv",
        "datas/X_train/molecular_train_filled.csv",
    ]
    for file in files:
        report_missing_values(file)
