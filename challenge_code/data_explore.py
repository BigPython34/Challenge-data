from src.data.data_exploration.data_exploration import (
    run_comprehensive_data_discovery,
)


if __name__ == "__main__":
    CLINICAL_FILE_PATH = "datas/X_train/clinical_train.csv"
    MOLECULAR_FILE_PATH = "datas/X_train/molecular_train_filled.csv"

    run_comprehensive_data_discovery(
        CLINICAL_FILE_PATH, MOLECULAR_FILE_PATH, out_dir="reports/data_explore/train"
    )

    CLINICAL_FILE_PATH = "datas/X_test/clinical_test.csv"
    MOLECULAR_FILE_PATH = "datas/X_test/molecular_test_filled.csv"

    run_comprehensive_data_discovery(
        CLINICAL_FILE_PATH, MOLECULAR_FILE_PATH, out_dir="reports/data_explore/test"
    )
