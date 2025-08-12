import pandas as pd


def generate_fake_target_test(input_file, output_file):
    # Read the input CSV file
    clinical_data = pd.read_csv(input_file)

    # Create a new DataFrame with the same IDs and fake target values
    fake_target_data = pd.DataFrame(
        {
            "ID": clinical_data["ID"],
            "OS_YEARS": [1] * len(clinical_data),
            "OS_STATUS": [0.0] * len(clinical_data),
        }
    )

    # Save the fake target data to a new CSV file
    fake_target_data.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = "datas/X_test/clinical_test.csv"
    output_file = "datas/X_test/fake_target_test.csv"
    generate_fake_target_test(input_file, output_file)
