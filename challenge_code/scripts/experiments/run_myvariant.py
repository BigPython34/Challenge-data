import pandas as pd
from src.data.data_extraction.myvariant_cleaner import MyVariantCleaner
from src.data.data_extraction.molecular_imputer import MolecularImputer


def main():
    print("=" * 60)
    print("  WORKFLOW D'ENRICHISSEMENT ET D'IMPUTATION DES DONNÉES MOLÉCULAIRES")
    print("=" * 60)

    # --- Configuration des chemins ---
    RAW_MYVARIANT_PATH = "datas/variant_data.jsonl"
    CLEANED_MYVARIANT_PATH = "datas/variant_data_bis.jsonl"

    RAW_TRAIN_PATH = "datas/X_train/molecular_train.csv"
    RAW_TEST_PATH = "datas/X_test/molecular_test.csv"

    FILLED_TRAIN_PATH = "datas/X_train/molecular_train_filled.csv"
    FILLED_TEST_PATH = "datas/X_test/molecular_test_filled.csv"


    print("\n[ÉTAPE 1/2] Nettoyage des données MyVariant...")
    cleaner = MyVariantCleaner()
    cleaner.process_raw_jsonl(RAW_MYVARIANT_PATH, CLEANED_MYVARIANT_PATH)


    print("\n[ÉTAPE 2/2] Imputation des fichiers moléculaires...")
    imputer = MolecularImputer(myvariant_cleaned_path=CLEANED_MYVARIANT_PATH)

    # Charger les dataframes bruts
    try:
        train_df = pd.read_csv(RAW_TRAIN_PATH)
        test_df = pd.read_csv(RAW_TEST_PATH)
    except FileNotFoundError as e:
        print(f"❌ Fichier brut manquant : {e}. Arrêt du script.")
        return

    # Imputer le dataframe d'entraînement
    train_filled_df = imputer.impute_dataframe(train_df)
    train_filled_df.to_csv(FILLED_TRAIN_PATH, index=False)
    print(f"✓ Fichier d'entraînement imputé sauvegardé à : {FILLED_TRAIN_PATH}")

    # Imputer le dataframe de test
    test_filled_df = imputer.impute_dataframe(test_df)
    test_filled_df.to_csv(FILLED_TEST_PATH, index=False)
    print(f"✓ Fichier de test imputé sauvegardé à : {FILLED_TEST_PATH}")

    print("\n" + "=" * 60)
    print("  Workflow terminé avec succès !")
    print("=" * 60)


if __name__ == "__main__":
    main()
