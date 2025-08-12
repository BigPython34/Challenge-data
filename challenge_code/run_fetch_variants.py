from src.data.data_extraction.myvariant_fetcher import MyVariantFetcher


def main():
    print("=" * 60)
    print("  WORKFLOW DE RÉCUPÉRATION DES DONNÉES BRUTES MYVARIANT")
    print("=" * 60)

    # --- Configuration des chemins ---
    # Liste de tous les fichiers à scanner pour trouver les variants
    INPUT_FILES = [
        "datas/X_train/molecular_train.csv",
        "datas/X_test/molecular_test.csv",
    ]

    # Fichier de sortie qui sera lu par le MyVariantCleaner
    OUTPUT_PATH = "datas/variant_data.jsonl"

    # --- Exécution ---
    fetcher = MyVariantFetcher(concurrency_limit=50, retries=3)
    fetcher.run(input_files=INPUT_FILES, output_path=OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("  Workflow terminé avec succès !")
    print(f"  Le fichier '{OUTPUT_PATH}' est prêt pour l'étape de nettoyage.")
    print("=" * 60)


if __name__ == "__main__":
    main()
