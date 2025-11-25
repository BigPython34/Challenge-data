import pandas as pd
import argparse
import sys

def calculate_proportion_of_ones(file_path: str, column_name: str):
    """
    Lit un fichier CSV, analyse une colonne spécifiée et calcule la proportion de '1's.

    Args:
        file_path (str): Le chemin vers le fichier CSV.
        column_name (str): Le nom de la colonne à analyser.
    """
    print("=" * 60)
    print("  Analyse de Proportion de '1' dans une Colonne CSV")
    print("=" * 60)
    
    # --- 1. Chargement des données ---
    try:
        print(f"Lecture du fichier : '{file_path}'...")
        df = pd.read_csv(file_path)
        print(f"✓ Fichier chargé. Dimensions : {df.shape}")
    except FileNotFoundError:
        print(f"\n❌ ERREUR : Fichier non trouvé à l'emplacement '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR : Impossible de lire le fichier CSV. Détail : {e}")
        sys.exit(1)

    # --- 2. Vérification de la colonne ---
    if column_name not in df.columns:
        print(f"\n❌ ERREUR : La colonne '{column_name}' n'a pas été trouvée dans le fichier.")
        print("\nColonnes disponibles :")
        for col in df.columns:
            print(f"  - {col}")
        sys.exit(1)

    # Extraire la colonne
    column_data = df[column_name]
    
    # --- 3. Calcul de la proportion ---
    # S'assurer que la colonne est numérique pour les comparaisons
    if not pd.api.types.is_numeric_dtype(column_data):
        print(f"\n⚠️ AVERTISSEMENT : La colonne '{column_name}' n'est pas de type numérique.")
        # On essaie de la convertir, en mettant les erreurs en NaN
        column_data = pd.to_numeric(column_data, errors='coerce')
        print("   -> Tentative de conversion en numérique.")

    # Compter les valeurs nulles avant le calcul
    total_rows = len(column_data)
    non_null_rows = column_data.notna().sum()
    null_count = total_rows - non_null_rows
    
    if non_null_rows == 0:
        print(f"\n❌ ERREUR : La colonne '{column_name}' ne contient aucune valeur numérique valide.")
        sys.exit(1)

    # Calculer la proportion de 1
    count_of_ones = (column_data == 1).sum()
    proportion = count_of_ones / non_null_rows
    
    # --- 4. Affichage des résultats ---
    print("\n--- RÉSULTATS ---")
    print(f"Analyse de la colonne : '{column_name}'")
    print("-" * 25)
    print(f"Nombre total de lignes          : {total_rows}")
    print(f"Nombre de valeurs non-nulles    : {non_null_rows}")
    if null_count > 0:
        print(f"Nombre de valeurs nulles/non-num : {null_count}")
    print(f"Nombre d'occurrences de '1'     : {count_of_ones}")
    print(f"\nProportion de '1' (parmi non-nuls) : {proportion:.4f} ({proportion:.2%})")
    print("=" * 60)


if __name__ == "__main__":
    l=["DATA_PROFILE_clinical_only","DATA_PROFILE_complete","DATA_PROFILE_molecular_only","DATA_PROFILE_no_data"]
    for col in l:
        calculate_proportion_of_ones("datasets_processed/X_test_processed.csv", col)