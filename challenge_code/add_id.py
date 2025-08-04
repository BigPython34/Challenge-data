import pandas as pd
import os

print("=== SCRIPT D'AJOUT DES ID AUX DONNÉES DE TEST TRAITÉES ===")

# Chemins vers les fichiers
original_test_path = "datas/X_test/clinical_test.csv"
processed_test_path = "datasets_processed/X_test_processed.csv"

# Vérifier que les fichiers existent
if not os.path.exists(original_test_path):
    print(f"ERREUR : Fichier original introuvable : {original_test_path}")
elif not os.path.exists(processed_test_path):
    print(f"ERREUR : Fichier traité introuvable : {processed_test_path}")
else:
    # Charger les données
    df_original = pd.read_csv(original_test_path)
    df_processed = pd.read_csv(processed_test_path)
    print("Fichiers chargés avec succès.")

    # Vérification de sécurité : s'assurer que le nombre de lignes correspond
    if len(df_original) != len(df_processed):
        print(
            "ERREUR : Le nombre de lignes entre le fichier original et le fichier traité ne correspond pas."
        )
        print(
            f"Original : {len(df_original)} lignes | Traité : {len(df_processed)} lignes"
        )
    else:
        # Insérer la colonne ID au début du dataframe traité
        # On suppose que l'ordre des lignes a été préservé tout au long de la pipeline
        df_processed.insert(0, "ID", df_original["ID"].astype(str).values)
        print("La colonne 'ID' a été ajoutée au début du DataFrame de test.")

        # Sauvegarder le fichier en écrasant l'ancien
        df_processed.to_csv(processed_test_path, index=False)
        print(
            f"Le fichier {processed_test_path} a été mis à jour avec la colonne 'ID'."
        )
        print(f"Nouvelle shape du fichier : {df_processed.shape}")

print("\nScript terminé.")
