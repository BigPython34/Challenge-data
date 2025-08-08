import pandas as pd
import re
from collections import Counter
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from src.config import (
    ALL_IMPORTANT_GENES,
    CYTOGENETIC_FAVORABLE,
    CYTOGENETIC_ADVERSE,
    CYTOGENETIC_INTERMEDIATE,
)


def run_comprehensive_data_discovery(clinical_path: str, molecular_path: str):
    """
    Analyse les données brutes de manière agnostique PUIS les compare à la config.
    """
    print("=" * 60)
    print("  SCRIPT DE DÉCOUVERTE COMPLET : DATA vs. CONFIG")
    print("=" * 60)

    # =================================================================
    # PARTIE 1 : ANALYSE MOLÉCULAIRE (DATA-DRIVEN)
    # =================================================================
    print("\n--- PARTIE 1: ANALYSE MOLÉCULAIRE ---")
    try:
        mol_df = pd.read_csv(molecular_path)

        # 1. Analyse d'abondance : quels sont TOUS les gènes dans nos données ?
        all_gene_counts = mol_df["GENE"].value_counts()
        genes_from_data = set(all_gene_counts.index)

        print(
            f"\n[1.1] Découverte : {len(genes_from_data)} gènes uniques trouvés dans les données."
        )
        print("      Top 25 des gènes les plus mutés (tous confondus) :")
        print(all_gene_counts.head(25).to_string())

        # 2. Analyse croisée : comparons avec notre config
        genes_from_config = set(ALL_IMPORTANT_GENES)

        print("\n[1.2] Analyse croisée (Données vs. Config) :")

        # Gènes de la config qui sont bien présents dans les données
        found_in_data = genes_from_config.intersection(genes_from_data)
        print(
            f"\n---> {len(found_in_data)} gènes de la config sont PRÉSENTS dans les données."
        )

        # Gènes de la config qui n'existent pas dans nos données (inutiles de créer une feature)
        missing_from_data = genes_from_config.difference(genes_from_data)
        if missing_from_data:
            print(
                f"\n---> ATTENTION : {len(missing_from_data)} gènes de la config sont ABSENTS des données (features à ignorer) :"
            )
            print(f"      {sorted(list(missing_from_data))}")

        # Gènes fréquents dans les données mais qui ne sont pas dans notre config
        # C'est ici qu'on peut trouver des pépites !
        potential_new_genes = {g for g in genes_from_data if g not in genes_from_config}
        if potential_new_genes:
            print(
                f"\n---> OPPORTUNITÉ : {len(potential_new_genes)} gènes existent dans les données mais sont ABSENTS de la config."
            )
            print(
                "      Suggestion : Envisager d'ajouter les plus fréquents à votre config ?"
            )
            # Afficher les plus fréquents de cette nouvelle liste
            new_gene_counts = all_gene_counts[list(potential_new_genes)].sort_values(
                ascending=False
            )
            print(new_gene_counts.head(15).to_string())

    except FileNotFoundError:
        print(f"ERREUR: Fichier moléculaire non trouvé à {molecular_path}")

    # =================================================================
    # PARTIE 2 : ANALYSE CYTOGENETIQUE
    # =================================================================
    print("\n\n--- PARTIE 2: ANALYSE CYTOGENETIQUE ---")
    try:
        clin_df = pd.read_csv(clinical_path, low_memory=False)
        cytogenetics_series = clin_df["CYTOGENETICS"].dropna().str.lower()

        # 2.1 Analyse dirigée : vérifier la présence des patterns de la config
        print(
            f"\n[2.1] Analyse dirigée : Vérification des {len(CYTOGENETIC_FAVORABLE)+len(CYTOGENETIC_ADVERSE)+len(CYTOGENETIC_INTERMEDIATE)} patterns de la config..."
        )
        all_patterns = {
            "FAVORABLE": CYTOGENETIC_FAVORABLE,
            "ADVERSE": CYTOGENETIC_ADVERSE,
            "INTERMEDIATE": CYTOGENETIC_INTERMEDIATE,
        }
        for category, patterns in all_patterns.items():
            print(f"\n  --- Catégorie {category} ---")
            for pattern in patterns:
                hits = cytogenetics_series.str.contains(pattern, regex=True).sum()
                if hits > 0:
                    print(f"    Pattern: {pattern:<30} | Trouvé chez {hits} patient(s)")

        # 2.2 Analyse ouverte : trouver les termes les plus fréquents
        print(
            "\n[2.2] Analyse ouverte : Recherche des termes (tokens) les plus fréquents..."
        )
        # On normalise un peu en remplaçant les délimiteurs par des espaces
        all_text = " ".join(cytogenetics_series.tolist())
        # On ne garde que les caractères alphanumériques, +, - et espaces
        clean_text = re.sub(r"[^a-zA-Z0-9\s\+\-]", " ", all_text)
        tokens = [
            token for token in clean_text.split() if not token.isdigit() or token == "8"
        ]  # Exclure les nombres purs sauf '8' (pour +8)

        token_counts = Counter(tokens)

        print(
            "      Top 30 des tokens les plus fréquents (pistes pour de nouvelles features) :"
        )
        for token, count in token_counts.most_common(30):
            print(f"        '{token}': {count} occurrences")

    except FileNotFoundError:
        print(f"ERREUR: Fichier clinique non trouvé à {clinical_path}")


# --- COMMENT UTILISER LE SCRIPT ---
if __name__ == "__main__":
    CLINICAL_FILE_PATH = "challenge_code/datas/X_train/clinical_train.csv"
    MOLECULAR_FILE_PATH = "challenge_code/datas/X_train/molecular_train_filled.csv"

    run_comprehensive_data_discovery(CLINICAL_FILE_PATH, MOLECULAR_FILE_PATH)
