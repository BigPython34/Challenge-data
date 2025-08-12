# Fichier: src/data/external_data_manager.py
import pandas as pd
import json
import requests
import time
import numpy as np


class ExternalDataManager:
    """
    Gère le chargement et la combinaison de plusieurs sources de données externes
    (COSMIC et OncoKB) pour l'enrichissement des données sur les gènes, et gère
    également le cache des scores de pathogénicité des variants.
    """

    def __init__(
        self,
        cosmic_path: str,
        oncokb_path: str,
        variant_cache_path: str = "datas/external/variant_cache.json",
    ):
        """
        Initialise le gestionnaire en chargeant, traitant et combinant
        les informations sur les gènes, et en initialisant le cache des variants.
        """
        # Charger et traiter chaque source de données sur les gènes
        cosmic_info = self._load_cosmic(cosmic_path)
        oncokb_info = self._load_oncokb(oncokb_path)

        # Combiner les informations des deux sources
        self.gene_info_data = self._combine_gene_info(cosmic_info, oncokb_info)

        # Initialiser le cache pour les scores de variants (logique inchangée)
        self.variant_cache_path = variant_cache_path
        self.variant_scores = self._load_variant_cache()

    def _load_cosmic(self, path: str) -> pd.DataFrame:
        """Charge et nettoie le fichier du COSMIC Cancer Gene Census."""
        print("[ExternalData] Chargement de COSMIC...")
        try:
            df = pd.read_csv(path)
            df.rename(columns={"Gene Symbol": "GENE"}, inplace=True)
            df["is_oncogene"] = (
                df["Role in Cancer"]
                .str.contains("oncogene", case=False, na=False)
                .astype(int)
            )
            df["is_tumor_suppressor"] = (
                df["Role in Cancer"]
                .str.contains("TSG", case=False, na=False)
                .astype(int)
            )
            grouped = df.groupby("GENE")[["is_oncogene", "is_tumor_suppressor"]].max()
            print(f"   -> {len(grouped)} gènes traités depuis COSMIC.")
            return grouped
        except Exception as e:
            print(f"   [AVERTISSEMENT] Impossible de charger le fichier COSMIC : {e}")
            return pd.DataFrame()

    def _load_oncokb(self, path: str) -> pd.DataFrame:
        """Charge et nettoie le fichier de la liste de gènes d'OncoKB."""
        print("[ExternalData] Chargement d'OncoKB...")
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                usecols=["Hugo Symbol", "Gene Type"],
                on_bad_lines="warn",
            )
            df.rename(columns={"Hugo Symbol": "GENE"}, inplace=True)
            df["is_oncogene"] = (
                df["Gene Type"]
                .str.contains("ONCOGENE", case=False, na=False)
                .astype(int)
            )
            df["is_tumor_suppressor"] = (
                df["Gene Type"].str.contains("TSG", case=False, na=False).astype(int)
            )
            grouped = df.groupby("GENE")[["is_oncogene", "is_tumor_suppressor"]].max()
            print(f"   -> {len(grouped)} gènes traités depuis OncoKB.")
            return grouped
        except Exception as e:
            print(f"   [AVERTISSEMENT] Impossible de charger le fichier OncoKB : {e}")
            return pd.DataFrame()

    def _combine_gene_info(
        self, cosmic_df: pd.DataFrame, oncokb_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine les informations de COSMIC et OncoKB en utilisant une logique 'OU'."""
        print("[ExternalData] Combinaison des sources de données sur les gènes...")
        combined_df = pd.concat([cosmic_df, oncokb_df])
        final_df = combined_df.groupby(level=0).max().astype(int)
        print(f"   -> {len(final_df)} gènes uniques dans la base de données combinée.")
        return final_df

    def get_gene_info(self):
        """Retourne les informations combinées sur les gènes, prêtes à être fusionnées."""
        return self.gene_info_data

    # --- Les méthodes pour le cache des variants restent INCHANGÉES ---

    def _load_variant_cache(self):
        """Charge le cache des scores de variants ou le retourne vide."""
        try:
            with open(self.variant_cache_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_variant_cache(self):
        """Sauvegarde le cache."""
        with open(self.variant_cache_path, "w") as f:
            json.dump(self.variant_scores, f)

    def fetch_and_cache_cadd_scores(self, variants_df: pd.DataFrame):
        """Interroge MyVariant.info pour les variants non cachés."""
        print("[ExternalData] Récupération des scores de pathogénicité (CADD)...")
        variants_df["hgv_id"] = variants_df.apply(
            lambda r: f"chr{r['CHR']}:g.{int(r['START'])}{r['REF']}>{r['ALT']}", axis=1
        )
        unique_variants = variants_df["hgv_id"].unique()
        to_fetch = [v for v in unique_variants if v not in self.variant_scores]

        if not to_fetch:
            print("   -> Tous les scores de variants sont déjà en cache.")
            return

        print(f"   -> {len(to_fetch)} nouveaux variants à interroger...")
        for i in range(0, len(to_fetch), 1000):
            batch = to_fetch[i : i + 1000]
            headers = {"content-type": "application/x-www-form-urlencoded"}
            try:
                res = requests.post(
                    "https://myvariant.info/v1/variant",
                    data={"ids": ",".join(batch), "fields": "dbnsfp.cadd.phred"},
                    headers=headers,
                )
                res.raise_for_status()  # Lève une exception si le statut est une erreur (4xx ou 5xx)
                results = res.json()
                for result in results:
                    query = result["query"]
                    score = (
                        result.get("dbnsfp", {}).get("cadd", {}).get("phred", np.nan)
                    )
                    self.variant_scores[query] = score
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(
                    f"  [ERREUR API] Échec de l'interrogation du lot {i//1000 + 1}: {e}"
                )

        self._save_variant_cache()
        print("   -> Cache des scores de variants mis à jour.")

    def get_variant_scores(self):
        """Retourne le dictionnaire des scores de variants."""
        return self.variant_scores
