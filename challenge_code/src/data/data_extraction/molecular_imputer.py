import pandas as pd
import json
from typing import Dict, List
import numpy as np


class MolecularImputer:
    """
    Impute les valeurs manquantes dans les dataframes moléculaires en utilisant
    un cache de données nettoyées de MyVariant.
    """

    def __init__(self, myvariant_cleaned_path: str):
        self.variant_data = self._load_myvariant_cache(myvariant_cleaned_path)
        self.imputation_report = {}

    def _load_myvariant_cache(self, path: str) -> Dict:
        """Charge le cache de données MyVariant nettoyées."""
        print(f"Chargement du cache MyVariant depuis : {path}")
        try:
            data = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line.strip())
                        vid, content = list(record.items())[0]
                        data[vid] = content
            print(f"✓ {len(data)} variants chargés dans le cache.")
            return data
        except Exception as e:
            print(f"❌ Erreur lors du chargement du cache MyVariant : {e}")
            return {}

    def _make_variant_id(self, row: pd.Series) -> str:
        """Crée un identifiant de variant unique pour une ligne de dataframe."""
        try:
            return (
                f"chr{int(row['CHR'])}:g.{int(row['START'])}{row['REF']}>{row['ALT']}"
            )
        except (ValueError, TypeError):
            return None

    def _get_best_annotation(self, ann_raw: list) -> dict:
        """Parcourt une liste d'annotations SnpEff et retourne la plus pertinente (impact le plus élevé)."""
        if not ann_raw or not isinstance(ann_raw, list):
            return {}
        impact_order = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}
        best_ann, max_impact_score = None, 0
        for ann in ann_raw:
            if isinstance(ann, dict):
                impact_str = ann.get("putative_impact")
                current_impact_score = impact_order.get(impact_str, 0)
                if current_impact_score > max_impact_score:
                    max_impact_score = current_impact_score
                    best_ann = ann
        return (
            best_ann
            if best_ann
            else (ann_raw[0] if ann_raw and isinstance(ann_raw[0], dict) else {})
        )

    def _fill_row(self, row: pd.Series) -> pd.Series:
        """Logique d'imputation AMÉLIORÉE qui extrait aussi l'IMPACT."""
        vid = row["variant_id"]
        data = self.variant_data.get(vid)
        if not data:
            return row

        snpeff_data = data.get("snpeff", {}) or {}
        ann_raw = snpeff_data.get("ann", [])
        ann = self._get_best_annotation(ann_raw)

        # Imputation
        if pd.isna(row.get("GENE")) and "gene_name" in ann:
            row["GENE"] = ann["gene_name"]
        if pd.isna(row.get("PROTEIN_CHANGE")) and "hgvs_p" in ann:
            row["PROTEIN_CHANGE"] = ann["hgvs_p"]
        if pd.isna(row.get("EFFECT")) and "effect" in ann:
            row["EFFECT"] = ann["effect"]

        # --- AJOUT CRUCIAL ICI ---
        # On extrait la nouvelle information sur l'impact
        if "putative_impact" in ann:
            row["IMPACT"] = ann["putative_impact"]

        vcf = data.get("vcf", {}) or {}
        if pd.isna(row.get("VAF")) and isinstance(vcf.get("freq"), (float, int)):
            row["VAF"] = vcf["freq"] / 100.0
        if pd.isna(row.get("DEPTH")) and isinstance(vcf.get("dp"), (int, float)):
            row["DEPTH"] = vcf["dp"]
        return row

    def impute_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'imputation à un dataframe complet."""
        print(f"Début de l'imputation pour un dataframe de shape {df.shape}...")
        # S'assurer que la colonne IMPACT existe pour l'imputation
        if "IMPACT" not in df.columns:
            df["IMPACT"] = np.nan
        df["variant_id"] = df.apply(self._make_variant_id, axis=1)
        filled_df = df.apply(self._fill_row, axis=1)
        filled_df = filled_df.drop(columns=["variant_id"])
        print("✓ Imputation terminée.")
        return filled_df
