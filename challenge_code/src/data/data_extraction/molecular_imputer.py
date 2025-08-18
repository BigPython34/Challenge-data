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
        """Crée un identifiant de variant unique de manière robuste."""
        try:
            # Gérer le cas où CHR est un nombre ou une chaîne comme 'X', 'Y'
            chr_val = str(row["CHR"]).replace(".0", "")  # Retire ".0" si c'est un float
            return f"chr{chr_val}:g.{int(row['START'])}{row['REF']}>{row['ALT']}"
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
        """
        Logique d'imputation et d'enrichissement pour une seule ligne de données moléculaires.
        Cette version gère les formats d'annotation variables (dict vs list) et
        inclut un débogage ciblé pour des variants spécifiques.
        """
        vid = row.get("variant_id")

        # --- BLOC DE DÉBOGAGE CIBLÉ (peut être commenté en production) ---
        # Mettez ici l'ID du variant que vous voulez tracer.
        TARGET_VID = "chrX:g.15349337C>T"
        is_target_variant = vid == TARGET_VID

        if is_target_variant:
            print(
                f"\n--- DEBUG: Variant CIBLE TROUVÉ {TARGET_VID} (Patient ID: {row.get('ID')}) ---"
            )
            print(
                f"  Ligne originale: EFFECT='{row.get('EFFECT')}', IMPACT='{row.get('IMPACT')}'"
            )
        # --- FIN DU BLOC DE DÉBOGAGE ---

        if not vid:
            return row  # Ne peut rien faire si l'ID n'a pas pu être généré

        data = self.variant_data.get(vid)
        if not data:
            if is_target_variant:
                print(
                    f"  [ERREUR DEBUG] Variant '{vid}' non trouvé dans le cache self.variant_data !"
                )
            return row

        if is_target_variant:
            print(f"  [INFO DEBUG] Variant '{vid}' trouvé dans le cache.")

        snpeff_data = data.get("snpeff", {}) or {}
        ann_raw = snpeff_data.get(
            "ann"
        )  # On ne met pas de valeur par défaut pour mieux analyser

        if not ann_raw:
            if is_target_variant:
                print(
                    "  [ERREUR DEBUG] Aucune clé 'ann' trouvée dans la section 'snpeff' du JSON !"
                )
            return row

        # --- CORRECTION CLÉ : Gérer les cas où 'ann' est un dict ou une liste ---
        annotations_list = []
        if isinstance(ann_raw, list):
            annotations_list = ann_raw
        elif isinstance(ann_raw, dict):
            annotations_list = [ann_raw]

        if not annotations_list:
            if is_target_variant:
                print("  [AVERTISSEMENT DEBUG] La clé 'ann' est vide !")
            return row

        # Sélectionner la meilleure annotation basée sur l'impact
        ann = self._get_best_annotation(annotations_list)

        if is_target_variant:
            print(f"  [INFO DEBUG] Meilleure annotation sélectionnée : {ann}")

        # --- Imputation et Enrichissement ---
        if pd.isna(row.get("GENE")) and ann.get("gene_name"):
            row["GENE"] = ann["gene_name"]

        if pd.isna(row.get("PROTEIN_CHANGE")) and ann.get("hgvs_p"):
            row["PROTEIN_CHANGE"] = ann["hgvs_p"]

        if pd.isna(row.get("EFFECT")) and ann.get("effect"):
            row["EFFECT"] = ann["effect"]
            if is_target_variant:
                print(f"    -> EFFECT a été imputé avec : '{ann['effect']}'")

        # Enrichissement avec la colonne IMPACT
        if ann.get("putative_impact"):
            row["IMPACT"] = ann["putative_impact"]
            if is_target_variant:
                print(
                    f"    -> IMPACT a été rempli/créé avec : '{ann['putative_impact']}'"
                )

        # Imputation de VAF et DEPTH depuis la section VCF
        vcf = data.get("vcf", {}) or {}
        if pd.isna(row.get("VAF")) and isinstance(vcf.get("freq"), (float, int)):
            row["VAF"] = vcf["freq"] / 100.0

        if pd.isna(row.get("DEPTH")) and isinstance(vcf.get("dp"), (int, float)):
            row["DEPTH"] = vcf["dp"]

        if is_target_variant:
            print(
                f"  Ligne finale: EFFECT='{row.get('EFFECT')}', IMPACT='{row.get('IMPACT')}'"
            )
            print("--- FIN DU BLOC DE DÉBOGAGE ---")

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
