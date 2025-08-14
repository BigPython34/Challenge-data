import json
import time
from typing import Dict

import numpy as np
import pandas as pd
import requests


class ExternalDataManager:
    """Combine COSMIC (v102 GRCh38) and OncoKB gene annotations; manage variant cache."""

    def __init__(
        self,
        cosmic_path: str,
        oncokb_path: str,
        variant_cache_path: str = "datas/external/variant_cache.json",
    ) -> None:
        cosmic_info = self._load_cosmic(cosmic_path)
        oncokb_info = self._load_oncokb(oncokb_path)
        self.gene_info_data = self._combine_gene_info(cosmic_info, oncokb_info)
        self.variant_cache_path = variant_cache_path
        self.variant_scores = self._load_variant_cache()

    # ---------------- COSMIC -----------------
    def _load_cosmic(self, path: str) -> pd.DataFrame:
        print("[ExternalData] Chargement de COSMIC...")
        try:
            df = pd.read_csv(
                path, sep="\t", dtype=str, engine="python", on_bad_lines="skip"
            )
        except Exception as e:
            print(f"   [AVERTISSEMENT] Impossible de lire COSMIC: {e}")
            return pd.DataFrame()

        df.columns = [str(c).strip() for c in df.columns]
        if "GENE_SYMBOL" not in df.columns:
            print("   [AVERTISSEMENT] Colonne GENE_SYMBOL absente dans COSMIC.")
            return pd.DataFrame()

        keep = {
            "GENE_SYMBOL": "GENE",
            "ROLE_IN_CANCER": "ROLE_IN_CANCER",
            "MUTATION_TYPES": "MUTATION_TYPES",
            "TRANSLOCATION_PARTNER": "TRANSLOCATION_PARTNER",
            "MOLECULAR_GENETICS": "MOLECULAR_GENETICS",
            "TIER": "TIER",
            "SOMATIC": "SOMATIC",
            "GERMLINE": "GERMLINE",
            "TUMOUR_TYPES_SOMATIC": "TUMOUR_TYPES_SOMATIC",
        }
        present = {k: v for k, v in keep.items() if k in df.columns}
        df = df[list(present.keys())].rename(columns=present)

        df["GENE"] = df["GENE"].astype(str).str.strip()
        for c in [
            "ROLE_IN_CANCER",
            "MUTATION_TYPES",
            "TRANSLOCATION_PARTNER",
            "MOLECULAR_GENETICS",
            "SOMATIC",
            "GERMLINE",
            "TUMOUR_TYPES_SOMATIC",
        ]:
            if c in df.columns:
                df[c] = df[c].fillna("")
        if "TIER" in df.columns:
            df["TIER"] = pd.to_numeric(df["TIER"], errors="coerce")

        role = df.get("ROLE_IN_CANCER", pd.Series([""] * len(df)))
        df["is_oncogene"] = role.str.contains("oncogene", case=False, na=False).astype(
            int
        )
        df["is_tumor_suppressor"] = role.str.contains(
            r"TSG|tumour\s*suppressor|tumor\s*suppressor", case=False, na=False
        ).astype(int)

        df["cosmic_is_fusion_gene"] = (
            role.str.contains("fusion", case=False, na=False)
            | df.get("TRANSLOCATION_PARTNER", pd.Series([""] * len(df)))
            .str.strip()
            .ne("")
            | df.get("MUTATION_TYPES", pd.Series([""] * len(df))).str.contains(
                r"\bT\b", case=False, na=False
            )
        ).astype(int)

        mut_s = df.get("MUTATION_TYPES", pd.Series([""] * len(df))).str.upper()
        for tok, outcol in {
            "MIS": "cosmic_has_missense_common",
            "N": "cosmic_has_nonsense_common",
            "F": "cosmic_has_frameshift_common",
            "T": "cosmic_has_translocation_common",
            "S": "cosmic_has_splice_common",
            "A": "cosmic_has_amplification_common",
            "D": "cosmic_has_deletion_common",
        }.items():
            df[outcol] = mut_s.str.contains(rf"\b{tok}\b", na=False).astype(int)

        mg = df.get("MOLECULAR_GENETICS", pd.Series([""] * len(df))).str.upper()
        df["molgen_dominant"] = mg.str.contains(r"\bDOM\b", na=False).astype(int)
        df["molgen_recessive"] = mg.str.contains(r"\bREC\b", na=False).astype(int)
        for tok, outcol in {
            "E": "molgen_E",
            "L": "molgen_L",
            "M": "molgen_M",
            "O": "molgen_O",
        }.items():
            df[outcol] = mg.str.contains(rf"\b{tok}\b", na=False).astype(int)

        som = df.get("SOMATIC")
        ger = df.get("GERMLINE")
        df["cosmic_has_somatic_evidence"] = (
            som.str.lower().eq("y").astype(int) if som is not None else 0
        )
        df["cosmic_has_germline_evidence"] = (
            ger.str.lower().eq("y").astype(int) if ger is not None else 0
        )
        tts = df.get("TUMOUR_TYPES_SOMATIC", pd.Series([""] * len(df))).str.upper()
        df["cosmic_in_aml"] = tts.str.contains(r"\bAML\b", na=False).astype(int)
        df["cosmic_in_leukemia_lymphoma"] = tts.str.contains(
            "LEUK|LYMPH", na=False
        ).astype(int)

        agg_cols = [
            "is_oncogene",
            "is_tumor_suppressor",
            "cosmic_is_fusion_gene",
            "cosmic_has_missense_common",
            "cosmic_has_nonsense_common",
            "cosmic_has_frameshift_common",
            "cosmic_has_translocation_common",
            "cosmic_has_splice_common",
            "cosmic_has_amplification_common",
            "cosmic_has_deletion_common",
            "molgen_dominant",
            "molgen_recessive",
            "molgen_E",
            "molgen_L",
            "molgen_M",
            "molgen_O",
            "cosmic_has_somatic_evidence",
            "cosmic_has_germline_evidence",
            "cosmic_in_aml",
            "cosmic_in_leukemia_lymphoma",
        ]
        grouped = df.groupby("GENE")[agg_cols].max()
        if "TIER" in df.columns:
            grouped = grouped.join(
                df.groupby("GENE")["TIER"].min().rename("cosmic_tier_min")
            )
        for c in agg_cols:
            grouped[c] = grouped[c].fillna(0).astype(int)
        print(f"   -> {len(grouped)} gènes traités depuis COSMIC.")
        return grouped

    # ---------------- OncoKB -----------------
    def _load_oncokb(self, path: str) -> pd.DataFrame:
        print("[ExternalData] Chargement d'OncoKB...")
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                usecols=["Hugo Symbol", "Gene Type"],
                on_bad_lines="warn",
            )
        except Exception as e:
            print(f"   [AVERTISSEMENT] Impossible de lire OncoKB: {e}")
            return pd.DataFrame()
        df = df.rename(columns={"Hugo Symbol": "GENE"})
        df["is_oncogene"] = (
            df["Gene Type"].astype(str).str.contains("ONCOGENE", case=False, na=False)
        )
        df["is_tumor_suppressor"] = (
            df["Gene Type"].astype(str).str.contains("TSG", case=False, na=False)
        )
        grouped = (
            df.groupby("GENE")[["is_oncogene", "is_tumor_suppressor"]].max().astype(int)
        )
        print(f"   -> {len(grouped)} gènes traités depuis OncoKB.")
        return grouped

    # -------------- Combine sources ----------
    def _combine_gene_info(
        self, cosmic_df: pd.DataFrame, oncokb_df: pd.DataFrame
    ) -> pd.DataFrame:
        print("[ExternalData] Combinaison des sources de données sur les gènes...")
        if cosmic_df is None or cosmic_df.empty:
            return oncokb_df.copy()
        if oncokb_df is None or oncokb_df.empty:
            return cosmic_df.copy()
        combined = pd.concat([cosmic_df, oncokb_df], axis=0, sort=False)

        def agg_fun(x: pd.Series):
            if x.name == "cosmic_tier_min":
                return x.min(skipna=True)
            return pd.to_numeric(x, errors="coerce").max()

        out = combined.groupby(level=0).agg(agg_fun)
        for c in out.columns:
            if c == "cosmic_tier_min":
                continue
            series = pd.to_numeric(out[c], errors="coerce")
            if series.dropna().isin([0, 1]).all():
                out[c] = series.fillna(0).astype(int)
        print(f"   -> {len(out)} gènes uniques dans la base combinée.")
        return out

    def get_gene_info(self) -> pd.DataFrame:
        return self.gene_info_data

    # --------------- Variant cache -----------
    def _load_variant_cache(self) -> Dict[str, float]:
        try:
            with open(self.variant_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_variant_cache(self) -> None:
        with open(self.variant_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.variant_scores, f)

    def fetch_and_cache_cadd_scores(self, variants_df: pd.DataFrame) -> None:
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
                    timeout=30,
                )
                res.raise_for_status()
                results = res.json()
                for result in results:
                    query = result.get("query")
                    score = (
                        result.get("dbnsfp", {}).get("cadd", {}).get("phred", np.nan)
                    )
                    if query is not None:
                        self.variant_scores[query] = score
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(
                    f"  [ERREUR API] Échec de l'interrogation du lot {i//1000 + 1}: {e}"
                )
        self._save_variant_cache()
        print("   -> Cache des scores de variants mis à jour.")

    def get_variant_scores(self) -> Dict[str, float]:
        return self.variant_scores
