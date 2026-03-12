import json
import time
import os
from typing import Dict, Set, Optional

import numpy as np
import pandas as pd
import requests
from .myvariant_cleaner import MyVariantCleaner


class ExternalDataManager:
    """Combine COSMIC/OncoKB gene annotations, ClinVar signals, and variant caches."""

    def __init__(
        self,
        cosmic_path: str,
        oncokb_path: str,
        variant_cache_path: str = "datas/external/variant_cache.json",
        clinvar_path: Optional[str] = None,
    ) -> None:
        cosmic_info = self._load_cosmic(cosmic_path)
        oncokb_info = self._load_oncokb(oncokb_path)
        self.gene_info_data = self._combine_gene_info(cosmic_info, oncokb_info)
        self.variant_cache_path = variant_cache_path
        self.variant_scores = self._load_variant_cache()
        # MyVariant annotation cache (JSONL raw + cleaned)
        self.myvariant_raw_path = os.path.join("datas", "variant_data.jsonl")
        self.myvariant_cleaned_path = os.path.join("datas", "variant_data_bis.jsonl")
        self.variant_annotations: Dict[str, dict] = self._load_myvariant_cleaned(
            self.myvariant_cleaned_path
        )
        self.clinvar_lookup: Dict[str, str] = (
            self._load_clinvar_annotations(clinvar_path)
            if clinvar_path
            else {}
        )

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
            "CHR_BAND": "CHR_BAND",
            "CHROMOSOME": "CHROMOSOME",
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
        # Keep minimum tier and earliest non-null chr band for gene
        if "TIER" in df.columns:
            grouped = grouped.join(
                df.groupby("GENE")["TIER"].min().rename("cosmic_tier_min")
            )
        # Chromosome band/arm mapping (first non-null per gene)
        if "CHR_BAND" in df.columns:
            band_first = (
                df.assign(CHR_BAND=df["CHR_BAND"].replace({"": np.nan}))
                .groupby("GENE")["CHR_BAND"]
                .first()
                .rename("cosmic_chr_band")
            )
            grouped = grouped.join(band_first)
        if "CHROMOSOME" in df.columns:
            chrom_first = (
                df.assign(CHROMOSOME=df["CHROMOSOME"].replace({"": np.nan}))
                .groupby("GENE")["CHROMOSOME"]
                .first()
                .rename("cosmic_chromosome")
            )
            grouped = grouped.join(chrom_first)
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
        # Derive chromosome arm from COSMIC band if available (e.g., 5q31 -> 5q)
        if "cosmic_chr_band" in out.columns:
            arm = (
                out["cosmic_chr_band"]
                .astype(str)
                .str.extract(r"^([0-9XY]+[pq])", expand=False)
            )
            out["cosmic_chr_arm"] = arm
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

    def _load_clinvar_annotations(self, path: Optional[str]) -> Dict[str, str]:
        if not path:
            return {}
        if not os.path.exists(path):
            print(f"[ExternalData] Fichier ClinVar introuvable: {path}")
            return {}
        print("[ExternalData] Chargement de ClinVar (peut être long)...")

        header_cols: Optional[list[str]] = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("#CHROM"):
                        header_cols = line.lstrip("#").strip().split("\t")
                        break
        except Exception as exc:  # noqa: BLE001
            print(f"   [AVERTISSEMENT] Lecture préliminaire de ClinVar impossible: {exc}")
            return {}

        if not header_cols or not {"CHROM", "POS", "REF", "ALT", "INFO"}.issubset(
            header_cols
        ):
            print(
                "   [AVERTISSEMENT] Impossible d'identifier l'en-tête ClinVar (colonnes CHROM, POS, REF, ALT, INFO)."
            )
            return {}

        try:
            clinvar_df = pd.read_csv(
                path,
                sep="\t",
                comment="#",
                compression="infer",
                dtype=str,
                names=header_cols,
                header=None,
                engine="python",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   [AVERTISSEMENT] Impossible de lire ClinVar: {exc}")
            return {}

        required = {"CHROM", "POS", "REF", "ALT", "INFO"}
        missing_cols = required - set(clinvar_df.columns)
        if missing_cols:
            print(
                "   [AVERTISSEMENT] Colonnes requises manquantes dans ClinVar (CHROM, POS, REF, ALT, INFO)."
            )
            return {}

        df = clinvar_df[list(required)].copy()
        df = df[~df["ALT"].astype(str).str.contains(",", na=False)]
        df["CLNSIG"] = df["INFO"].astype(str).str.extract(r"CLNSIG=([^;]+)")
        df = df.dropna(subset=["CLNSIG"])
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["POS", "REF", "ALT"])
        if df.empty:
            return {}

        chrom = df["CHROM"].astype(str).str.replace(r"^chr", "", regex=True).str.strip()
        variant_ids = (
            "chr"
            + chrom
            + ":g."
            + df["POS"].astype(int).astype(str)
            + df["REF"].astype(str)
            + ">"
            + df["ALT"].astype(str)
        )
        df["variant_id"] = variant_ids
        dedup = df.dropna(subset=["variant_id"]).drop_duplicates("variant_id")
        print(f"   -> {len(dedup)} variants ClinVar indexés.")
        return dedup.set_index("variant_id")["CLNSIG"].to_dict()

    @staticmethod
    def _build_variant_id_series(df: pd.DataFrame) -> Optional[pd.Series]:
        required = {"CHR", "START", "REF", "ALT"}
        if not required.issubset(df.columns):
            return None
        tmp = df[list(required)].copy()
        tmp["CHR"] = tmp["CHR"].astype(str).str.replace(r"^chr", "", regex=True).str.strip()
        tmp["START"] = pd.to_numeric(tmp["START"], errors="coerce").astype("Int64")
        tmp["REF"] = tmp["REF"].astype(str).str.strip()
        tmp["ALT"] = tmp["ALT"].astype(str).str.strip()
        tmp = tmp.dropna(subset=["CHR", "START", "REF", "ALT"])
        if tmp.empty:
            return None
        variant_ids = (
            "chr"
            + tmp["CHR"]
            + ":g."
            + tmp["START"].astype(int).astype(str)
            + tmp["REF"]
            + ">"
            + tmp["ALT"]
        )
        series = pd.Series(index=df.index, dtype=object)
        series.loc[tmp.index] = variant_ids
        return series

    def merge_clinvar_annotations(self, maf_df: pd.DataFrame) -> pd.DataFrame:
        if maf_df is None or maf_df.empty or not self.clinvar_lookup:
            return maf_df
        variant_ids = self._build_variant_id_series(maf_df)
        if variant_ids is None:
            return maf_df
        annotated = maf_df.copy()
        annotated["clinvar_variant_id"] = variant_ids
        annotated["clinvar_clnsig"] = annotated["clinvar_variant_id"].map(
            self.clinvar_lookup
        )
        return annotated

    @property
    def has_clinvar_annotations(self) -> bool:
        return bool(self.clinvar_lookup)

    def fetch_and_cache_cadd_scores(self, variants_df: pd.DataFrame) -> None:
        print("[ExternalData] Récupération des scores de pathogénicité (CADD)...")
        required = {"CHR", "START", "REF", "ALT"}
        if not required.issubset(variants_df.columns):
            print(
                "   [INFO] Colonnes requises manquantes pour CADD, préchargement ignoré."
            )
            return

        df = variants_df[list(required)].copy()
        # Coerce types and sanitize
        df["START"] = pd.to_numeric(df["START"], errors="coerce")
        df["CHR"] = (
            df["CHR"].astype(str).str.replace(r"^chr", "", regex=True).str.strip()
        )
        df["REF"] = df["REF"].astype(str).str.strip()
        df["ALT"] = df["ALT"].astype(str).str.strip()
        # SNV-only: single-letter REF/ALT
        snv_mask = (df["REF"].str.len() == 1) & (df["ALT"].str.len() == 1)
        df = df.loc[snv_mask]
        df = df.dropna(subset=["CHR", "START", "REF", "ALT"])
        if df.empty:
            print("   [INFO] Aucun variant SNV valide à précharger pour CADD.")
            return
        df["hgv_id"] = df.apply(
            lambda r: f"chr{r['CHR']}:g.{int(r['START'])}{r['REF']}>{r['ALT']}", axis=1
        )
        unique_variants = df["hgv_id"].dropna().unique()

        def _needs_fetch(key: str) -> bool:
            if key not in self.variant_scores:
                return True
            val = self.variant_scores.get(key)
            try:
                return val is None or (isinstance(val, float) and np.isnan(val))
            except Exception:
                return True

        to_fetch = [v for v in unique_variants if _needs_fetch(v)]
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
                    data={
                        "ids": ",".join(batch),
                        "fields": "cadd.phred,dbnsfp.cadd.phred",
                        "assembly": "hg38",
                    },
                    headers=headers,
                    timeout=30,
                )
                res.raise_for_status()
                results = res.json()
                for result in results:
                    query = result.get("query")
                    # prefer top-level cadd.phred (if available), else fallback to dbnsfp.cadd.phred
                    score = None
                    if isinstance(result, dict):
                        cadd = result.get("cadd")
                        if isinstance(cadd, dict):
                            score = cadd.get("phred", score)
                        if score is None:
                            dbnsfp = result.get("dbnsfp")
                            if isinstance(dbnsfp, dict):
                                cadd2 = dbnsfp.get("cadd")
                                if isinstance(cadd2, dict):
                                    score = cadd2.get("phred", score)
                    if score is None:
                        score = np.nan
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

    # --------------- MyVariant annotations (snpeff/vcf) -----------
    def _load_myvariant_cleaned(self, cleaned_path: str) -> Dict[str, dict]:
        annotations: Dict[str, dict] = {}
        try:
            with open(cleaned_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and len(obj) == 1:
                            vid, data = next(iter(obj.items()))
                            annotations[vid] = data
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return annotations

    @staticmethod
    def _build_variant_ids_from_df(df: pd.DataFrame, snv_only: bool = True) -> Set[str]:
        required = {"CHR", "START", "REF", "ALT"}
        if not required.issubset(df.columns):
            return set()
        tmp = df[list(required)].copy()
        tmp["START"] = pd.to_numeric(tmp["START"], errors="coerce")
        tmp["CHR"] = (
            tmp["CHR"].astype(str).str.replace(r"^chr", "", regex=True).str.strip()
        )
        tmp["REF"] = tmp["REF"].astype(str).str.strip()
        tmp["ALT"] = tmp["ALT"].astype(str).str.strip()
        if snv_only:
            snv_mask = (tmp["REF"].str.len() == 1) & (tmp["ALT"].str.len() == 1)
            tmp = tmp.loc[snv_mask]
        tmp = tmp.dropna(subset=["CHR", "START", "REF", "ALT"])
        if tmp.empty:
            return set()
        vids = (
            tmp.apply(
                lambda r: f"chr{r['CHR']}:g.{int(r['START'])}{r['REF']}>{r['ALT']}",
                axis=1,
            )
            .dropna()
            .unique()
            .tolist()
        )
        return set(vids)

    def _fetch_myvariant_to_file_sync(
        self, variant_ids: Set[str], output_path: str
    ) -> None:
        """Fetch MyVariant snpeff/vcf annotations using batch POST requests and write JSONL of {vid: payload}."""
        ids = list(variant_ids)
        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(0, len(ids), 1000):
                batch = ids[i : i + 1000]
                try:
                    res = requests.post(
                        "https://myvariant.info/v1/variant",
                        data={"ids": ",".join(batch), "fields": "snpeff,vcf"},
                        headers={"content-type": "application/x-www-form-urlencoded"},
                        timeout=60,
                    )
                    res.raise_for_status()
                    results = res.json()
                    for item in results:
                        vid = item.get("query")
                        if vid:
                            f.write(json.dumps({vid: item}, ensure_ascii=False) + "\n")
                except requests.exceptions.RequestException as e:
                    print(f"  [ERREUR API] MyVariant lot {i//1000 + 1}: {e}")

    def ensure_myvariant_cache_for_df(
        self,
        variants_df: pd.DataFrame,
        snv_only: bool = True,
        batch_temp_path: Optional[str] = None,
    ) -> None:
        """Ensure MyVariant annotations JSONL cache contains all variants from df (skip those already cached)."""
        vids = self._build_variant_ids_from_df(variants_df, snv_only=snv_only)
        if not vids:
            print("[ExternalData] Aucun variant valide trouvé pour MyVariant.")
            return
        cached = set(self.variant_annotations.keys())
        to_fetch = list(vids - cached)
        if not to_fetch:
            print(
                "[ExternalData] Toutes les annotations MyVariant sont déjà en cache (cleaned)."
            )
            return
        # Prepare paths
        os.makedirs(os.path.dirname(self.myvariant_raw_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.myvariant_cleaned_path), exist_ok=True)
        # Fetch only missing into a temp file to avoid rewriting existing raw
        tmp_out = batch_temp_path or (self.myvariant_raw_path + ".tmp")
        print(
            f"[ExternalData] Téléchargement MyVariant pour {len(to_fetch)} variants manquants..."
        )
        self._fetch_myvariant_to_file_sync(set(to_fetch), tmp_out)
        # Append to main raw jsonl and remove temp
        with open(self.myvariant_raw_path, "a", encoding="utf-8") as dst:
            try:
                with open(tmp_out, "r", encoding="utf-8") as src:
                    for line in src:
                        if line.strip():
                            dst.write(line)
            finally:
                try:
                    os.remove(tmp_out)
                except OSError:
                    pass
        # Clean entire raw into cleaned JSONL (idempotent)
        cleaner = MyVariantCleaner()
        cleaner.process_raw_jsonl(self.myvariant_raw_path, self.myvariant_cleaned_path)
        # Reload cleaned cache into memory
        self.variant_annotations = self._load_myvariant_cleaned(
            self.myvariant_cleaned_path
        )
        print(
            f"[ExternalData] Cache MyVariant prêt ({len(self.variant_annotations)} variants nettoyés)."
        )

    def get_variant_annotations(self) -> Dict[str, dict]:
        return self.variant_annotations
