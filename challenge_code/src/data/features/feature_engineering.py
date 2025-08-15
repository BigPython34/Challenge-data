"""
Feature Engineering module for AML survival analysis.

All numeric thresholds, gene sets, toggles and encodings are configured in src.config
for full experiment traceability.
"""

import re
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from ...config import (
    # Clinical
    CLINICAL_NUMERIC_COLUMNS,
    CLINICAL_RATIOS,
    CLINICAL_THRESHOLDS,
    CLINICAL_LOG_COLUMNS,
    MISSINGNESS_POLICY,
    # Cytogenetics
    CYTOGENETIC_FAVORABLE,
    CYTOGENETIC_ADVERSE,
    CYTOGENETIC_INTERMEDIATE,
    COMPLEX_KARYOTYPE_MIN_ABNORMALITIES,
    ELN_CYTO_RISK_ENCODING,
    CYTO_FEATURE_TOGGLES,
    CYTOGENETIC_COMMON_MONOSOMIES,
    CYTOGENETIC_COMMON_TRISOMIES,
    # Molecular
    ALL_IMPORTANT_GENES,
    GENE_PATHWAYS,
    MOLECULAR_FEATURE_TOGGLES,
    ELN_MOLECULAR_RISK_ENCODING,
    TP53_HIGH_VAF_THRESHOLD,
    MOLECULAR_VAF_THRESHOLDS,
    MOLECULAR_GENE_FREQ_FILTER,
    # Redundancy
    REDUNDANCY_POLICY,
    # Caps
    COMPLEX_ABNORMALITIES_CAP,
)

from src.data.data_extraction.external_data_manager import ExternalDataManager


class ClinicalFeatureEngineering:
    """Handles clinical feature creation."""

    @staticmethod
    def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
        clinical_df = df.copy()
        clinical_df = ClinicalFeatureEngineering._ensure_numeric_columns(
            clinical_df, CLINICAL_NUMERIC_COLUMNS
        )
        # Missingness indicators (config-driven)
        if MISSINGNESS_POLICY.get("create_indicators", True):
            for col in CLINICAL_NUMERIC_COLUMNS:
                if col in clinical_df.columns:
                    clinical_df[f"{col}_missing"] = clinical_df[col].isna().astype(int)
        clinical_df = ClinicalFeatureEngineering._create_clinical_ratios(clinical_df)
        clinical_df = ClinicalFeatureEngineering._create_clinical_thresholds(
            clinical_df
        )
        clinical_df = ClinicalFeatureEngineering._create_composite_scores(clinical_df)
        clinical_df = ClinicalFeatureEngineering._create_log_transformations(
            clinical_df
        )
        clinical_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        clinical_df = _apply_redundancy_policy(clinical_df)
        return clinical_df

    @staticmethod
    def _ensure_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _create_clinical_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Create clinical ratios based on config mapping."""
        ratios: Dict[str, Tuple[str, str]] = CLINICAL_RATIOS
        for ratio_name, pair in ratios.items():
            numerator, denominator = pair
            if numerator in df.columns and denominator in df.columns:
                # Guard against divide-by-zero and keep NaN for zero/absent denominator
                denom = df[denominator].replace({0: np.nan})
                df[ratio_name] = df[numerator] / denom
                df[ratio_name] = df[ratio_name].replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def _create_clinical_thresholds(df: pd.DataFrame) -> pd.DataFrame:
        for feature_name, (column, operator, threshold) in CLINICAL_THRESHOLDS.items():
            if column in df.columns:
                if operator == "<":
                    df[feature_name] = (df[column] < threshold).astype(int)
                elif operator == "<=":
                    df[feature_name] = (df[column] <= threshold).astype(int)
                elif operator == ">":
                    df[feature_name] = (df[column] > threshold).astype(int)
                elif operator == ">=":
                    df[feature_name] = (df[column] >= threshold).astype(int)
                else:
                    df[feature_name] = (df[column] == threshold).astype(int)

        return df

    @staticmethod
    def _create_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
        # Cytopenia score (0-3, based on anemia + thrombocytopenia + neutropenia)
        cytopenia_components = [
            "anemia_moderate",
            "thrombocytopenia_moderate",
            "neutropenia_moderate",
        ]
        if all(col in df.columns for col in cytopenia_components):
            df["cytopenia_score"] = df[cytopenia_components].sum(axis=1)
            df["pancytopenia"] = (df["cytopenia_score"] == 3).astype(int)

        # Proliferation score (blasts + leukocytosis)
        proliferation_components = ["high_blast_count", "leukocytosis_high"]
        if all(col in df.columns for col in proliferation_components):
            df["proliferation_score"] = df[proliferation_components].sum(axis=1)

        return df

    @staticmethod
    def _create_log_transformations(df: pd.DataFrame) -> pd.DataFrame:
        for col in CLINICAL_LOG_COLUMNS:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                vals = vals.where(vals >= 0, np.nan)
                df[f"log_{col}"] = np.log1p(vals)
        return df

    @staticmethod
    def create_center_one_hot_encoding(final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create one-hot encoding for CENTER variable.

        Parameters
        ----------
        final_df : pd.DataFrame
            DataFrame containing CENTER column

        Returns
        -------
        pd.DataFrame
            DataFrame with one-hot encoded CENTER variables
        """
        if "CENTER" not in final_df.columns:
            return final_df

        center_values = final_df["CENTER"].fillna("Unknown")
        unique_centers = sorted(center_values.unique())

        for center in unique_centers:
            column_name = f"CENTER_{center}"
            final_df[column_name] = (center_values == center).astype(int)

        final_df = final_df.drop(columns=["CENTER"])

        return final_df

    # Backward-compatible alias
    _create_center_one_hot_encoding = create_center_one_hot_encoding


class CytogeneticFeatureExtraction:
    """Extracts cytogenetic features based on ELN 2022 classification."""

    @staticmethod
    def extract_cytogenetic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
        if "CYTOGENETICS" not in df.columns or "ID" not in df.columns:
            return pd.DataFrame(
                columns=["ID"]
            )  # Retourne un DataFrame vide avec ID si pas de colonne

        result_df = pd.DataFrame(index=df.index)

        result_df["cyto_is_missing"] = df["CYTOGENETICS"].isnull().astype(int)
        # Clean and prepare cytogenetics data
        cytogenetics_clean = df["CYTOGENETICS"].fillna("").str.strip()

        # === BASIC CHROMOSOME CHARACTERISTICS ===
        result_df = CytogeneticFeatureExtraction._extract_basic_chromosome_features(
            result_df, cytogenetics_clean
        )

        # === ELN 2022 CYTOGENETIC ABNORMALITIES ===
        result_df = (
            CytogeneticFeatureExtraction._extract_eln2022_cytogenetic_abnormalities(
                result_df, cytogenetics_clean
            )
        )

        # === CYTOGENETIC COMPLEXITY METRICS ===
        result_df = CytogeneticFeatureExtraction._extract_cytogenetic_complexity(
            result_df, cytogenetics_clean
        )

        # === ELN 2022 FINAL RISK CLASSIFICATION ===
        result_df = CytogeneticFeatureExtraction._calculate_eln2022_cytogenetic_risk(
            result_df
        )

        # Drop the CYTOGENETICS column after extraction and set NaN for derived features when missing,
        # but keep the cyto_is_missing indicator intact.
        cols_to_nan = [c for c in result_df.columns if c != "cyto_is_missing"]
        result_df.loc[result_df["cyto_is_missing"] == 1, cols_to_nan] = np.nan

        # Ajout sécurisé de la colonne ID
        result_df["ID"] = df["ID"].values

        # Réorganiser pour mettre ID en premier (optionnel)
        cols = ["ID"] + [c for c in result_df.columns if c != "ID"]
        result_df = result_df[cols]

        return result_df

    @staticmethod
    def _extract_basic_chromosome_features(
        result_df: pd.DataFrame, cytogenetics: pd.Series
    ) -> pd.DataFrame:
        """Extract basic chromosome count and sex chromosome information."""
        # Chromosome count with validation
        chromosome_count = cytogenetics.str.extract(r"(\d+)")[0]
        chromosome_count = pd.to_numeric(chromosome_count, errors="coerce")
        # Keep NaN for unknown/out-of-range instead of coercing to 46 to avoid false diploid assumptions
        chromosome_count = chromosome_count.where(
            (chromosome_count >= 30) & (chromosome_count <= 80), np.nan
        )
        result_df["chromosome_count"] = chromosome_count

        # Sex chromosome determination
        result_df["sex_chromosomes"] = 0.5  # Backward-compatible numeric encoding
        xx_mask = cytogenetics.str.contains(r"\bXX\b", case=False, na=False)
        xy_mask = cytogenetics.str.contains(r"\bXY\b", case=False, na=False)
        result_df.loc[xx_mask, "sex_chromosomes"] = 0
        result_df.loc[xy_mask, "sex_chromosomes"] = 1
        # Add one-hot encoding for robustness (keeps numeric column for compatibility)
        result_df["SEX_XX"] = xx_mask.astype(int)
        result_df["SEX_XY"] = xy_mask.astype(int)
        result_df["SEX_UNKNOWN"] = (~xx_mask & ~xy_mask).astype(int)

        return result_df

    @staticmethod
    def _extract_eln2022_cytogenetic_abnormalities(
        result_df: pd.DataFrame, cytogenetics: pd.Series
    ) -> pd.DataFrame:
        """Extract specific cytogenetic abnormalities according to ELN 2022 classification."""

        def contains_pattern(pattern):
            return cytogenetics.str.contains(
                pattern, case=False, regex=True, na=False
            ).astype(int)

        # === FAVORABLE ABNORMALITIES ===
        favorable_features = {
            "t_8_21": CYTOGENETIC_FAVORABLE[0],
            "inv_16": CYTOGENETIC_FAVORABLE[1],
            "t_16_16": CYTOGENETIC_FAVORABLE[2],
            "t_15_17": CYTOGENETIC_FAVORABLE[3],
        }

        for feature_name, pattern in favorable_features.items():
            result_df[feature_name] = contains_pattern(pattern)

        result_df["any_favorable_cyto"] = result_df[
            list(favorable_features.keys())
        ].max(axis=1)

        # === INTERMEDIATE ABNORMALITIES ===
        # Normal karyotype: robust detection of 46,XX/46,XY allowing optional comma/space and clone counts in brackets
        has_normal_code = cytogenetics.str.contains(
            r"\b46[, ]?(?:XX|XY)\b", case=False, na=False
        )
        has_abnormal_markers = cytogenetics.str.contains(
            r"del\(|inv\(|t\(|add\(|der\(|ins\(|i\(|\+|-", case=False, na=False
        )
        result_df["normal_karyotype"] = (
            has_normal_code & ~has_abnormal_markers
        ).astype(int)
        result_df["trisomy_8"] = contains_pattern(CYTOGENETIC_INTERMEDIATE[1])
        # Additional intermediate markers (optional features)
        if len(CYTOGENETIC_INTERMEDIATE) > 2:
            result_df["t_9_11"] = contains_pattern(CYTOGENETIC_INTERMEDIATE[2])
        if len(CYTOGENETIC_INTERMEDIATE) > 3:
            result_df["kmt2a_rearrangement"] = contains_pattern(
                CYTOGENETIC_INTERMEDIATE[3]
            )
        # Useful extra binary markers often reported
        result_df["minus_Y"] = cytogenetics.str.contains(
            r"-Y\b", case=False, na=False
        ).astype(int)
        result_df["plus_21"] = cytogenetics.str.contains(
            r"\+21\b", case=False, na=False
        ).astype(int)

        # === ADVERSE ABNORMALITIES ===
        # Keep named columns for the first few canonical adverse patterns
        adverse_features = {
            "del_5q_or_mono5": CYTOGENETIC_ADVERSE[0],
            "monosomy_7_or_del7q": CYTOGENETIC_ADVERSE[1],
            "del_17p_or_i17q": CYTOGENETIC_ADVERSE[2],
        }

        for feature_name, pattern in adverse_features.items():
            result_df[feature_name] = contains_pattern(pattern)

        # Backward-compatible aliases (maintain old column names if downstream expects them)
        result_df["del_5q"] = result_df["del_5q_or_mono5"]
        result_df["monosomy_7"] = result_df["monosomy_7_or_del7q"]
        result_df["del_17p"] = result_df["del_17p_or_i17q"]

        # Additionally compute any adverse according to ALL configured adverse patterns
        if len(CYTOGENETIC_ADVERSE) > 0:
            # Combine all patterns into a single regex OR
            combined_pat = "(?:" + ")|(?:".join(CYTOGENETIC_ADVERSE) + ")"
            any_adv_series = cytogenetics.str.contains(
                combined_pat, case=False, regex=True, na=False
            ).astype(int)
        else:
            any_adv_series = pd.Series(0, index=cytogenetics.index)

        # any_adverse_cyto is the OR of named adverse flags and the combined config-based check
        base_any = result_df[list(adverse_features.keys())].max(axis=1)
        result_df["any_adverse_cyto"] = base_any.combine(
            any_adv_series, func=lambda a, b: int(max(a, b))
        )

        # Optionally add common monosomy/trisomy binary flags for audit/aux features
        if CYTO_FEATURE_TOGGLES.get("include_common_events", True):
            for patt in CYTOGENETIC_COMMON_MONOSOMIES:
                chrom = patt.replace("\\b", "").lstrip("-")
                col = f"mono_{chrom}"
                result_df[col] = cytogenetics.str.contains(
                    patt, case=False, regex=True, na=False
                ).astype(int)
            for patt in CYTOGENETIC_COMMON_TRISOMIES:
                chrom = patt.replace("\\b", "").lstrip("+")
                col = f"tri_{chrom}"
                result_df[col] = cytogenetics.str.contains(
                    patt, case=False, regex=True, na=False
                ).astype(int)

        return result_df

    @staticmethod
    def _extract_cytogenetic_complexity(
        result_df: pd.DataFrame, cytogenetics: pd.Series
    ) -> pd.DataFrame:
        """Extract cytogenetic complexity metrics."""

        # Helper to clean bracketed clone counts like [5] and tolerate malformed braces
        def _remove_brackets(s: str) -> str:
            if not isinstance(s, str):
                return ""
            # Remove square or curly bracketed contents, e.g., [5], {6}
            s = re.sub(r"\[[^\]]*\]", "", s)
            s = re.sub(r"\{[^\}]*\}", "", s)
            # Drop stray unmatched braces to avoid polluting tokens
            s = re.sub(r"[\[\]\{\}]", "", s)
            return s

        # Count events from a pure events string (no baseline digits or XX/XY)
        def _count_from_events_str(events_str: str) -> dict:
            events_str = events_str or ""
            return {
                "n_t": len(re.findall(r"t\(", events_str, flags=re.IGNORECASE)),
                "n_del": len(re.findall(r"del\(", events_str, flags=re.IGNORECASE)),
                "n_inv": len(re.findall(r"inv\(", events_str, flags=re.IGNORECASE)),
                "n_add": len(re.findall(r"add\(", events_str, flags=re.IGNORECASE)),
                "n_der": len(re.findall(r"der\(", events_str, flags=re.IGNORECASE)),
                "n_ins": len(re.findall(r"ins\(", events_str, flags=re.IGNORECASE)),
                "n_i": len(re.findall(r"\bi\(", events_str, flags=re.IGNORECASE)),
                "n_dic": len(re.findall(r"\bdic\(", events_str, flags=re.IGNORECASE)),
                "n_plus": len(
                    re.findall(r"\+(?:\d+|X|Y|mar)\b", events_str, flags=re.IGNORECASE)
                ),
                "n_minus": len(
                    re.findall(r"-(?:\d+|X|Y)\b", events_str, flags=re.IGNORECASE)
                ),
                "n_mar": len(re.findall(r"\+mar\b", events_str, flags=re.IGNORECASE)),
                "has_idem": (
                    1 if re.search(r"\bidem\b", events_str, flags=re.IGNORECASE) else 0
                ),
            }

        def _count_events_in_clone(clone_text: str) -> dict:
            text = _remove_brackets(clone_text)
            # Remove baseline leading tokens like '46', 'XX/XY' from the start of the clone
            tokens = [t.strip() for t in text.split(",") if t.strip()]
            # Drop the first two tokens when they look like baseline count and sex
            start_idx = 0
            if len(tokens) >= 1 and re.fullmatch(r"\d{2,}", tokens[0]):
                start_idx = 1
            if len(tokens) >= 2 and re.fullmatch(
                r"X{2}|XY", tokens[1], flags=re.IGNORECASE
            ):
                start_idx = 2
            events_str = ",".join(tokens[start_idx:])
            return _count_from_events_str(events_str)

        def _events_string_from_clone(clone_text: str) -> str:
            """Return just the events portion of a clone (without baseline), cleaned of brackets."""
            text = _remove_brackets(clone_text)
            tokens = [t.strip() for t in text.split(",") if t.strip()]
            start_idx = 0
            if len(tokens) >= 1 and re.fullmatch(r"\d{2,}", tokens[0]):
                start_idx = 1
            if len(tokens) >= 2 and re.fullmatch(
                r"X{2}|XY", tokens[1], flags=re.IGNORECASE
            ):
                start_idx = 2
            return ",".join(tokens[start_idx:])

        def count_abnormalities_and_types(cyto_string: str) -> dict:
            if not isinstance(cyto_string, str) or cyto_string.strip() == "":
                # Return zeros for all counters
                return {
                    k: 0
                    for k in [
                        "n_t",
                        "n_del",
                        "n_inv",
                        "n_add",
                        "n_der",
                        "n_ins",
                        "n_i",
                        "n_plus",
                        "n_minus",
                        "n_mar",
                        "has_idem",
                        "clone_count",
                        "total",
                    ]
                }
            # Split clones by '/'
            clones = [c.strip() for c in cyto_string.split("/") if c.strip()]
            agg = {
                "n_t": 0,
                "n_del": 0,
                "n_inv": 0,
                "n_add": 0,
                "n_der": 0,
                "n_ins": 0,
                "n_i": 0,
                "n_dic": 0,
                "n_plus": 0,
                "n_minus": 0,
                "n_mar": 0,
                "has_idem": 0,
            }
            prev_events = ""
            for clone in clones:
                events_str = _events_string_from_clone(clone)
                # Expand 'idem' clones: inherit previous events and add new ones
                idem_here = (
                    1 if re.search(r"\bidem\b", events_str, flags=re.IGNORECASE) else 0
                )
                if idem_here:
                    # Remove the token 'idem' and any adjacent punctuation
                    addon = re.sub(
                        r"\bidem\b\s*,?", "", events_str, flags=re.IGNORECASE
                    ).strip(", ")
                    expanded = prev_events
                    if addon:
                        expanded = f"{expanded},{addon}" if expanded else addon
                    c = _count_from_events_str(expanded)
                else:
                    c = _count_from_events_str(events_str)
                for k in agg:
                    agg[k] += c[k]
                # Ensure 'has_idem' reflects token presence even when expanded
                agg["has_idem"] += idem_here
                # Update prev_events for next potential 'idem' usage
                prev_events = (
                    re.sub(r"\bidem\b\s*,?", "", events_str, flags=re.IGNORECASE).strip(
                        ", "
                    )
                    or prev_events
                )
            # Derive totals
            structural = (
                agg["n_t"]
                + agg["n_del"]
                + agg["n_inv"]
                + agg["n_add"]
                + agg["n_der"]
                + agg["n_ins"]
                + agg["n_i"]
                + agg["n_dic"]
            )
            numerical = agg["n_plus"] + agg["n_minus"]
            total = structural + numerical
            # Cap to avoid extreme outliers dominating (config-driven)
            total = min(total, COMPLEX_ABNORMALITIES_CAP)
            agg.update(
                {
                    "clone_count": len(clones),
                    "total": total,
                    "structural_count": structural,
                    "numerical_count": numerical,
                }
            )
            return agg

        # Apply and expand into columns
        counts_df = cytogenetics.apply(count_abnormalities_and_types).apply(pd.Series)
        # Rename for clarity
        counts_df.rename(
            columns={
                "total": "num_cyto_abnormalities",
            },
            inplace=True,
        )
        # Merge into result
        for col in counts_df.columns:
            result_df[col] = counts_df[col].fillna(0).astype(int)

        result_df["complex_karyotype"] = (
            result_df["num_cyto_abnormalities"] >= COMPLEX_KARYOTYPE_MIN_ABNORMALITIES
        ).astype(int)
        # Provide a stabilized, monotonic transform without hard capping information
        result_df["log_num_cyto_abnormalities"] = np.log1p(
            result_df["num_cyto_abnormalities"].astype(float)
        )
        result_df["has_derivative_chromosome"] = cytogenetics.str.contains(
            r"\bder\(", case=False, na=False
        ).astype(int)
        result_df["has_marker_chromosome"] = cytogenetics.str.contains(
            r"\+mar\b", case=False, na=False
        ).astype(int)

        return result_df

    @staticmethod
    def _calculate_eln2022_cytogenetic_risk(result_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final ELN 2022 cytogenetic risk with configurable encoding."""
        favorable = result_df.get("any_favorable_cyto", 0).fillna(0).astype(int)
        adverse = result_df.get("any_adverse_cyto", 0).fillna(0).astype(
            int
        ) | result_df.get("complex_karyotype", 0).fillna(0).astype(int)
        label = pd.Series(1, index=result_df.index)  # default intermediate
        label = label.mask(favorable == 1, 0)
        label = label.mask(adverse == 1, 2)

        encoding = ELN_CYTO_RISK_ENCODING or {
            "encode_as": "ordinal",
            "weights": {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0},
        }
        if encoding.get("encode_as", "ordinal") == "one_hot":
            result_df["eln_cyto_favorable"] = (label == 0).astype(int)
            result_df["eln_cyto_intermediate"] = (label == 1).astype(int)
            result_df["eln_cyto_adverse"] = (label == 2).astype(int)
        else:
            w = encoding.get(
                "weights", {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0}
            )
            mapping = {
                0: w.get("favorable", 0.0),
                1: w.get("intermediate", 0.7),
                2: w.get("adverse", 1.0),
            }
            result_df["eln_cyto_risk"] = label.map(mapping).astype(float)
        return result_df


class MolecularFeatureExtraction:
    """Creates molecular features based on ELN 2022 prognostic mutations."""

    @staticmethod
    def extract_molecular_risk_features(
        df: pd.DataFrame,
        maf_df: pd.DataFrame,
        important_genes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        important_genes = important_genes or ALL_IMPORTANT_GENES

        # Frequency filter (optional)
        cfg = MOLECULAR_GENE_FREQ_FILTER or {}
        if cfg.get("enabled", False):
            min_total = int(cfg.get("min_total_count", 5))
            ref = cfg.get("reference", "reports")
            if ref == "reports":
                train_p = cfg.get("train_counts_path")
                test_p = cfg.get("test_counts_path")
                counts = []
                for pth in [train_p, test_p]:
                    if pth and os.path.exists(pth):
                        dfc = pd.read_csv(pth)
                        # expected columns: GENE,count
                        if "GENE" in dfc.columns and "count" in dfc.columns:
                            counts.append(dfc.set_index("GENE")["count"])
                if counts:
                    total_counts = counts[0]
                    for s in counts[1:]:
                        total_counts = total_counts.add(s, fill_value=0)
                    keep = {g for g, c in total_counts.items() if c >= min_total}
                    important_genes = [g for g in important_genes if g in keep]
            elif ref == "current" and maf_df is not None and not maf_df.empty:
                gene_col = None
                for cand in ["GENE", "Hugo_Symbol", "Gene", "Gene_Symbol"]:
                    if cand in maf_df.columns:
                        gene_col = cand
                        break
                if gene_col:
                    vc = maf_df[gene_col].value_counts()
                    keep = set(vc[vc >= min_total].index.tolist())
                    important_genes = [g for g in important_genes if g in keep]

        if df is None or df.empty:
            return pd.DataFrame()

        # Initialize molecular features dataframe
        molecular_df = pd.DataFrame({"ID": df["ID"].astype(str).unique()})

        # === BINARY MUTATION STATUS ===
        molecular_df = MolecularFeatureExtraction._extract_binary_mutations(
            molecular_df, maf_df, important_genes
        )

        # === VAF-BASED FEATURES ===
        molecular_df = MolecularFeatureExtraction._extract_vaf_features(
            molecular_df, maf_df
        )

        # === MUTATION TYPE CLASSIFICATION ===
        molecular_df = MolecularFeatureExtraction._extract_mutation_types(
            molecular_df, maf_df
        )

        # === PATHWAY-LEVEL ALTERATIONS ===
        molecular_df = MolecularFeatureExtraction._extract_pathway_alterations(
            molecular_df
        )

        # === CLINICALLY RELEVANT CO-MUTATIONS ===
        molecular_df = MolecularFeatureExtraction._extract_comutation_patterns(
            molecular_df
        )

        # === ELN 2022 MOLECULAR RISK CLASSIFICATION ===
        molecular_df = MolecularFeatureExtraction._calculate_eln2022_molecular_risk(
            molecular_df
        )

        return molecular_df

    @staticmethod
    # Version corrigée et plus robuste

    def _extract_binary_mutations(molecular_df, maf_df, important_genes):
        if molecular_df is None or molecular_df.empty:
            return molecular_df

        # Normalize ID as string
        molecular_df["ID"] = molecular_df["ID"].astype(str)
        if maf_df is None or maf_df.empty:
            for gene in important_genes:
                molecular_df[f"mut_{gene}"] = 0
            return molecular_df

        # Flexible gene column detection
        gene_col = None
        for cand in ["GENE", "Hugo_Symbol", "Gene", "Gene_Symbol"]:
            if cand in maf_df.columns:
                gene_col = cand
                break
        if gene_col is None:
            for gene in important_genes:
                molecular_df[f"mut_{gene}"] = 0
            return molecular_df

        maf = maf_df.copy()
        maf["ID"] = maf["ID"].astype(str)
        piv = (
            maf.loc[maf[gene_col].isin(important_genes), ["ID", gene_col]]
            .dropna()
            .assign(val=1)
            .drop_duplicates()
            .pivot_table(index="ID", columns=gene_col, values="val", fill_value=0)
            .reset_index()
        )
        piv.columns = ["ID", *[f"mut_{c}" for c in piv.columns if c != "ID"]]
        molecular_df = molecular_df.merge(piv, on="ID", how="left")
        # Ensure all configured genes exist as columns
        for g in important_genes:
            col = f"mut_{g}"
            if col not in molecular_df.columns:
                molecular_df[col] = 0
        # Fill and cast
        gene_cols = [f"mut_{g}" for g in important_genes]
        molecular_df[gene_cols] = molecular_df[gene_cols].fillna(0).astype(int)
        return molecular_df

    @staticmethod
    def _extract_vaf_features(
        molecular_df: pd.DataFrame, maf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract VAF-based features for prognostically important genes."""
        if maf_df is None or maf_df.empty or molecular_df is None or molecular_df.empty:
            return molecular_df
        maf = maf_df.copy()
        maf["ID"] = maf["ID"].astype(str)
        vaf_col = None
        for cand in ["VAF", "Variant_Allele_Frequency", "tumor_f"]:
            if cand in maf.columns:
                vaf_col = cand
                break
        if vaf_col is None:
            return molecular_df
        maf[vaf_col] = pd.to_numeric(maf[vaf_col], errors="coerce")

        def max_vaf_for_gene(gene: str) -> pd.Series:
            rows = maf[maf["GENE"] == gene]
            return rows.groupby("ID")[vaf_col].max()

        # Key genes with specific high-VAF rules
        vaf_thresholds = dict(MOLECULAR_VAF_THRESHOLDS)
        vaf_thresholds["TP53"] = TP53_HIGH_VAF_THRESHOLD
        for gene, thr in vaf_thresholds.items():
            if gene in maf["GENE"].values:
                max_vaf = max_vaf_for_gene(gene)
                molecular_df[f"vaf_max_{gene}"] = (
                    molecular_df["ID"].map(max_vaf).fillna(0.0)
                )
                if gene == "FLT3":
                    # ITD text detection as positive regardless of VAF availability
                    gene_rows = maf[maf["GENE"] == gene]
                    has_itd = (
                        gene_rows[
                            [
                                c
                                for c in ["EFFECT", "PROTEIN_CHANGE"]
                                if c in gene_rows.columns
                            ]
                        ]
                        .astype(str)
                        .apply(
                            lambda s: s.str.contains(
                                "ITD|internal tandem duplication", case=False, na=False
                            )
                        )
                        .any(axis=1)
                    )
                    itd_ids = set(gene_rows.loc[has_itd, "ID"].astype(str))
                    high = (molecular_df[f"vaf_max_{gene}"] >= thr) | molecular_df[
                        "ID"
                    ].isin(itd_ids)
                else:
                    high = molecular_df[f"vaf_max_{gene}"] >= thr
                molecular_df[f"{gene}_high_VAF"] = high.astype(int)
            else:
                molecular_df[f"vaf_max_{gene}"] = 0.0
                molecular_df[f"{gene}_high_VAF"] = 0
        return molecular_df

    @staticmethod
    def _extract_mutation_types(
        molecular_df: pd.DataFrame, maf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract mutation type classifications for key genes."""
        # TP53 truncating mutations
        if "TP53" in maf_df["GENE"].values:
            tp53_patients = maf_df[maf_df["GENE"] == "TP53"]
            # Broaden patterns to catch common variant ontology names
            truncating_pattern = (
                r"nonsense|frameshift|frameshift_variant|splice_site|"
                r"splice_acceptor|splice_donor|stop_gained|stop_lost|start_lost"
            )
            tp53_truncating = tp53_patients[
                tp53_patients["EFFECT"]
                .astype(str)
                .str.contains(truncating_pattern, case=False, na=False)
            ]["ID"].unique()
            molecular_df["TP53_truncating"] = molecular_df.index.isin(
                tp53_truncating
            ).astype(int)
        else:
            molecular_df["TP53_truncating"] = 0

        # CEBPA biallelic mutations
        if "CEBPA" in maf_df["GENE"].values:
            cebpa_counts = maf_df[maf_df["GENE"] == "CEBPA"]["ID"].value_counts()
            biallelic_patients = cebpa_counts[cebpa_counts >= 2].index
            molecular_df["CEBPA_biallelic"] = molecular_df.index.isin(
                biallelic_patients
            ).astype(int)
        else:
            molecular_df["CEBPA_biallelic"] = 0

        # FLT3 mutation subtypes: ITD and TKD (e.g., D835/I836)
        if "FLT3" in maf_df["GENE"].values:
            flt3_rows = maf_df[maf_df["GENE"] == "FLT3"].copy()
            for col in ["EFFECT", "PROTEIN_CHANGE"]:
                if col in flt3_rows.columns:
                    flt3_rows[col] = flt3_rows[col].astype(str)
            has_itd = (
                flt3_rows[
                    [c for c in ["EFFECT", "PROTEIN_CHANGE"] if c in flt3_rows.columns]
                ]
                .apply(
                    lambda s: s.str.contains(
                        r"ITD|internal tandem duplication", case=False, na=False
                    )
                )
                .any(axis=1)
            )
            itd_ids = set(flt3_rows.loc[has_itd, "ID"].astype(str))
            tkd_pat = (
                flt3_rows["PROTEIN_CHANGE"].str.contains(
                    r"D835|I836", case=False, na=False
                )
                if "PROTEIN_CHANGE" in flt3_rows.columns
                else pd.Series(False, index=flt3_rows.index)
            )
            tkd_ids = set(flt3_rows.loc[tkd_pat, "ID"].astype(str))
            molecular_df["FLT3_ITD"] = (
                molecular_df.index.astype(str).isin(itd_ids).astype(int)
            )
            molecular_df["FLT3_TKD"] = (
                molecular_df.index.astype(str).isin(tkd_ids).astype(int)
            )
        else:
            molecular_df["FLT3_ITD"] = 0
            molecular_df["FLT3_TKD"] = 0

        return molecular_df

    @staticmethod
    def _extract_comutation_patterns(molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Extract clinically relevant co-mutation patterns."""
        # NPM1+/FLT3- : Favorable prognosis
        molecular_df["NPM1_pos_FLT3_neg"] = (
            (molecular_df.get("mut_NPM1", 0) == 1)
            & (molecular_df.get("mut_FLT3", 0) == 0)
        ).astype(int)

        return molecular_df

    @staticmethod
    def _extract_pathway_alterations(molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Extract pathway-level alteration features."""
        toggles = MOLECULAR_FEATURE_TOGGLES.get(
            "pathway_features", {"binary": True, "count": False}
        )
        for pathway_name, pathway_genes in GENE_PATHWAYS.items():
            pathway_columns = [
                f"mut_{gene}"
                for gene in pathway_genes
                if f"mut_{gene}" in molecular_df.columns
            ]

            if pathway_columns:
                if toggles.get("binary", True):
                    molecular_df[f"{pathway_name}_altered"] = (
                        molecular_df[pathway_columns].sum(axis=1) > 0
                    ).astype(int)
                if toggles.get("count", False):
                    molecular_df[f"{pathway_name}_count"] = molecular_df[
                        pathway_columns
                    ].sum(axis=1)

        return molecular_df

    @staticmethod
    def _calculate_eln2022_molecular_risk(molecular_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ELN 2022 molecular risk classification."""
        # Worst-rule with configurable encoding
        favorable = (molecular_df.get("mut_NPM1", 0) == 1).astype(int)
        adverse_cols = [
            c
            for c in [
                "mut_TP53",
                "mut_ASXL1",
                "mut_RUNX1",
                "mut_BCOR",
                "mut_EZH2",
                "mut_SF3B1",
                "mut_SRSF2",
                "mut_U2AF1",
                "mut_ZRSR2",
                "mut_STAG2",
            ]
            if c in molecular_df.columns
        ]
        adverse = (
            (molecular_df[adverse_cols].sum(axis=1) > 0).astype(int)
            if adverse_cols
            else 0
        )

        label = pd.Series(1, index=molecular_df.index)
        label = label.mask(favorable == 1, 0)
        label = label.mask(adverse == 1, 2)

        encoding = ELN_MOLECULAR_RISK_ENCODING or {
            "encode_as": "ordinal",
            "weights": {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0},
        }
        if encoding.get("encode_as", "ordinal") == "one_hot":
            molecular_df["eln_mol_favorable"] = (label == 0).astype(int)
            molecular_df["eln_mol_intermediate"] = (label == 1).astype(int)
            molecular_df["eln_mol_adverse"] = (label == 2).astype(int)
        else:
            w = encoding.get(
                "weights", {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0}
            )
            mapping = {
                0: w.get("favorable", 0.0),
                1: w.get("intermediate", 0.7),
                2: w.get("adverse", 1.0),
            }
            molecular_df["eln_mol_risk"] = label.map(mapping).astype(float)
        return molecular_df

    @staticmethod
    def create_molecular_burden_features(maf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mutation burden statistics.
        """
        mutation_counts = (
            maf_df.groupby("ID").size().reset_index(name="total_mutations")
        )
        if "VAF" in maf_df.columns:
            maf_df = maf_df.copy()
            maf_df["VAF"] = pd.to_numeric(maf_df["VAF"], errors="coerce")
            vaf_stats = (
                maf_df.groupby("ID")["VAF"]
                .agg(
                    [
                        ("vaf_mean", "mean"),
                        ("vaf_median", "median"),
                        ("vaf_max", "max"),
                        ("vaf_std", "std"),
                    ]
                )
                .reset_index()
            )
            vaf_stats["vaf_std"] = vaf_stats["vaf_std"].fillna(0)
            high_vaf_counts = (
                maf_df[maf_df["VAF"] > 0.4]
                .groupby("ID")
                .size()
                .reset_index(name="high_vaf_mutations")
            )
        else:
            # Provide zeros when VAF is unavailable
            ids = mutation_counts["ID"]
            vaf_stats = pd.DataFrame(
                {
                    "ID": ids,
                    "vaf_mean": 0.0,
                    "vaf_median": 0.0,
                    "vaf_max": 0.0,
                    "vaf_std": 0.0,
                }
            )
            high_vaf_counts = pd.DataFrame({"ID": ids, "high_vaf_mutations": 0})

        burden_df = mutation_counts.merge(vaf_stats, on="ID", how="left")
        burden_df = burden_df.merge(high_vaf_counts, on="ID", how="left")
        burden_df["high_vaf_mutations"] = (
            burden_df["high_vaf_mutations"].fillna(0).astype(int)
        )
        burden_df["high_vaf_ratio"] = (
            burden_df["high_vaf_mutations"] / burden_df["total_mutations"]
        )
        burden_df["high_vaf_ratio"] = burden_df["high_vaf_ratio"].fillna(0)
        return burden_df

    @staticmethod
    def get_favorable_molecular_mask(molecular_df: pd.DataFrame) -> pd.Series:
        """Get boolean mask for favorable molecular features."""
        favorable_conditions = []

        if "mut_NPM1" in molecular_df.columns:
            npm1_favorable = (molecular_df["mut_NPM1"] == 1) & (
                molecular_df.get("mut_FLT3", 0) == 0
            )
            favorable_conditions.append(npm1_favorable)

        if "CEBPA_biallelic" in molecular_df.columns:
            favorable_conditions.append(molecular_df["CEBPA_biallelic"] == 1)

        return (
            pd.concat(favorable_conditions, axis=1).any(axis=1)
            if favorable_conditions
            else None
        )

    @staticmethod
    def get_adverse_molecular_mask(molecular_df: pd.DataFrame) -> pd.Series:
        """Get boolean mask for adverse molecular features."""
        adverse_conditions = []
        adverse_genes = ["TP53", "ASXL1", "RUNX1", "BCOR", "EZH2"]

        for gene in adverse_genes:
            if f"mut_{gene}" in molecular_df.columns:
                adverse_conditions.append(molecular_df[f"mut_{gene}"] == 1)

        if "FLT3_high_VAF" in molecular_df.columns:
            adverse_conditions.append(molecular_df["FLT3_high_VAF"] == 1)

        return (
            pd.concat(adverse_conditions, axis=1).any(axis=1)
            if adverse_conditions
            else None
        )

    @staticmethod
    def _extract_impact_features(
        base_df: pd.DataFrame, maf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Crée des features agrégées par patient basées sur la colonne 'IMPACT'.
        """
        if "IMPACT" not in maf_df.columns:
            return pd.DataFrame(index=base_df["ID"].unique())

        # S'assurer que la colonne est de type catégoriel pour un one-hot encoding propre
        maf_df["IMPACT"] = maf_df["IMPACT"].astype("category")

        # One-hot encode la colonne IMPACT
        impact_dummies = pd.get_dummies(
            maf_df["IMPACT"], prefix="impact", dummy_na=False
        )

        # Concaténer l'ID pour le groupby
        impact_with_id = pd.concat([maf_df["ID"], impact_dummies], axis=1)

        # Agréger par patient : on compte combien de mutations de chaque type d'impact il a
        impact_features = impact_with_id.groupby("ID").sum()

        return impact_features.reset_index()

    @staticmethod
    def create_all_molecular_features(
        base_df: pd.DataFrame,
        maf_df: pd.DataFrame,
        external_data_manager: ExternalDataManager,
    ) -> pd.DataFrame:
        """
        Orchestrateur final pour créer TOUTES les features moléculaires, y compris externes et d'impact.
        """
        # --- Enrichir maf_df avec les données de COSMIC/OncoKB ---
        if not external_data_manager.gene_info_data.empty:
            maf_df = maf_df.merge(
                external_data_manager.get_gene_info(),
                left_on="GENE",
                right_index=True,
                how="left",
            ).fillna({"is_oncogene": 0, "is_tumor_suppressor": 0})

        # --- Extraction des différents types de features ---
        risk_features = MolecularFeatureExtraction.extract_molecular_risk_features(
            base_df, maf_df
        )
        burden_features = MolecularFeatureExtraction.create_molecular_burden_features(
            maf_df
        )
        impact_features = MolecularFeatureExtraction._extract_impact_features(
            base_df, maf_df
        )

        # Créer des features agrégées à partir des données externes (oncogene/tsg)
        external_features = pd.DataFrame(index=base_df["ID"].unique())
        if "is_oncogene" in maf_df.columns:
            oncogene_counts = maf_df.groupby("ID")["is_oncogene"].sum()
            tsg_counts = maf_df.groupby("ID")["is_tumor_suppressor"].sum()
            external_features["num_oncogene_muts"] = external_features.index.map(
                oncogene_counts
            )
            external_features["num_tsg_muts"] = external_features.index.map(tsg_counts)
            # Also expose binary presence flags to decouple burden from presence
            external_features["any_oncogene_mut"] = (
                external_features["num_oncogene_muts"].fillna(0) > 0
            ).astype(int)
            external_features["any_tsg_mut"] = (
                external_features["num_tsg_muts"].fillna(0) > 0
            ).astype(int)

        # Agrégations génériques sur les colonnes COSMIC/MOLGEN si présentes
        # - Pour chaque colonne binaire cosmic_* ou molgen_*, on ajoute un count par patient
        #   et un flag binaire (any_*) indiquant la présence d'au moins un gène muté portant ce flag
        # - Pour cosmic_tier_min (numérique), on prend le minimum parmi les gènes mutés du patient
        cosmic_bool_cols = [
            c
            for c in maf_df.columns
            if (c.startswith("cosmic_") or c.startswith("molgen_"))
            and c != "cosmic_tier_min"
        ]
        # Sélectionner uniquement celles qui sont numériques (pour éviter les objets inattendus)
        cosmic_bool_cols = [
            c for c in cosmic_bool_cols if pd.api.types.is_numeric_dtype(maf_df[c])
        ]

        if cosmic_bool_cols:
            # Remplir NaN par 0 pour l'agrégation
            maf_cosmic = maf_df[["ID", *cosmic_bool_cols]].copy()
            maf_cosmic[cosmic_bool_cols] = maf_cosmic[cosmic_bool_cols].fillna(0)
            grp = maf_cosmic.groupby("ID")[cosmic_bool_cols]
            counts = grp.sum()
            any_flags = (counts > 0).astype(int)
            # Renommer colonnes pour clarté
            counts = counts.add_prefix("").add_suffix("_count")
            any_flags = any_flags.add_prefix("any_")
            # Fusionner dans external_features
            external_features = external_features.join(counts, how="left")
            external_features = external_features.join(any_flags, how="left")

        # cosmic_tier_min: prendre le minimum par patient parmi les gènes mutés
        if "cosmic_tier_min" in maf_df.columns:
            # Ne garder que les lignes où une mutation est reportée (ID existe)
            tier_series = maf_df.groupby("ID")["cosmic_tier_min"].min()
            external_features["cosmic_min_tier_mut_genes"] = (
                external_features.index.map(tier_series)
            )
        external_features = (
            external_features.reset_index().rename(columns={"index": "ID"}).fillna(0)
        )

        # --- Fusionner tous les DataFrames moléculaires en un seul ---
        all_molecular_df = risk_features
        for df_to_merge in [burden_features, external_features, impact_features]:
            if not df_to_merge.empty:
                all_molecular_df = pd.merge(
                    all_molecular_df, df_to_merge, on="ID", how="outer"
                )

        all_molecular_df = all_molecular_df.fillna(0)
        # Redundancy pruning pass
        all_molecular_df = _apply_redundancy_policy(all_molecular_df)
        return all_molecular_df


class IntegratedFeatureEngineering:
    @staticmethod
    def combine_all_features(
        clinical_df: pd.DataFrame,
        molecular_df: Optional[pd.DataFrame] = None,
        burden_df: Optional[pd.DataFrame] = None,
        cyto_df: Optional[pd.DataFrame] = None,
        use_center_ohe: bool = False,
    ) -> pd.DataFrame:

        final_df = clinical_df.copy()
        final_df["ID"] = final_df["ID"].astype(str)

        # Fusionner les dataframes
        if molecular_df is not None and not molecular_df.empty:
            molecular_df["ID"] = molecular_df["ID"].astype(str)
            final_df = final_df.merge(molecular_df, on="ID", how="left")

        if burden_df is not None and not burden_df.empty:
            burden_df["ID"] = burden_df["ID"].astype(str)
            final_df = final_df.merge(burden_df, on="ID", how="left")

        if cyto_df is not None and not cyto_df.empty:
            # L'index de cyto_df est l'ID patient
            cyto_df.index = cyto_df.index.astype(str)
            final_df = final_df.merge(
                cyto_df, left_on="ID", right_index=True, how="left"
            )

        mutation_cols = [
            col
            for col in final_df.columns
            if col.startswith(
                ("mut_", "vaf_", "CEBPA_", "TP53_", "pathway_", "eln_molecular_risk")
            )
        ]
        final_df[mutation_cols] = final_df[mutation_cols].fillna(0)

        # S'assurer que les colonnes de comptage sont des entiers
        count_cols = [
            col
            for col in final_df.columns
            if "_count" in col or "total_mutations" in col
        ]
        final_df[count_cols] = final_df[count_cols].fillna(0).astype(int)

        # Optionally include CENTER one-hot if requested; otherwise drop CENTER for safety
        if use_center_ohe and "CENTER" in final_df.columns:
            final_df = ClinicalFeatureEngineering.create_center_one_hot_encoding(
                final_df
            )
        final_df = final_df.drop(columns=["CYTOGENETICS", "CENTER"], errors="ignore")

        # Final redundancy pruning
        final_df = _apply_redundancy_policy(final_df)
        return final_df


# -----------------
# Redundancy cleanup
# -----------------
def _apply_redundancy_policy(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    policy = REDUNDANCY_POLICY or {}
    drop_cols: List[str] = []

    # Drop numeric sex encoding when one-hot is present
    if policy.get("drop_sex_numeric_if_ohe", True):
        if {"SEX_XX", "SEX_XY"}.issubset(
            df.columns
        ) and "sex_chromosomes" in df.columns:
            drop_cols.append("sex_chromosomes")

    # Drop *_count when *_altered exists
    if policy.get("drop_count_when_binary_exists", True):
        for c in df.columns:
            if c.endswith("_count"):
                alt = c.replace("_count", "_altered")
                if alt in df.columns:
                    drop_cols.append(c)

    # Prune missingness indicators except whitelisted
    if policy.get("prune_missingness_indicators", True):
        keep = set(MISSINGNESS_POLICY.get("keep_columns", []))
        miss = [c for c in df.columns if c.endswith("_missing")]
        for c in miss:
            base = c[:-8]
            if base not in keep:
                drop_cols.append(c)

    # Explicit drops
    for c in policy.get("explicit_drop", []):
        if c in df.columns:
            drop_cols.append(c)

    if drop_cols:
        df = df.drop(columns=sorted(set(drop_cols)), errors="ignore")
    return df
