import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
from ...config import (
    # Clinical
    CLINICAL_NUMERIC_COLUMNS,
    CLINICAL_RATIOS,
    CLINICAL_THRESHOLDS,
    CLINICAL_LOG_COLUMNS,
    MISSINGNESS_POLICY,
    CREATE_LOG_COLUMNS,
    COMPLEX_KARYOTYPE_MIN_ABNORMALITIES,
)
from .pruning import _apply_redundancy_policy


class ClinicalFeatureEngineering:
    """Gère la création de caractéristiques à partir des données cliniques."""

    # Le code de cette classe reste inchangé car il était déjà excellent.
    # Je le reproduis ici pour que le fichier soit complet.
    @staticmethod
    def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
        clinical_df = df.copy()
        clinical_df = ClinicalFeatureEngineering._ensure_numeric_columns(
            clinical_df, CLINICAL_NUMERIC_COLUMNS
        )
        if MISSINGNESS_POLICY.get("create_indicators", True):
            for col in CLINICAL_NUMERIC_COLUMNS:
                if col in clinical_df.columns:
                    clinical_df[f"{col}_missing"] = clinical_df[col].isna().astype(int)
        clinical_df = ClinicalFeatureEngineering._create_clinical_ratios(clinical_df)
        clinical_df = ClinicalFeatureEngineering._create_clinical_thresholds(
            clinical_df
        )
        clinical_df = ClinicalFeatureEngineering._create_composite_scores(clinical_df)
        if CREATE_LOG_COLUMNS:
            clinical_df = ClinicalFeatureEngineering._create_log_transformations(
                clinical_df
            )
        clinical_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return clinical_df

    @staticmethod
    def _ensure_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _create_clinical_ratios(df: pd.DataFrame) -> pd.DataFrame:
        for ratio_name, (num, den) in CLINICAL_RATIOS.items():
            if num in df.columns and den in df.columns:
                denom = df[den].replace({0: np.nan})
                df[ratio_name] = df[num] / denom
        return df

    @staticmethod
    def _create_clinical_thresholds(df: pd.DataFrame) -> pd.DataFrame:
        for f_name, (col, op, th) in CLINICAL_THRESHOLDS.items():
            if col in df.columns:
                if op == "<":
                    df[f_name] = (df[col] < th).astype(int)
                elif op == "<=":
                    df[f_name] = (df[col] <= th).astype(int)
                elif op == ">":
                    df[f_name] = (df[col] > th).astype(int)
                elif op == ">=":
                    df[f_name] = (df[col] >= th).astype(int)
                else:
                    df[f_name] = (df[col] == th).astype(int)
        return df

    @staticmethod
    def _create_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
        cytopenia_components = [
            "anemia_moderate",
            "thrombocytopenia_moderate",
            "neutropenia_moderate",
        ]
        if all(col in df.columns for col in cytopenia_components):
            df["cytopenia_score"] = df[cytopenia_components].sum(axis=1)
            df["pancytopenia"] = (df["cytopenia_score"] == 3).astype(int)
        proliferation_components = ["high_blast_count", "leukocytosis_high"]
        if all(col in df.columns for col in proliferation_components):
            df["proliferation_score"] = df[proliferation_components].sum(axis=1)
        return df

    @staticmethod
    def _create_log_transformations(df: pd.DataFrame) -> pd.DataFrame:
        for col in CLINICAL_LOG_COLUMNS:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                df[f"log_{col}"] = np.log1p(vals.where(vals >= 0))
        return df

    @staticmethod
    def create_center_one_hot_encoding(final_df: pd.DataFrame) -> pd.DataFrame:
        if "CENTER" not in final_df.columns:
            return final_df
        center_ohe = pd.get_dummies(final_df["CENTER"], prefix="CENTER", dummy_na=True)
        final_df = pd.concat([final_df.drop("CENTER", axis=1), center_ohe], axis=1)
        return final_df


class CytogeneticFeatureExtraction:
    """
    Version finale, robuste et simplifiée après plusieurs cycles d'audit.
    Correction critique du bug de référence mémoire et de la séparation des clones.
    """

    @staticmethod
    def _count_events_in_text(text: str) -> Dict[str, int]:
        if not isinstance(text, str):
            text = ""
        counts = {
            k: 0
            for k in [
                "n_t",
                "n_del",
                "n_inv",
                "n_add",
                "n_der",
                "n_ins",
                "n_i",
                "n_dic",
                "n_plus",
                "n_minus",
                "n_mar",
                "n_ring",
                "n_dmin",
            ]
        }
        tokens = text.replace(";", ",").split(",")
        for token in tokens:
            token = token.strip().lower()
            if not token:
                continue
            if token.startswith("t("):
                counts["n_t"] += 1
            elif token.startswith("del(") or token.startswith("del,"):
                counts["n_del"] += 1
            elif token.startswith("inv("):
                counts["n_inv"] += 1
            elif token.startswith("add("):
                counts["n_add"] += 1
            elif token.startswith("der("):
                counts["n_der"] += 1
            elif token.startswith("ins("):
                counts["n_ins"] += 1
            elif token.startswith("i("):
                counts["n_i"] += 1
            elif token.startswith("dic("):
                counts["n_dic"] += 1
            elif token.startswith("r("):
                counts["n_ring"] += 1
            elif token.startswith("+mar"):
                counts["n_mar"] += 1
            elif "dmin" in token:
                counts["n_dmin"] += 1
            elif token.startswith("+"):
                counts["n_plus"] += 1
            elif token.startswith("-"):
                counts["n_minus"] += 1
        return counts

    @staticmethod
    def _parse_and_count_all_clones(cyto_string: str) -> Dict:
        default_counts = {
            k: 0
            for k in [
                "n_t",
                "n_del",
                "n_inv",
                "n_add",
                "n_der",
                "n_ins",
                "n_i",
                "n_dic",
                "n_plus",
                "n_minus",
                "n_mar",
                "n_ring",
                "n_dmin",
            ]
        }
        default_output = {
            **default_counts,
            "has_idem": 0,
            "clone_count": 0,
            "total_cell_count": 0,
            "main_clone_cell_count": 0,
            "main_clone_abnormality_count": 0,
            "num_cyto_abnormalities": 0,
        }

        if pd.isna(cyto_string):
            return default_output

        text = str(cyto_string).strip()
        if re.fullmatch(r"normal", text, re.IGNORECASE):
            return {
                **default_output,
                "clone_count": 1,
                "total_cell_count": 1,
                "main_clone_cell_count": 1,
            }

        if not re.search(r"^\s*\d{1,2}", text):
            if re.search(r">=3|>3|complex|multiple abnormalities", text, re.IGNORECASE):
                total_abns = 3
                return {
                    **default_output,
                    "clone_count": 1,
                    "total_cell_count": 1,
                    "main_clone_cell_count": 1,
                    "main_clone_abnormality_count": total_abns,
                    "num_cyto_abnormalities": total_abns,
                }
            if re.search(r"del\s*y", text, re.IGNORECASE):
                text += ",-y"
            if re.search(r"del\s*5", text, re.IGNORECASE):
                text += ",del(5)"
            if re.search(r"del\s*7", text, re.IGNORECASE):
                text += ",del(7)"
            counts = CytogeneticFeatureExtraction._count_events_in_text(text)
            total_abns = sum(counts.values())
            return {
                **default_output,
                **counts,
                "clone_count": 1,
                "total_cell_count": 1,
                "main_clone_cell_count": 1,
                "main_clone_abnormality_count": total_abns,
                "num_cyto_abnormalities": total_abns,
            }

        # --- CORRECTION CRITIQUE ICI ---
        # On ne split plus sur ';', qui est trop ambigu.
        raw_clones = re.split(r"\s*/\s*|\s{2,}", text)

        clones_data = []
        last_clone_counts = default_counts.copy()
        has_idem_flag = 0

        for clone_text in raw_clones:
            if not (clone_text := clone_text.strip()):
                continue

            cell_count_match = re.search(
                r"\[\s*(?:cp)?\s*(\d+)(?:/\d+)?\s*\]$", clone_text
            )
            cell_count = int(cell_count_match.group(1)) if cell_count_match else 1
            clone_text = (
                clone_text[: cell_count_match.start()].strip()
                if cell_count_match
                else clone_text
            )

            current_counts = (
                last_clone_counts.copy()
                if "idem" in clone_text.lower()
                else default_counts.copy()
            )
            events_text = re.sub(
                r"^\s*\d{1,2}(?:-\d{1,2})?\s*,\s*X[XY]*", "", clone_text, 1
            ).strip(", ")

            if "idem" in events_text.lower():
                has_idem_flag = 1
                events_text = re.sub(
                    r"idem", "", events_text, flags=re.IGNORECASE
                ).strip(", ")

            new_events_counts = CytogeneticFeatureExtraction._count_events_in_text(
                events_text
            )
            for key, value in new_events_counts.items():
                current_counts[key] += value

            clones_data.append(
                {
                    "counts": current_counts.copy(),
                    "cell_count": cell_count,
                    "abn_count": sum(current_counts.values()),
                }
            )
            last_clone_counts = current_counts

        if not clones_data:
            return default_output

        total_abnormalities_weighted = sum(
            clone["abn_count"] * clone["cell_count"] for clone in clones_data
        )
        total_cells = sum(c["cell_count"] for c in clones_data)
        avg_abnormalities = (
            total_abnormalities_weighted / total_cells if total_cells > 0 else 0
        )

        most_complex_clone = max(clones_data, key=lambda c: c["abn_count"])
        main_cell_clone = max(clones_data, key=lambda c: c["cell_count"])

        return {
            **most_complex_clone["counts"],
            "has_idem": has_idem_flag,
            "clone_count": len(clones_data),
            "total_cell_count": total_cells,
            "main_clone_cell_count": main_cell_clone["cell_count"],
            "main_clone_abnormality_count": main_cell_clone["abn_count"],
            "num_cyto_abnormalities": avg_abnormalities,
        }

    @staticmethod
    def extract_cytogenetic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
        if "CYTOGENETICS" not in df.columns:
            return pd.DataFrame(index=df.index)

        cyto_string_series = df["CYTOGENETICS"]
        parsed_data = cyto_string_series.apply(
            CytogeneticFeatureExtraction._parse_and_count_all_clones
        )
        counts_df = pd.DataFrame(parsed_data.tolist(), index=df.index)

        result_df = pd.DataFrame({"ID": df["ID"]}, index=df.index)
        result_df["cyto_is_missing"] = cyto_string_series.isnull().astype(int)

        chromosome_count = cyto_string_series.str.extract(
            r"^\s*(\d+)", expand=False
        ).astype(float)
        result_df["chromosome_count"] = chromosome_count
        result_df["hypodiploidy"] = (chromosome_count < 46).astype(int)
        result_df["hyperdiploidy"] = (chromosome_count >= 49).astype(int)
        result_df["near_triploidy"] = (
            (chromosome_count >= 60) & (chromosome_count <= 80)
        ).astype(int)
        result_df["sex_chromosomes"] = np.nan
        result_df.loc[
            cyto_string_series.str.contains(r"XX", na=False, case=False),
            "sex_chromosomes",
        ] = 0
        result_df.loc[
            cyto_string_series.str.contains(r"XY", na=False, case=False),
            "sex_chromosomes",
        ] = 1
        result_df["SEX_XX"] = (result_df["sex_chromosomes"] == 0).astype(int)
        result_df["SEX_XY"] = (result_df["sex_chromosomes"] == 1).astype(int)
        result_df["SEX_UNKNOWN"] = result_df["sex_chromosomes"].isna().astype(int)

        SPECIFIC_ABNORMALITIES_TO_FLAG = {
            "trisomy_8": r"\+\s*8(?![0-9])|\btris(omy|omia)?\s*8(?![0-9])",
            "t_9_11": r"t\s*\(\s*9\s*;\s*11\s*\)",
            "minus_Y": r"\bminus_y\b|-\s*y(?![a-zA-Z0-9])|\bdely\b",
            "plus_21": r"\+\s*21(?![0-9])|\btris(omy|omia)?\s*21(?![0-9])",
            "del_5q_or_mono5": r"-\s*5(?![0-9])|\b(mono|del)\w*\s*5|\bdel\s*\(\s*5\s*\)\s*\(q",
            "monosomy_7_or_del7q": r"-\s*7(?![0-9])|\b(mono|del)\w*\s*7|\bdel\s*\(\s*7\s*\)\s*\(q",
            "del_17p_or_i17q": r"del\s*\(\s*17\s*\)\s*\(p|i\s*\(\s*17\s*\)\s*\(q|17p-",
            "rearr_3q26": r"inv\(3\).*q26|t\(3;.*\).*q26|t\(.*;3\).*q26",
            "del_12p": r"del\s*\(\s*12\s*\)\s*\(p",
        }
        for name, pattern in SPECIFIC_ABNORMALITIES_TO_FLAG.items():
            result_df[name] = cyto_string_series.str.contains(
                pattern, case=False, na=False, regex=True
            ).astype(int)

        result_df = pd.concat([result_df, counts_df], axis=1)

        has_normal_clone = cyto_string_series.str.contains(
            r"^\s*46\s*,\s*X[XY]\s*(\[.+\])?$|\bnormal\b",
            case=False,
            na=False,
            regex=True,
        )
        result_df["normal_karyotype"] = (
            has_normal_clone & (result_df["num_cyto_abnormalities"] == 0)
        ).astype(int)

        result_df["del_5q"] = result_df["del_5q_or_mono5"]
        result_df["monosomy_7"] = result_df["monosomy_7_or_del7q"]
        result_df["del_17p"] = result_df["del_17p_or_i17q"]

        result_df["structural_count"] = counts_df[
            [
                "n_t",
                "n_del",
                "n_inv",
                "n_add",
                "n_der",
                "n_ins",
                "n_i",
                "n_dic",
                "n_ring",
            ]
        ].sum(axis=1)
        result_df["numerical_count"] = counts_df[["n_plus", "n_minus"]].sum(axis=1)

        result_df["complex_karyotype"] = (
            result_df["num_cyto_abnormalities"] >= COMPLEX_KARYOTYPE_MIN_ABNORMALITIES
        ).astype(int)
        result_df["has_derivative_chromosome"] = (counts_df["n_der"] > 0).astype(int)
        result_df["has_marker_chromosome"] = (counts_df["n_mar"] > 0).astype(int)
        result_df["has_ring_chromosome"] = (counts_df["n_ring"] > 0).astype(int)
        result_df["has_double_minutes"] = (counts_df["n_dmin"] > 0).astype(int)
        result_df["incomplete_karyotype"] = cyto_string_series.str.contains(
            r"\binc\b|incomplete", na=False, regex=True
        ).astype(int)

        cp_max = cyto_string_series.str.findall(r"\[cp(\d+)\]").apply(
            lambda x: max([int(n) for n in x]) if isinstance(x, list) and x else 0
        )
        result_df["cp_max"] = cp_max
        result_df["cp_any"] = (cp_max > 0).astype(int)

        autosomal_mono_count = cyto_string_series.str.count(
            r"-(?:[1-9]|1\d|2[0-2])(?![0-9])"
        ).fillna(0)
        result_df["monosomal_karyotype"] = (
            (autosomal_mono_count >= 2)
            | ((autosomal_mono_count == 1) & (result_df["structural_count"] > 0))
        ).astype(int)

        adverse_flags = [
            "del_5q_or_mono5",
            "monosomy_7_or_del7q",
            "del_17p_or_i17q",
            "rearr_3q26",
            "monosomal_karyotype",
        ]
        result_df["any_adverse_cyto"] = result_df[adverse_flags].max(axis=1)

        adverse_final = result_df["any_adverse_cyto"] | result_df["complex_karyotype"]
        favorable_final = cyto_string_series.str.contains(
            r"t\s*\(\s*8\s*;\s*21\s*\)|inv\s*\(\s*16\s*\)|t\s*\(\s*16\s*;\s*16\s*\)",
            case=False,
            na=False,
            regex=True,
        ).astype(int)

        label = pd.Series(1, index=df.index)
        label = label.mask(favorable_final == 1, 0)
        label = label.mask(adverse_final == 1, 2)

        result_df["eln_cyto_favorable"] = (label == 0).astype(int)
        result_df["eln_cyto_intermediate"] = (label == 1).astype(int)
        result_df["eln_cyto_adverse"] = (label == 2).astype(int)

        result_df.loc[
            result_df["cyto_is_missing"] == 1,
            [c for c in result_df.columns if c not in ["ID", "cyto_is_missing"]],
        ] = np.nan
        final_cols = sorted([c for c in result_df.columns if c != "ID"])
        return result_df[["ID"] + final_cols]
