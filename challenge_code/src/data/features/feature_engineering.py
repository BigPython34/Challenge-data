"""
Feature engineering module for AML survival analysis.

This module handles the creation of clinically relevant features for predicting
overall survival in AML patients based on ELN 2022 guidelines.

Functions:
- Clinical feature engineering (blood counts, ratios, thresholds)
- Cytogenetic risk classification (ELN 2022)
- Molecular risk assessment (mutations, pathways)
- Feature combination and integration
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
from ...config import (
    CYTOGENETIC_FAVORABLE,
    CYTOGENETIC_ADVERSE,
    CYTOGENETIC_INTERMEDIATE,
    ALL_IMPORTANT_GENES,
    GENE_PATHWAYS,
)


def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable and prognostic clinical features.

    Based on established prognostic factors in AML:
    - Blood counts and their ratios
    - Bone marrow blast percentage
    - Cytopenias (anemia, thrombocytopenia, neutropenia)

    Parameters
    ----------
    df : pd.DataFrame
        Clinical data

    Returns
    -------
    pd.DataFrame
        Clinical features with medical relevance
    """
    clinical_df = df.copy()

    # ===== BASE FEATURES (cleaning) =====
    numeric_columns = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    clinical_df = _ensure_numeric_columns(clinical_df, numeric_columns)

    # ===== CLINICALLY RELEVANT RATIOS =====
    clinical_df = _create_clinical_ratios(clinical_df)

    # ===== CLINICAL THRESHOLDS (binary) =====
    clinical_df = _create_clinical_thresholds(clinical_df)

    # ===== COMPOSITE CLINICAL SCORES =====
    clinical_df = _create_composite_scores(clinical_df)

    # ===== LOG TRANSFORMATIONS (for skewed distributions) =====
    clinical_df = _create_log_transformations(clinical_df)

    # Handle infinite values
    clinical_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return clinical_df


def _ensure_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Ensure specified columns are numeric."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _create_clinical_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically relevant ratios."""
    ratios = {
        "neutrophil_ratio": ("ANC", "WBC"),  # Granulocytic maturation
        "monocyte_ratio": ("MONOCYTES", "WBC"),  # Unfavorable if elevated
        "platelet_wbc_ratio": ("PLT", "WBC"),  # General hematopoiesis
        "blast_platelet_ratio": ("BM_BLAST", "PLT"),  # Additional ratio
    }

    for ratio_name, (numerator, denominator) in ratios.items():
        if numerator in df.columns and denominator in df.columns:
            df[ratio_name] = df[numerator] / df[denominator]
            df[ratio_name] = df[ratio_name].replace([np.inf, -np.inf], np.nan)

    return df


def _create_clinical_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary features based on clinical thresholds."""
    thresholds = {
        # Anemia thresholds
        "anemia_moderate": ("HB", "<", 10),
        "anemia_severe": ("HB", "<", 8),
        # Thrombocytopenia thresholds
        "thrombocytopenia_moderate": ("PLT", "<", 100),
        "thrombocytopenia_severe": ("PLT", "<", 50),
        # Neutropenia thresholds
        "neutropenia_moderate": ("ANC", "<", 1.5),
        "neutropenia_severe": ("ANC", "<", 1.0),
        # Other thresholds
        "leukocytosis_high": ("WBC", ">", 30),
        "high_blast_count": ("BM_BLAST", ">", 20),
    }

    for feature_name, (column, operator, threshold) in thresholds.items():
        if column in df.columns:
            if operator == "<":
                df[feature_name] = (df[column] < threshold).astype(int)
            elif operator == ">":
                df[feature_name] = (df[column] > threshold).astype(int)

    return df


def _create_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create composite clinical scores."""
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


def _create_log_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Create log transformations for skewed distributions."""
    log_columns = ["WBC", "PLT", "ANC", "MONOCYTES"]
    for col in log_columns:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0))
    return df


def extract_cytogenetic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cytogenetic features based on ELN 2022 classification.

    This function extracts clinically relevant cytogenetic features including:
    - Basic chromosome characteristics
    - ELN 2022 risk-specific abnormalities
    - Cytogenetic complexity metrics
    - Final ELN 2022 cytogenetic risk score

    Parameters
    ----------
    df : pd.DataFrame
        Must contain "CYTOGENETICS" column with karyotype information

    Returns
    -------
    pd.DataFrame
        Features indexed by patient ID with cytogenetic characteristics
    """
    if "CYTOGENETICS" not in df.columns:
        return df

    result_df = pd.DataFrame(index=df.index)

    # Clean and prepare cytogenetics data
    cytogenetics_clean = df["CYTOGENETICS"].fillna("46,XX").str.strip()

    # === BASIC CHROMOSOME CHARACTERISTICS ===
    result_df = _extract_basic_chromosome_features(result_df, cytogenetics_clean)

    # === ELN 2022 CYTOGENETIC ABNORMALITIES ===
    result_df = _extract_eln2022_cytogenetic_abnormalities(
        result_df, cytogenetics_clean
    )

    # === CYTOGENETIC COMPLEXITY METRICS ===
    result_df = _extract_cytogenetic_complexity(result_df, cytogenetics_clean)

    # === ELN 2022 FINAL RISK CLASSIFICATION ===
    result_df = _calculate_eln2022_cytogenetic_risk(result_df)

    return result_df


def extract_molecular_risk_features(
    df: pd.DataFrame, maf_df: pd.DataFrame, important_genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create molecular features based on ELN 2022 prognostic mutations.

    This function creates comprehensive molecular features including:
    - Binary mutation status for important genes
    - VAF-based features for key prognostic genes
    - Mutation type classification
    - Clinically relevant co-mutation patterns
    - Pathway-level alterations
    - ELN 2022 molecular risk classification

    Parameters
    ----------
    df : pd.DataFrame
        Clinical data with patient IDs
    maf_df : pd.DataFrame
        Mutation data with columns: ID, GENE, VAF, EFFECT
    important_genes : List[str], optional
        Genes to analyze. If None, uses ALL_IMPORTANT_GENES from config

    Returns
    -------
    pd.DataFrame
        Molecular features indexed by patient ID
    """
    if important_genes is None:
        important_genes = ALL_IMPORTANT_GENES

    # Initialize molecular features dataframe
    molecular_df = pd.DataFrame(index=df["ID"].unique())

    # === BINARY MUTATION STATUS ===
    molecular_df = _extract_binary_mutations(molecular_df, maf_df, important_genes)

    # === VAF-BASED FEATURES ===
    molecular_df = _extract_vaf_features(molecular_df, maf_df)

    # === MUTATION TYPE CLASSIFICATION ===
    molecular_df = _extract_mutation_types(molecular_df, maf_df)

    # === CLINICALLY RELEVANT CO-MUTATIONS ===
    molecular_df = _extract_comutation_patterns(molecular_df)

    # === PATHWAY-LEVEL ALTERATIONS ===
    molecular_df = _extract_pathway_alterations(molecular_df)

    # === ELN 2022 MOLECULAR RISK CLASSIFICATION ===
    molecular_df = _calculate_eln2022_molecular_risk(molecular_df)

    return molecular_df.reset_index().rename(columns={"index": "ID"})


def create_molecular_burden_features(maf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create global mutation burden statistics.

    Parameters
    ----------
    maf_df : pd.DataFrame
        Mutation data with columns ID, GENE, VAF, EFFECT

    Returns
    -------
    pd.DataFrame
        Mutation burden features per patient
    """
    # Total mutations per patient
    mutation_counts = maf_df.groupby("ID").size().reset_index(name="total_mutations")

    # VAF statistics
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

    # Fill NaN in std with 0 (single variant)
    vaf_stats["vaf_std"] = vaf_stats["vaf_std"].fillna(0)

    # High VAF mutations (>0.4, possible germline)
    high_vaf_counts = (
        maf_df[maf_df["VAF"] > 0.4]
        .groupby("ID")
        .size()
        .reset_index(name="high_vaf_mutations")
    )

    # Combine all statistics
    burden_df = mutation_counts.merge(vaf_stats, on="ID", how="left")
    burden_df = burden_df.merge(high_vaf_counts, on="ID", how="left")
    burden_df["high_vaf_mutations"] = (
        burden_df["high_vaf_mutations"].fillna(0).astype(int)
    )

    # High VAF mutation ratio
    burden_df["high_vaf_ratio"] = (
        burden_df["high_vaf_mutations"] / burden_df["total_mutations"]
    )
    burden_df["high_vaf_ratio"] = burden_df["high_vaf_ratio"].fillna(0)

    return burden_df


def combine_all_features(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    burden_df: pd.DataFrame,
    cyto_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all features and create integrated risk scores.

    This function merges all feature categories and creates integrated scores
    while preserving individual features to avoid information loss.

    Parameters
    ----------
    clinical_df, molecular_df, burden_df, cyto_df : pd.DataFrame
        Feature dataframes from different categories

    Returns
    -------
    pd.DataFrame
        Complete dataset with all features and integrated scores
    """
    # Start with clinical data as base
    final_df = clinical_df.copy()
    final_df["ID"] = final_df["ID"].astype(str)

    # === MERGE ALL FEATURE CATEGORIES ===
    final_df = _merge_feature_dataframes(final_df, molecular_df, burden_df, cyto_df)

    # === FILL MISSING VALUES STRATEGICALLY ===
    final_df = _fill_missing_values_strategically(final_df)

    # === CREATE INTEGRATED SCORES ===
    final_df = _create_integrated_risk_scores(final_df)

    # === ADD INTERACTION FEATURES ===
    final_df = _create_interaction_features(final_df)

    return final_df


def get_clean_feature_lists() -> Dict[str, List[str]]:
    """
    Return organized feature lists by category based on ELN 2022 classification.

    Returns
    -------
    Dict[str, List[str]]
        Features organized by clinical category and data type
    """
    # Define all feature categories
    clinical_base = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "AGE"]
    clinical_ratios = [
        "neutrophil_ratio",
        "monocyte_ratio",
        "platelet_wbc_ratio",
        "blast_platelet_ratio",
    ]
    clinical_binary = [
        "anemia_severe",
        "thrombocytopenia_severe",
        "neutropenia_severe",
        "leukocytosis_high",
        "high_blast_count",
        "pancytopenia",
    ]
    clinical_scores = ["cytopenia_score", "proliferation_score", "clinical_risk_score"]

    # Cytogenetic features
    cytogenetic = ["normal_karyotype", "complex_karyotype", "eln_cyto_risk"]

    # Molecular features
    molecular_mutations = [f"mut_{gene}" for gene in ALL_IMPORTANT_GENES]
    molecular_derived = ["eln_molecular_risk", "TP53_truncating", "CEBPA_biallelic"]

    # Mutation burden
    mutation_burden = [
        "total_mutations",
        "vaf_mean",
        "vaf_median",
        "vaf_max",
        "high_vaf_ratio",
    ]

    # Integrated scores
    integrated_scores = ["eln_integrated_risk", "comprehensive_risk_score"]

    return {
        "clinical_base": clinical_base,
        "clinical_ratios": clinical_ratios,
        "clinical_binary": clinical_binary,
        "clinical_scores": clinical_scores,
        "cytogenetic": cytogenetic,
        "molecular_mutations": molecular_mutations,
        "molecular_derived": molecular_derived,
        "mutation_burden": mutation_burden,
        "integrated_scores": integrated_scores,
    }


# ===== PRIVATE HELPER FUNCTIONS =====


def _extract_basic_chromosome_features(
    result_df: pd.DataFrame, cytogenetics: pd.Series
) -> pd.DataFrame:
    """Extract basic chromosome count and sex chromosome information."""
    # Chromosome count with validation
    chromosome_count = cytogenetics.str.extract(r"(\d+)")[0]
    chromosome_count = pd.to_numeric(chromosome_count, errors="coerce")
    chromosome_count = chromosome_count.where(
        (chromosome_count >= 30) & (chromosome_count <= 80), 46
    )
    result_df["chromosome_count"] = chromosome_count.fillna(46).astype(int)

    # Sex chromosome determination
    result_df["sex_chromosomes"] = 0.5  # Default for unknown
    xx_mask = cytogenetics.str.contains(r"\bXX\b", case=False, na=False)
    xy_mask = cytogenetics.str.contains(r"\bXY\b", case=False, na=False)
    result_df.loc[xx_mask, "sex_chromosomes"] = 0
    result_df.loc[xy_mask, "sex_chromosomes"] = 1

    return result_df


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

    result_df["any_favorable_cyto"] = result_df[list(favorable_features.keys())].max(
        axis=1
    )

    # === INTERMEDIATE ABNORMALITIES ===
    result_df["normal_karyotype"] = cytogenetics.str.match(
        CYTOGENETIC_INTERMEDIATE[0], na=False
    ).astype(int)
    result_df["trisomy_8"] = contains_pattern(CYTOGENETIC_INTERMEDIATE[1])

    # === ADVERSE ABNORMALITIES ===
    adverse_features = {
        "del_5q": CYTOGENETIC_ADVERSE[0],
        "monosomy_7": CYTOGENETIC_ADVERSE[1],
        "del_17p": CYTOGENETIC_ADVERSE[2],
    }

    for feature_name, pattern in adverse_features.items():
        result_df[feature_name] = contains_pattern(pattern)

    result_df["any_adverse_cyto"] = result_df[list(adverse_features.keys())].max(axis=1)

    return result_df


def _extract_cytogenetic_complexity(
    result_df: pd.DataFrame, cytogenetics: pd.Series
) -> pd.DataFrame:
    """Extract cytogenetic complexity metrics."""
    # Count total abnormalities
    result_df["num_cyto_abnormalities"] = (
        cytogenetics.str.count(",").fillna(0).astype(int)
    )

    # Complex karyotype: ≥3 unrelated chromosome abnormalities
    result_df["complex_karyotype"] = (result_df["num_cyto_abnormalities"] >= 3).astype(
        int
    )

    return result_df


def _calculate_eln2022_cytogenetic_risk(result_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final ELN 2022 cytogenetic risk score."""
    # Initialize with intermediate risk (1)
    result_df["eln_cyto_risk"] = 1

    # Favorable risk (0)
    favorable_mask = result_df["any_favorable_cyto"] == 1
    result_df.loc[favorable_mask, "eln_cyto_risk"] = 0

    # Adverse risk (2)
    adverse_mask = (result_df["any_adverse_cyto"] == 1) | (
        result_df["complex_karyotype"] == 1
    )
    result_df.loc[adverse_mask, "eln_cyto_risk"] = 2

    return result_df


def _extract_binary_mutations(
    molecular_df: pd.DataFrame, maf_df: pd.DataFrame, important_genes: List[str]
) -> pd.DataFrame:
    """Extract binary mutation status for all important genes."""
    for gene in important_genes:
        column_name = f"mut_{gene}"
        if gene in maf_df["GENE"].values:
            mutated_patients = maf_df[maf_df["GENE"] == gene]["ID"].unique()
            molecular_df[column_name] = molecular_df.index.isin(
                mutated_patients
            ).astype(int)
        else:
            molecular_df[column_name] = 0
    return molecular_df


def _extract_vaf_features(
    molecular_df: pd.DataFrame, maf_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract VAF-based features for prognostically important genes."""
    vaf_important_genes = ["TP53", "FLT3", "NPM1", "CEBPA", "DNMT3A"]

    for gene in vaf_important_genes:
        if gene in maf_df["GENE"].values:
            gene_vaf = maf_df[maf_df["GENE"] == gene].groupby("ID")["VAF"].max()
            molecular_df[f"vaf_max_{gene}"] = molecular_df.index.map(gene_vaf).fillna(0)
            molecular_df[f"{gene}_high_VAF"] = (
                molecular_df[f"vaf_max_{gene}"] > 0.5
            ).astype(int)
        else:
            molecular_df[f"vaf_max_{gene}"] = 0.0
            molecular_df[f"{gene}_high_VAF"] = 0
    return molecular_df


def _extract_mutation_types(
    molecular_df: pd.DataFrame, maf_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract mutation type classifications for key genes."""
    # TP53 truncating mutations
    if "TP53" in maf_df["GENE"].values:
        tp53_patients = maf_df[maf_df["GENE"] == "TP53"]
        truncating_effects = ["nonsense", "frameshift", "splice_site", "stop_gained"]
        tp53_truncating = tp53_patients[
            tp53_patients["EFFECT"].str.contains(
                "|".join(truncating_effects), case=False, na=False
            )
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

    return molecular_df


def _extract_comutation_patterns(molecular_df: pd.DataFrame) -> pd.DataFrame:
    """Extract clinically relevant co-mutation patterns."""
    # NPM1+/FLT3- : Favorable prognosis
    molecular_df["NPM1_pos_FLT3_neg"] = (
        (molecular_df.get("mut_NPM1", 0) == 1) & (molecular_df.get("mut_FLT3", 0) == 0)
    ).astype(int)

    # Triple mutation: DNMT3A + NPM1 + FLT3
    molecular_df["DNMT3A_NPM1_FLT3"] = (
        (molecular_df.get("mut_DNMT3A", 0) == 1)
        & (molecular_df.get("mut_NPM1", 0) == 1)
        & (molecular_df.get("mut_FLT3", 0) == 1)
    ).astype(int)

    return molecular_df


def _extract_pathway_alterations(molecular_df: pd.DataFrame) -> pd.DataFrame:
    """Extract pathway-level alteration features."""
    for pathway_name, pathway_genes in GENE_PATHWAYS.items():
        pathway_columns = [
            f"mut_{gene}"
            for gene in pathway_genes
            if f"mut_{gene}" in molecular_df.columns
        ]

        if pathway_columns:
            molecular_df[f"{pathway_name}_altered"] = (
                molecular_df[pathway_columns].sum(axis=1) > 0
            ).astype(int)
            molecular_df[f"{pathway_name}_count"] = molecular_df[pathway_columns].sum(
                axis=1
            )
        else:
            molecular_df[f"{pathway_name}_altered"] = 0
            molecular_df[f"{pathway_name}_count"] = 0

    return molecular_df


def _calculate_eln2022_molecular_risk(molecular_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ELN 2022 molecular risk classification."""
    # Initialize with intermediate risk (1)
    molecular_df["eln_molecular_risk"] = 1

    # Favorable molecular features (0)
    favorable_mask = _get_favorable_molecular_mask(molecular_df)
    if favorable_mask is not None:
        molecular_df.loc[favorable_mask, "eln_molecular_risk"] = 0

    # Adverse molecular features (2)
    adverse_mask = _get_adverse_molecular_mask(molecular_df)
    if adverse_mask is not None:
        molecular_df.loc[adverse_mask, "eln_molecular_risk"] = 2

    return molecular_df


def _get_favorable_molecular_mask(molecular_df: pd.DataFrame) -> pd.Series:
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


def _get_adverse_molecular_mask(molecular_df: pd.DataFrame) -> pd.Series:
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


def _merge_feature_dataframes(
    final_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    burden_df: pd.DataFrame,
    cyto_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all feature dataframes with proper handling of missing data."""
    # Merge molecular features
    if not molecular_df.empty:
        molecular_df["ID"] = molecular_df["ID"].astype(str)
        final_df = final_df.merge(molecular_df, on="ID", how="left")

    # Merge mutation burden features
    if not burden_df.empty:
        burden_df["ID"] = burden_df["ID"].astype(str)
        final_df = final_df.merge(burden_df, on="ID", how="left")

    # Merge cytogenetic features
    if not cyto_df.empty:
        cyto_df_reset = cyto_df.reset_index()
        if "index" in cyto_df_reset.columns:
            cyto_df_reset = cyto_df_reset.rename(columns={"index": "ID"})
        cyto_df_reset["ID"] = cyto_df_reset["ID"].astype(str)
        final_df = final_df.merge(cyto_df_reset, on="ID", how="left")

    return final_df


def _fill_missing_values_strategically(final_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values based on clinical knowledge."""
    # Missing mutations = no mutation detected
    mutation_cols = [col for col in final_df.columns if col.startswith("mut_")]
    for col in mutation_cols:
        final_df[col] = final_df[col].fillna(0).astype(int)

    # Missing mutation burden = no mutations
    burden_cols = [
        "total_mutations",
        "high_vaf_mutations",
        "vaf_mean",
        "vaf_median",
        "vaf_max",
        "high_vaf_ratio",
    ]
    for col in burden_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    # Missing cytogenetic abnormalities = normal/not detected
    cyto_cols = [
        col
        for col in final_df.columns
        if any(
            col.startswith(prefix)
            for prefix in ["t_", "inv_", "del_", "complex_", "normal_", "any_"]
        )
    ]
    for col in cyto_cols:
        final_df[col] = final_df[col].fillna(0).astype(int)

    # Missing cytogenetic risk = intermediate (conservative)
    if "eln_cyto_risk" in final_df.columns:
        final_df["eln_cyto_risk"] = final_df["eln_cyto_risk"].fillna(1)

    return final_df


def _create_integrated_risk_scores(final_df: pd.DataFrame) -> pd.DataFrame:
    """Create integrated risk scores while preserving individual components."""
    # ELN 2022 integrated risk
    final_df["eln_integrated_risk"] = final_df.get("eln_cyto_risk", 1)

    # For normal cytogenetics, use molecular risk
    final_df = _apply_molecular_risk_for_normal_cytogenetics(final_df)

    # Clinical composite scores
    final_df["clinical_risk_score"] = final_df.get("cytopenia_score", 0) + final_df.get(
        "proliferation_score", 0
    )

    # Comprehensive risk score (weighted combination)
    final_df = _calculate_comprehensive_risk_score(final_df)

    return final_df


def _apply_molecular_risk_for_normal_cytogenetics(
    final_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply molecular risk for patients with normal cytogenetics."""
    if (
        "eln_molecular_risk" in final_df.columns
        and "normal_karyotype" in final_df.columns
    ):
        normal_cyto_mask = final_df["normal_karyotype"] == 1
        final_df.loc[normal_cyto_mask, "eln_integrated_risk"] = final_df.loc[
            normal_cyto_mask, "eln_molecular_risk"
        ]
    return final_df


def _calculate_comprehensive_risk_score(final_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive risk score from multiple components."""
    risk_components = []

    if "eln_integrated_risk" in final_df.columns:
        risk_components.append(final_df["eln_integrated_risk"] * 0.5)

    if (
        "clinical_risk_score" in final_df.columns
        and final_df["clinical_risk_score"].max() > 0
    ):
        clinical_normalized = (
            final_df["clinical_risk_score"] / final_df["clinical_risk_score"].max()
        )
        risk_components.append(clinical_normalized * 0.3)

    if "total_mutations" in final_df.columns and final_df["total_mutations"].max() > 0:
        mutation_normalized = (
            final_df["total_mutations"] / final_df["total_mutations"].max()
        )
        risk_components.append(mutation_normalized * 0.2)

    if risk_components:
        final_df["comprehensive_risk_score"] = sum(risk_components)

    return final_df


def _create_interaction_features(final_df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically relevant interaction features."""
    # TP53 + complex cytogenetics (very high risk)
    if "mut_TP53" in final_df.columns and "complex_karyotype" in final_df.columns:
        final_df["TP53_complex_cyto"] = (
            (final_df["mut_TP53"] == 1) & (final_df["complex_karyotype"] == 1)
        ).astype(int)

    # FLT3/NPM1 interactions
    if "mut_FLT3" in final_df.columns and "mut_NPM1" in final_df.columns:
        final_df["FLT3_NPM1_interaction"] = 0  # Default: neither

        # NPM1+/FLT3- = favorable (1)
        npm1_only = (final_df["mut_NPM1"] == 1) & (final_df["mut_FLT3"] == 0)
        final_df.loc[npm1_only, "FLT3_NPM1_interaction"] = 1

        # NPM1+/FLT3+ = intermediate (2)
        both_mut = (final_df["mut_NPM1"] == 1) & (final_df["mut_FLT3"] == 1)
        final_df.loc[both_mut, "FLT3_NPM1_interaction"] = 2

    return final_df
