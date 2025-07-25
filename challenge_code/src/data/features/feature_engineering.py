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
    FAVORABLE_GENES,
    ADVERSE_GENES,
    INTERMEDIATE_GENES,
    GENE_PATHWAYS,
)


def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable and prognostic clinical features.

    Based on established prognostic factors in AML:
    - Age (continuous + categorical)
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
    for col in numeric_columns:
        if col in clinical_df.columns:
            clinical_df[col] = pd.to_numeric(clinical_df[col], errors="coerce")

    # ===== CLINICALLY RELEVANT RATIOS =====

    # Neutrophil/WBC ratio (measure of granulocytic maturation)
    clinical_df["neutrophil_ratio"] = clinical_df["ANC"] / clinical_df["WBC"]
    clinical_df["neutrophil_ratio"] = clinical_df["neutrophil_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Monocyte/WBC ratio (elevation = unfavorable prognosis)
    clinical_df["monocyte_ratio"] = clinical_df["MONOCYTES"] / clinical_df["WBC"]
    clinical_df["monocyte_ratio"] = clinical_df["monocyte_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Platelet/WBC ratio (general hematopoiesis measure)
    clinical_df["platelet_wbc_ratio"] = clinical_df["PLT"] / clinical_df["WBC"]
    clinical_df["platelet_wbc_ratio"] = clinical_df["platelet_wbc_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    # ===== CLINICAL THRESHOLDS (binary) =====

    # Anemia (HB < 10 g/dL = moderate, < 8 g/dL = severe)
    clinical_df["anemia_moderate"] = (clinical_df["HB"] < 10).astype(int)
    clinical_df["anemia_severe"] = (clinical_df["HB"] < 8).astype(int)

    # Thrombocytopenia (PLT < 100 = moderate, < 50 = severe)
    clinical_df["thrombocytopenia_moderate"] = (clinical_df["PLT"] < 100).astype(int)
    clinical_df["thrombocytopenia_severe"] = (clinical_df["PLT"] < 50).astype(int)

    # Neutropenia (ANC < 1.5 = moderate, < 1.0 = severe)
    clinical_df["neutropenia_moderate"] = (clinical_df["ANC"] < 1.5).astype(int)
    clinical_df["neutropenia_severe"] = (clinical_df["ANC"] < 1.0).astype(int)

    # Leukocytosis (WBC > 30 = high)
    clinical_df["leukocytosis_high"] = (clinical_df["WBC"] > 30).astype(int)

    # High bone marrow blast count (>20% = AML)
    clinical_df["high_blast_count"] = (clinical_df["BM_BLAST"] > 20).astype(int)

    # ===== COMPOSITE CLINICAL SCORES =====

    # Cytopenia score (0-3, based on anemia + thrombocytopenia + neutropenia)
    clinical_df["cytopenia_score"] = (
        clinical_df["anemia_moderate"]
        + clinical_df["thrombocytopenia_moderate"]
        + clinical_df["neutropenia_moderate"]
    )

    # Pancytopenia (all lineages affected)
    clinical_df["pancytopenia"] = (clinical_df["cytopenia_score"] == 3).astype(int)

    # Proliferation score (blasts + leukocytosis)
    clinical_df["proliferation_score"] = (
        clinical_df["high_blast_count"] + clinical_df["leukocytosis_high"]
    )

    # ===== LOG TRANSFORMATIONS (for skewed distributions) =====
    for col in ["WBC", "PLT", "ANC", "MONOCYTES"]:
        if col in clinical_df.columns:
            clinical_df[f"log_{col}"] = np.log1p(clinical_df[col].fillna(0))

    # ===== ADDITIONAL RATIOS =====
    clinical_df["blast_platelet_ratio"] = clinical_df["BM_BLAST"] / clinical_df["PLT"]

    # Handle infinite values
    clinical_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return clinical_df


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
    result_df = pd.DataFrame(index=df.index)

    # Clean and prepare cytogenetics data
    cytogenetics_raw = df["CYTOGENETICS"].fillna("46,XX")  # Default normal karyotype
    cytogenetics_clean = cytogenetics_raw.str.strip()

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

    # Favorable abnormalities
    result_df["t_8_21"] = contains_pattern(CYTOGENETIC_FAVORABLE[0])
    result_df["inv_16"] = contains_pattern(CYTOGENETIC_FAVORABLE[1])
    result_df["t_16_16"] = contains_pattern(CYTOGENETIC_FAVORABLE[2])
    result_df["t_15_17"] = contains_pattern(CYTOGENETIC_FAVORABLE[3])
    result_df["any_favorable_cyto"] = result_df[
        ["t_8_21", "inv_16", "t_16_16", "t_15_17"]
    ].max(axis=1)

    # Intermediate abnormalities
    result_df["normal_karyotype"] = cytogenetics.str.match(
        CYTOGENETIC_INTERMEDIATE[0], na=False
    ).astype(int)
    result_df["trisomy_8"] = contains_pattern(CYTOGENETIC_INTERMEDIATE[1])

    # Adverse abnormalities
    result_df["del_5q"] = contains_pattern(CYTOGENETIC_ADVERSE[0])
    result_df["monosomy_7"] = contains_pattern(CYTOGENETIC_ADVERSE[1])
    result_df["del_17p"] = contains_pattern(CYTOGENETIC_ADVERSE[2])

    adverse_abnormalities = ["del_5q", "monosomy_7", "del_17p"]
    result_df["any_adverse_cyto"] = result_df[adverse_abnormalities].max(axis=1)

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
    favorable_conditions = []
    if "mut_NPM1" in molecular_df.columns:
        npm1_favorable = (molecular_df["mut_NPM1"] == 1) & (
            molecular_df.get("mut_FLT3", 0) == 0
        )
        favorable_conditions.append(npm1_favorable)

    if "CEBPA_biallelic" in molecular_df.columns:
        favorable_conditions.append(molecular_df["CEBPA_biallelic"] == 1)

    if favorable_conditions:
        favorable_mask = pd.concat(favorable_conditions, axis=1).any(axis=1)
        molecular_df.loc[favorable_mask, "eln_molecular_risk"] = 0

    # Adverse molecular features (2)
    adverse_conditions = []
    adverse_genes = ["TP53", "ASXL1", "RUNX1", "BCOR", "EZH2"]

    for gene in adverse_genes:
        if f"mut_{gene}" in molecular_df.columns:
            adverse_conditions.append(molecular_df[f"mut_{gene}"] == 1)

    if "FLT3_high_VAF" in molecular_df.columns:
        adverse_conditions.append(molecular_df["FLT3_high_VAF"] == 1)

    if adverse_conditions:
        adverse_mask = pd.concat(adverse_conditions, axis=1).any(axis=1)
        molecular_df.loc[adverse_mask, "eln_molecular_risk"] = 2

    return molecular_df


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
    if (
        "eln_molecular_risk" in final_df.columns
        and "normal_karyotype" in final_df.columns
    ):
        normal_cyto_mask = final_df["normal_karyotype"] == 1
        final_df.loc[normal_cyto_mask, "eln_integrated_risk"] = final_df.loc[
            normal_cyto_mask, "eln_molecular_risk"
        ]

    # Clinical composite scores
    final_df["clinical_risk_score"] = final_df.get("cytopenia_score", 0) + final_df.get(
        "proliferation_score", 0
    )

    # Comprehensive risk score (weighted combination)
    risk_components = []
    if "eln_integrated_risk" in final_df.columns:
        risk_components.append(final_df["eln_integrated_risk"] * 0.5)
    if "clinical_risk_score" in final_df.columns:
        clinical_normalized = (
            final_df["clinical_risk_score"] / final_df["clinical_risk_score"].max()
        )
        risk_components.append(clinical_normalized * 0.3)
    if "total_mutations" in final_df.columns:
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
