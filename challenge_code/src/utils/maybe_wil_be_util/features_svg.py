"""
Feature engineering for acute myeloid leukemia (AML) survival modeling

This module handles the creation of clinically relevant features for predicting
overall survival in AML patients.

Guiding principles:
1. Features based on established medical knowledge (ELN 2022, WHO)
2. Simplicity and interpretability for clinicians
3. Robustness to missing data
4. Relevance for survival modeling
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
from ..config import (
    CYTOGENETIC_FAVORABLE,
    CYTOGENETIC_ADVERSE,
    CYTOGENETIC_INTERMEDIATE,
    ALL_IMPORTANT_GENES,
    GENE_PATHWAYS,
)


def extract_cytogenetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cytogenetic features based on ELN 2022 classification.

    This function extracts clinically relevant cytogenetic features including:
    - Basic chromosome characteristics (count, sex chromosomes)
    - ELN 2022 risk-specific abnormalities (favorable, intermediate, adverse)
    - Cytogenetic complexity metrics
    - Final ELN 2022 cytogenetic risk score

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the "CYTOGENETICS" column with karyotype information.

    Returns
    -------
    pd.DataFrame
        Features indexed by patient ID with all cytogenetic characteristics.
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


def _extract_basic_chromosome_features(
    result_df: pd.DataFrame, cytogenetics: pd.Series
) -> pd.DataFrame:
    """Extract basic chromosome count and sex chromosome information."""

    # Chromosome count with validation
    chromosome_count = cytogenetics.str.extract(r"(\d+)")[0]
    chromosome_count = pd.to_numeric(chromosome_count, errors="coerce")

    # Validate chromosome count (normal range 30-80, default to 46)
    chromosome_count = chromosome_count.where(
        (chromosome_count >= 30) & (chromosome_count <= 80), 46
    )
    result_df["chromosome_count"] = chromosome_count.fillna(46).astype(int)

    # Sex chromosome determination: XX=0, XY=1, unknown/ambiguous=0.5
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
        """Helper function to check if cytogenetics contains a specific pattern."""
        return cytogenetics.str.contains(
            pattern, case=False, regex=True, na=False
        ).astype(int)

    # === FAVORABLE ABNORMALITIES (ELN 2022) ===
    result_df["t_8_21"] = contains_pattern(CYTOGENETIC_FAVORABLE[0])  # t(8;21)
    result_df["inv_16"] = contains_pattern(CYTOGENETIC_FAVORABLE[1])  # inv(16)
    result_df["t_16_16"] = contains_pattern(CYTOGENETIC_FAVORABLE[2])  # t(16;16)
    result_df["t_15_17"] = contains_pattern(CYTOGENETIC_FAVORABLE[3])  # t(15;17)

    # Combined favorable cytogenetics
    result_df["any_favorable_cyto"] = result_df[
        ["t_8_21", "inv_16", "t_16_16", "t_15_17"]
    ].max(axis=1)

    # === INTERMEDIATE ABNORMALITIES (ELN 2022) ===
    result_df["normal_karyotype"] = cytogenetics.str.match(
        CYTOGENETIC_INTERMEDIATE[0], na=False
    ).astype(int)
    result_df["trisomy_8"] = contains_pattern(CYTOGENETIC_INTERMEDIATE[1])  # +8
    result_df["t_9_11"] = contains_pattern(CYTOGENETIC_INTERMEDIATE[2])  # t(9;11)
    result_df["kmt2a_rearranged"] = contains_pattern(
        CYTOGENETIC_INTERMEDIATE[3]
    )  # 11q23

    # === ADVERSE ABNORMALITIES (ELN 2022) ===
    result_df["del_5q"] = contains_pattern(CYTOGENETIC_ADVERSE[0])  # -5/del(5q)
    result_df["monosomy_7"] = contains_pattern(CYTOGENETIC_ADVERSE[1])  # -7/del(7q)
    result_df["del_17p"] = contains_pattern(CYTOGENETIC_ADVERSE[2])  # del(17p)
    result_df["abn_3q"] = contains_pattern(CYTOGENETIC_ADVERSE[3])  # inv(3)/t(3;3)
    result_df["t_6_9"] = contains_pattern(CYTOGENETIC_ADVERSE[4])  # t(6;9)
    result_df["t_9_22"] = contains_pattern(CYTOGENETIC_ADVERSE[5])  # t(9;22) BCR-ABL1

    # Combined adverse cytogenetics (excluding complex karyotype)
    adverse_abnormalities = [
        "del_5q",
        "monosomy_7",
        "del_17p",
        "abn_3q",
        "t_6_9",
        "t_9_22",
    ]
    result_df["any_adverse_cyto"] = result_df[adverse_abnormalities].max(axis=1)

    return result_df


def _extract_cytogenetic_complexity(
    result_df: pd.DataFrame, cytogenetics: pd.Series
) -> pd.DataFrame:
    """Extract cytogenetic complexity metrics."""

    # Count total abnormalities (comma-separated elements minus 1 for base karyotype)
    result_df["num_cyto_abnormalities"] = (
        cytogenetics.str.count(",").fillna(0).astype(int)
    )

    # Complex karyotype: ≥3 unrelated chromosome abnormalities (ELN 2022 definition)
    result_df["complex_karyotype"] = (result_df["num_cyto_abnormalities"] >= 3).astype(
        int
    )

    # Monosomal karyotype: ≥2 autosomal monosomies OR 1 autosomal monosomy + structural abnormalities
    monosomy_patterns = [r"-\d+\b", r"monosomy"]  # Detect autosomal monosomies
    result_df["monosomal_karyotype"] = 0
    for pattern in monosomy_patterns:
        monosomy_count = cytogenetics.str.count(pattern, flags=re.IGNORECASE)
        result_df["monosomal_karyotype"] = np.maximum(
            result_df["monosomal_karyotype"], (monosomy_count >= 2).astype(int)
        )

    return result_df


def _calculate_eln2022_cytogenetic_risk(result_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final ELN 2022 cytogenetic risk score."""

    # Initialize with intermediate risk (1)
    result_df["eln_cyto_risk"] = 1

    # Favorable risk (0): presence of any favorable abnormality
    favorable_mask = result_df["any_favorable_cyto"] == 1
    result_df.loc[favorable_mask, "eln_cyto_risk"] = 0

    # Adverse risk (2): presence of any adverse abnormality OR complex karyotype
    adverse_mask = (result_df["any_adverse_cyto"] == 1) | (
        result_df["complex_karyotype"] == 1
    )
    result_df.loc[adverse_mask, "eln_cyto_risk"] = 2

    return result_df


def extract_molecular_risk_features(
    df: pd.DataFrame, maf_df: pd.DataFrame, important_genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create molecular features based on ELN 2022 prognostic mutations.

    This function creates comprehensive molecular features including:
    - Binary mutation status for all important genes
    - VAF-based features for key prognostic genes
    - Mutation type classification (e.g., truncating vs missense)
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

    # Genes where VAF has prognostic significance
    vaf_important_genes = ["TP53", "FLT3", "NPM1", "CEBPA", "DNMT3A"]

    for gene in vaf_important_genes:
        if gene in maf_df["GENE"].values:
            gene_vaf = maf_df[maf_df["GENE"] == gene].groupby("ID")["VAF"].max()
            molecular_df[f"vaf_max_{gene}"] = molecular_df.index.map(gene_vaf).fillna(0)

            # High VAF threshold (>0.5 suggests clonal mutation or potential germline)
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

    # TP53: Distinguish truncating (loss-of-function) from missense mutations
    if "TP53" in maf_df["GENE"].values:
        tp53_patients = maf_df[maf_df["GENE"] == "TP53"]

        # Truncating mutations (complete loss of function)
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

    # CEBPA: Distinguish biallelic from monoallelic mutations (ELN 2022)
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

    # NPM1+/FLT3- : Favorable prognosis in CN-AML (ELN 2022)
    molecular_df["NPM1_pos_FLT3_neg"] = (
        (molecular_df.get("mut_NPM1", 0) == 1) & (molecular_df.get("mut_FLT3", 0) == 0)
    ).astype(int)

    # NPM1+/FLT3+ with low FLT3-ITD allelic ratio: Intermediate risk
    molecular_df["NPM1_pos_FLT3_low"] = (
        (molecular_df.get("mut_NPM1", 0) == 1)
        & (molecular_df.get("mut_FLT3", 0) == 1)
        & (molecular_df.get("FLT3_high_VAF", 0) == 0)
    ).astype(int)

    # Triple mutation: DNMT3A + NPM1 + FLT3 (common pattern)
    molecular_df["DNMT3A_NPM1_FLT3"] = (
        (molecular_df.get("mut_DNMT3A", 0) == 1)
        & (molecular_df.get("mut_NPM1", 0) == 1)
        & (molecular_df.get("mut_FLT3", 0) == 1)
    ).astype(int)

    # TP53 multi-hit: Multiple TP53 mutations (very adverse)
    # This will be enhanced when we have copy number data

    return molecular_df


def _extract_pathway_alterations(molecular_df: pd.DataFrame) -> pd.DataFrame:
    """Extract pathway-level alteration features."""

    for pathway_name, pathway_genes in GENE_PATHWAYS.items():
        # Check if any gene in the pathway is mutated
        pathway_columns = [
            f"mut_{gene}"
            for gene in pathway_genes
            if f"mut_{gene}" in molecular_df.columns
        ]

        if pathway_columns:
            molecular_df[f"{pathway_name}_altered"] = (
                molecular_df[pathway_columns].sum(axis=1) > 0
            ).astype(int)

            # Count number of genes altered in pathway
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

    # === FAVORABLE MOLECULAR FEATURES (0) ===
    favorable_conditions = []

    # NPM1 mutation without adverse-risk genetic lesions
    if "mut_NPM1" in molecular_df.columns:
        npm1_favorable = (molecular_df["mut_NPM1"] == 1) & (
            molecular_df.get("mut_FLT3", 0) == 0
        )  # No FLT3-ITD
        favorable_conditions.append(npm1_favorable)

    # Biallelic CEBPA mutations
    if "CEBPA_biallelic" in molecular_df.columns:
        favorable_conditions.append(molecular_df["CEBPA_biallelic"] == 1)

    # Apply favorable classification
    if favorable_conditions:
        favorable_mask = pd.concat(favorable_conditions, axis=1).any(axis=1)
        molecular_df.loc[favorable_mask, "eln_molecular_risk"] = 0

    # === ADVERSE MOLECULAR FEATURES (2) ===
    adverse_conditions = []

    # TP53 mutations (especially truncating)
    if "TP53_truncating" in molecular_df.columns:
        adverse_conditions.append(molecular_df["TP53_truncating"] == 1)
    elif "mut_TP53" in molecular_df.columns:
        adverse_conditions.append(molecular_df["mut_TP53"] == 1)

    # Other adverse genes from ELN 2022
    adverse_genes_list = [
        "ASXL1",
        "RUNX1",
        "BCOR",
        "EZH2",
        "SF3B1",
        "SRSF2",
        "STAG2",
        "U2AF1",
        "ZRSR2",
    ]
    for gene in adverse_genes_list:
        if f"mut_{gene}" in molecular_df.columns:
            adverse_conditions.append(molecular_df[f"mut_{gene}"] == 1)

    # FLT3-ITD with high allelic ratio
    if "FLT3_high_VAF" in molecular_df.columns:
        adverse_conditions.append(molecular_df["FLT3_high_VAF"] == 1)

    # Apply adverse classification
    if adverse_conditions:
        adverse_mask = pd.concat(adverse_conditions, axis=1).any(axis=1)
        molecular_df.loc[adverse_mask, "eln_molecular_risk"] = 2

    return molecular_df


def create_molecular_burden_features(maf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Créer des statistiques globales sur la charge mutationnelle

    Parameters:
    -----------
    maf_df : pd.DataFrame avec colonnes ID, GENE, VAF, EFFECT

    Returns:
    --------
    pd.DataFrame : Features de charge mutationnelle par patient
    """
    # Nombre total de mutations par patient
    mutation_counts = maf_df.groupby("ID").size().reset_index(name="total_mutations")

    # Statistiques sur les VAF
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

    # Remplir les NaN dans std par 0 (un seul variant)
    vaf_stats["vaf_std"] = vaf_stats["vaf_std"].fillna(0)


    high_vaf_counts = (
        maf_df[maf_df["VAF"] > 0.4]
        .groupby("ID")
        .size()
        .reset_index(name="high_vaf_mutations")
    )

    # Combiner toutes les statistiques
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


def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Créer des features cliniques interprétables et prognostiques

    Based on established prognostic factors in AML/MDS:
    - Age (continuous + categorical)
    - Blood counts and their ratios
    - Bone marrow blast percentage
    - Cytopenias (anemia, thrombocytopenia, neutropenia)

    Parameters:
    -----------
    df : pd.DataFrame avec données cliniques

    Returns:
    --------
    pd.DataFrame : Features cliniques nettoyées
    """
    clinical_df = df.copy()

    # ===== FEATURES DE BASE (nettoyage) =====


    numeric_columns = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    for col in numeric_columns:
        if col in clinical_df.columns:
            clinical_df[col] = pd.to_numeric(clinical_df[col], errors="coerce")

    # ===== RATIOS CLINIQUEMENT PERTINENTS =====

    # Ratio neutrophiles/globules blancs (mesure de la maturation granulocytaire)
    clinical_df["neutrophil_ratio"] = clinical_df["ANC"] / clinical_df["WBC"]
    clinical_df["neutrophil_ratio"] = clinical_df["neutrophil_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )


    clinical_df["monocyte_ratio"] = clinical_df["MONOCYTES"] / clinical_df["WBC"]
    clinical_df["monocyte_ratio"] = clinical_df["monocyte_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )


    clinical_df["platelet_wbc_ratio"] = clinical_df["PLT"] / clinical_df["WBC"]
    clinical_df["platelet_wbc_ratio"] = clinical_df["platelet_wbc_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    # ===== SEUILS CLINIQUES (binaires) =====


    clinical_df["anemia_moderate"] = (clinical_df["HB"] < 10).astype(int)
    clinical_df["anemia_severe"] = (clinical_df["HB"] < 8).astype(int)


    clinical_df["thrombocytopenia_moderate"] = (clinical_df["PLT"] < 100).astype(int)
    clinical_df["thrombocytopenia_severe"] = (clinical_df["PLT"] < 50).astype(int)


    clinical_df["neutropenia_moderate"] = (clinical_df["ANC"] < 1.5).astype(int)
    clinical_df["neutropenia_severe"] = (clinical_df["ANC"] < 1.0).astype(int)


    clinical_df["leukocytosis_high"] = (clinical_df["WBC"] > 30).astype(int)


    clinical_df["high_blast_count"] = (clinical_df["BM_BLAST"] > 20).astype(int)

    # ===== SCORES COMPOSITES CLINIQUES =====


    clinical_df["cytopenia_score"] = (
        clinical_df["anemia_moderate"]
        + clinical_df["thrombocytopenia_moderate"]
        + clinical_df["neutropenia_moderate"]
    )


    clinical_df["pancytopenia"] = (clinical_df["cytopenia_score"] == 3).astype(int)


    clinical_df["proliferation_score"] = (
        clinical_df["high_blast_count"] + clinical_df["leukocytosis_high"]
    )




    for col in ["WBC", "PLT", "ANC", "MONOCYTES"]:
        if col in clinical_df.columns:
            clinical_df[f"log_{col}"] = np.log1p(clinical_df[col].fillna(0))

    # ===== RATIOS ADDITIONNELS =====
    clinical_df["blast_platelet_ratio"] = clinical_df["BM_BLAST"] / clinical_df["PLT"]


    clinical_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return clinical_df


def combine_all_features(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    burden_df: pd.DataFrame,
    cyto_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all features and create integrated risk scores without data loss.

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

    # === CREATE ONE-HOT ENCODING FOR CENTER ===
    final_df = _create_center_one_hot_encoding(final_df)

    # === MERGE ALL FEATURE CATEGORIES ===
    final_df = _merge_feature_dataframes(final_df, molecular_df, burden_df, cyto_df)

    # === FILL MISSING VALUES STRATEGICALLY ===
    final_df = _fill_missing_values_strategically(final_df)

    # === CREATE INTEGRATED SCORES (PRESERVING INDIVIDUAL COMPONENTS) ===
    final_df = _create_integrated_risk_scores(final_df)

    # === ADD INTERACTION FEATURES ===
    final_df = _create_interaction_features(final_df)

    # === REMOVE CYTOGENETICS COLUMN (features already extracted) ===
    if "CYTOGENETICS" in final_df.columns:
        final_df = final_df.drop(columns=["CYTOGENETICS"])

    return final_df


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


def _create_center_one_hot_encoding(final_df: pd.DataFrame) -> pd.DataFrame:
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

    # Get unique centers, handling NaN values
    center_values = final_df["CENTER"].fillna("Unknown")
    unique_centers = sorted(center_values.unique())

    # Create one-hot encoding for each center
    for center in unique_centers:
        column_name = f"CENTER_{center}"
        final_df[column_name] = (center_values == center).astype(int)

    # Remove original CENTER column
    final_df = final_df.drop(columns=["CENTER"])

    return final_df


def _fill_missing_values_strategically(final_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values based on clinical knowledge and data characteristics."""

    # === MUTATION-RELATED FEATURES ===
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
        "vaf_std",
        "high_vaf_ratio",
    ]
    for col in burden_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    # === CYTOGENETIC FEATURES ===
    # Missing cytogenetic abnormalities = normal/not detected
    cyto_binary_cols = [
        col
        for col in final_df.columns
        if any(
            col.startswith(prefix)
            for prefix in [
                "t_",
                "inv_",
                "del_",
                "complex_",
                "normal_",
                "trisomy_",
                "abn_",
                "monosomy_",
                "any_",
            ]
        )
    ]
    for col in cyto_binary_cols:
        final_df[col] = final_df[col].fillna(0).astype(int)

    # Cytogenetic risk: missing = intermediate (most conservative)
    if "eln_cyto_risk" in final_df.columns:
        final_df["eln_cyto_risk"] = final_df["eln_cyto_risk"].fillna(1)

    # === PATHWAY FEATURES ===
    # Missing pathway alterations = no alteration
    pathway_cols = [
        col for col in final_df.columns if "_altered" in col or "_count" in col
    ]
    for col in pathway_cols:
        final_df[col] = final_df[col].fillna(0).astype(int)

    # === VAF FEATURES ===
    # Missing VAF = 0 (no mutation)
    vaf_cols = [col for col in final_df.columns if "vaf_" in col.lower()]
    for col in vaf_cols:
        final_df[col] = final_df[col].fillna(0)

    return final_df


def _create_integrated_risk_scores(final_df: pd.DataFrame) -> pd.DataFrame:
    """Create integrated risk scores while preserving individual components."""

    # === ELN 2022 INTEGRATED RISK ===
    # Start with cytogenetic risk
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

        # For other cases, take the worse of cytogenetic and molecular risk
        other_mask = ~normal_cyto_mask
        final_df.loc[other_mask, "eln_integrated_risk"] = np.maximum(
            final_df.loc[other_mask, "eln_integrated_risk"],
            final_df.loc[other_mask, "eln_molecular_risk"],
        )

    # === MUTATION BURDEN RISK SCORE ===
    if "total_mutations" in final_df.columns:
        # Create continuous mutation burden score (normalized)
        final_df["mutation_burden_score"] = (
            final_df["total_mutations"] / final_df["total_mutations"].max()
        )

        # Create categorical mutation burden (for interpretability)
        final_df["mutation_burden_category"] = pd.cut(
            final_df["total_mutations"], bins=[-np.inf, 2, 5, np.inf], labels=[0, 1, 2]
        ).astype(int)

    # === CLINICAL COMPOSITE SCORES ===
    final_df["clinical_risk_score"] = final_df.get("cytopenia_score", 0) + final_df.get(
        "proliferation_score", 0
    )

    # === COMPREHENSIVE RISK SCORE ===
    # Combine all risk dimensions with weights based on clinical evidence
    risk_components = []

    if "eln_integrated_risk" in final_df.columns:
        risk_components.append(final_df["eln_integrated_risk"] * 0.4)  # 40% weight

    if "clinical_risk_score" in final_df.columns:
        # Normalize clinical score (0-1 scale)
        clinical_normalized = (
            final_df["clinical_risk_score"] / final_df["clinical_risk_score"].max()
        )
        risk_components.append(clinical_normalized * 0.3)  # 30% weight

    if "mutation_burden_score" in final_df.columns:
        risk_components.append(final_df["mutation_burden_score"] * 0.3)  # 30% weight

    if risk_components:
        final_df["comprehensive_risk_score"] = sum(risk_components)

    return final_df


def _create_interaction_features(final_df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically relevant interaction features."""

    # === TP53 + COMPLEX CYTOGENETICS ===
    # Very high-risk combination
    if "mut_TP53" in final_df.columns and "complex_karyotype" in final_df.columns:
        final_df["TP53_complex_cyto"] = (
            (final_df["mut_TP53"] == 1) & (final_df["complex_karyotype"] == 1)
        ).astype(int)

    # === FLT3 + NPM1 INTERACTIONS ===
    # Different risk levels based on combination
    if "mut_FLT3" in final_df.columns and "mut_NPM1" in final_df.columns:
        final_df["FLT3_NPM1_interaction"] = 0  # Default: neither

        # NPM1+/FLT3- = favorable (1)
        npm1_only = (final_df["mut_NPM1"] == 1) & (final_df["mut_FLT3"] == 0)
        final_df.loc[npm1_only, "FLT3_NPM1_interaction"] = 1

        # NPM1+/FLT3+ = intermediate (2)
        both_mut = (final_df["mut_NPM1"] == 1) & (final_df["mut_FLT3"] == 1)
        final_df.loc[both_mut, "FLT3_NPM1_interaction"] = 2

        # NPM1-/FLT3+ = adverse (3)
        flt3_only = (final_df["mut_NPM1"] == 0) & (final_df["mut_FLT3"] == 1)
        final_df.loc[flt3_only, "FLT3_NPM1_interaction"] = 3

    # === AGE + MOLECULAR INTERACTIONS ===
    # Age modifies molecular risk impact
    if "mut_DNMT3A" in final_df.columns and final_df["AGE"].notna().any():
        # DNMT3A mutations have different impact in older vs younger patients
        final_df["DNMT3A_age_interaction"] = final_df["mut_DNMT3A"] * (
            final_df["AGE"] > 60
        ).astype(int)

    return final_df
