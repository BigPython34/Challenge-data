# Feature engineering (cytogénétique, moléculaire, clinique)
import pandas as pd
import numpy as np

from .. import config


def extract_advanced_cytogenetic_features(df):
    """Extraction avancée de features cytogénétiques"""
    # Original cytogenetic features
    cyto_features = pd.DataFrame(index=df.index)

    # Basic karyotype classification
    cyto_features["normal_karyotype"] = (
        df["CYTOGENETICS"]
        .str.contains(r"^46,X[XY]$", regex=True)
        .fillna(False)
        .astype(int)
    )

    # High-risk abnormalities
    cyto_features["complex_karyotype"] = df["CYTOGENETICS"].str.count(",") >= 3
    cyto_features["monosomy_7"] = (
        df["CYTOGENETICS"]
        .str.contains(r"-7|\(7;|\b7q-", regex=True)
        .fillna(False)
        .astype(int)
    )
    cyto_features["del_5q"] = (
        df["CYTOGENETICS"]
        .str.contains(r"del\(5q\)|\b5q-", regex=True)
        .fillna(False)
        .astype(int)
    )
    cyto_features["inv_3"] = (
        df["CYTOGENETICS"]
        .str.contains(r"inv\(3\)|t\(3;3\)", regex=True)
        .fillna(False)
        .astype(int)
    )
    cyto_features["abn_17p"] = (
        df["CYTOGENETICS"]
        .str.contains(r"del\(17p\)|17p-", regex=True)
        .fillna(False)
        .astype(int)
    )

    # Intermediate-risk abnormalities
    cyto_features["trisomy_8"] = (
        df["CYTOGENETICS"].str.contains(r"\+8", regex=True).fillna(False).astype(int)
    )
    cyto_features["t_9_11"] = (
        df["CYTOGENETICS"]
        .str.contains(r"t\(9;11\)", regex=True)
        .fillna(False)
        .astype(int)
    )

    # Favorable abnormalities
    cyto_features["inv_16"] = (
        df["CYTOGENETICS"]
        .str.contains(r"inv\(16\)|t\(16;16\)", regex=True)
        .fillna(False)
        .astype(int)
    )
    cyto_features["t_8_21"] = (
        df["CYTOGENETICS"]
        .str.contains(r"t\(8;21\)", regex=True)
        .fillna(False)
        .astype(int)
    )

    # Additional important abnormalities
    cyto_features["t_6_9"] = (
        df["CYTOGENETICS"]
        .str.contains(r"t\(6;9\)", regex=True)
        .fillna(False)
        .astype(int)
    )
    cyto_features["t_11q23"] = (
        df["CYTOGENETICS"]
        .str.contains(r"t\(11;|11q23", regex=True)
        .fillna(False)
        .astype(int)
    )
    cyto_features["monosomy_5"] = (
        df["CYTOGENETICS"].str.contains(r"-5", regex=True).fillna(False).astype(int)
    )
    cyto_features["del_7q"] = (
        df["CYTOGENETICS"]
        .str.contains(r"del\(7q\)|\b7q-", regex=True)
        .fillna(False)
        .astype(int)
    )
    cyto_features["abn_3q"] = (
        df["CYTOGENETICS"].str.contains(r"3q", regex=True).fillna(False).astype(int)
    )

    # Count number of abnormalities
    cyto_features["num_abnormalities"] = df["CYTOGENETICS"].str.count(",")
    cyto_features["num_abnormalities"].fillna(0, inplace=True)

    # Create cytogenetic risk score based on ELN 2017 classification
    cyto_features["cyto_risk_score"] = 1  # Default: intermediate

    # Favorable
    favorable_mask = (cyto_features["t_8_21"] == 1) | (cyto_features["inv_16"] == 1)
    cyto_features.loc[favorable_mask, "cyto_risk_score"] = 0

    # Adverse
    adverse_mask = (
        (cyto_features["complex_karyotype"] == 1)
        | (cyto_features["monosomy_7"] == 1)
        | (cyto_features["monosomy_5"] == 1)
        | (cyto_features["del_5q"] == 1)
        | (cyto_features["del_7q"] == 1)
        | (cyto_features["inv_3"] == 1)
        | (cyto_features["abn_17p"] == 1)
        | (cyto_features["abn_3q"] == 1)
        | (cyto_features["t_6_9"] == 1)
    )
    cyto_features.loc[adverse_mask, "cyto_risk_score"] = 2

    # Convert boolean to int
    for col in cyto_features.columns:
        if cyto_features[col].dtype == bool:
            cyto_features[col] = cyto_features[col].astype(int)

    return cyto_features


def create_advanced_molecular_features(df, maf_df, important_genes=None):
    """Création avancée de features moléculaires"""
    if important_genes is None:
        important_genes = config.IMPORTANT_GENES

    # Create a DataFrame to store gene mutation features
    gene_features = pd.DataFrame(index=df["ID"].unique())

    # Basic mutation presence features
    for gene in important_genes:
        mutated_patients = maf_df[maf_df["GENE"] == gene]["ID"].unique()
        gene_features[f"has_{gene}_mutation"] = gene_features.index.isin(
            mutated_patients
        ).astype(int)

    # Functional effect of mutations
    effect_types = ["missense", "nonsense", "frameshift", "splice_site"]

    for gene in important_genes:
        for effect in effect_types:
            patients = maf_df[
                (maf_df["GENE"] == gene)
                & (maf_df["EFFECT"].str.contains(effect, case=False, na=False))
            ]["ID"].unique()
            gene_features[f"{gene}_{effect}"] = gene_features.index.isin(
                patients
            ).astype(int)

    # VAF-weighted features
    for gene in important_genes:
        gene_vaf = (
            maf_df[maf_df["GENE"] == gene].groupby("ID")["VAF"].max().reset_index()
        )
        gene_vaf_dict = dict(zip(gene_vaf["ID"], gene_vaf["VAF"]))
        gene_features[f"{gene}_vaf"] = gene_features.index.map(gene_vaf_dict).fillna(0)
        gene_features[f"{gene}_impact"] = (
            gene_features[f"has_{gene}_mutation"] * gene_features[f"{gene}_vaf"]
        )

    # Clinically significant co-mutations
    gene_features["FLT3_NPM1_comut"] = (
        gene_features["has_FLT3_mutation"] & gene_features["has_NPM1_mutation"]
    ).astype(int)
    gene_features["DNMT3A_NPM1_comut"] = (
        gene_features["has_DNMT3A_mutation"] & gene_features["has_NPM1_mutation"]
    ).astype(int)

    # Pathway disruptions
    gene_features["TP53_pathway_disruption"] = gene_features["has_TP53_mutation"]
    gene_features["methylation_pathway_disruption"] = (
        (
            gene_features["has_DNMT3A_mutation"]
            | gene_features["has_TET2_mutation"]
            | gene_features["has_IDH1_mutation"]
            | gene_features["has_IDH2_mutation"]
        )
    ).astype(int)
    gene_features["splicing_pathway_disruption"] = (
        (gene_features["has_SRSF2_mutation"] | gene_features["has_SF3B1_mutation"])
    ).astype(int)

    # Molecular risk score
    gene_features["molecular_risk_score"] = 0
    favorable_mask = (gene_features["has_NPM1_mutation"] == 1) & (
        gene_features["has_FLT3_mutation"] == 0
    )
    gene_features.loc[favorable_mask, "molecular_risk_score"] = -1
    adverse_mask = (
        (gene_features["has_TP53_mutation"] == 1)
        | (gene_features["has_ASXL1_mutation"] == 1)
        | (gene_features["has_RUNX1_mutation"] == 1)
    )
    gene_features.loc[adverse_mask, "molecular_risk_score"] = 1

    return gene_features.reset_index().rename(columns={"index": "ID"})


def create_molecular_stats_features(maf_df):
    """Créer des statistiques sur les mutations"""
    # Mutation counts
    mutation_counts = maf_df.groupby("ID").size().reset_index(name="Nmut")

    # VAF statistics
    vaf_stats = (
        maf_df.groupby("ID")["VAF"].agg(["mean", "max", "min", "std"]).reset_index()
    )
    vaf_stats.rename(
        columns={
            "mean": "vaf_mean",
            "max": "vaf_max",
            "min": "vaf_min",
            "std": "vaf_std",
        },
        inplace=True,
    )

    # Effect counts
    effect_counts = pd.crosstab(maf_df["ID"], maf_df["EFFECT"]).reset_index()

    return mutation_counts, vaf_stats, effect_counts


def create_advanced_clinical_features(df):
    """Création avancée de features cliniques"""
    df_clinical = df.copy()

    # Basic ratio features
    df_clinical["ANC_to_WBC_ratio"] = df_clinical["ANC"] / df_clinical["WBC"]
    df_clinical["MONO_to_WBC_ratio"] = df_clinical["MONOCYTES"] / df_clinical["WBC"]
    df_clinical["MONO_to_LYM_ratio"] = df_clinical["MONOCYTES"] / (
        df_clinical["WBC"] - df_clinical["ANC"] - df_clinical["MONOCYTES"]
    )
    df_clinical["PLT_to_WBC_ratio"] = df_clinical["PLT"] / df_clinical["WBC"]

    # Replace infinities and NaNs
    df_clinical.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clinical.fillna(0, inplace=True)

    # Log transformations for skewed distributions
    for col in ["WBC", "PLT", "BM_BLAST"]:
        df_clinical[f"log_{col}"] = np.log1p(df_clinical[col])

    # Categorical features based on clinical thresholds
    df_clinical["severe_anemia"] = (df_clinical["HB"] < 8).astype(int)
    df_clinical["moderate_anemia"] = (
        (df_clinical["HB"] >= 8) & (df_clinical["HB"] < 10)
    ).astype(int)
    df_clinical["thrombocytopenia"] = (df_clinical["PLT"] < 100).astype(int)
    df_clinical["severe_thrombocytopenia"] = (df_clinical["PLT"] < 50).astype(int)
    df_clinical["leukocytosis"] = (df_clinical["WBC"] > 25).astype(int)
    df_clinical["high_blast"] = (df_clinical["BM_BLAST"] > 20).astype(int)

    # Composite clinical scores
    df_clinical["cytopenia_score"] = 0
    df_clinical.loc[df_clinical["HB"] < 10, "cytopenia_score"] += 1
    df_clinical.loc[df_clinical["PLT"] < 100, "cytopenia_score"] += 1
    df_clinical.loc[df_clinical["ANC"] < 1, "cytopenia_score"] += 1

    # Pancytopenia feature (all three cytopenias present)
    df_clinical["pancytopenia"] = (
        (df_clinical["HB"] < 10) & (df_clinical["PLT"] < 100) & (df_clinical["ANC"] < 1)
    ).astype(int)
    df_clinical.loc[df_clinical["ANC"] < 1.8, "cytopenia_score"] += 1

    df_clinical["proliferation_index"] = 0
    df_clinical.loc[df_clinical["BM_BLAST"] > 10, "proliferation_index"] += 1
    df_clinical.loc[df_clinical["WBC"] > 25, "proliferation_index"] += 1

    # Interaction terms
    df_clinical["blast_wbc_interaction"] = (
        df_clinical["BM_BLAST"] * df_clinical["WBC"] / 100
    )
    df_clinical["hb_plt_interaction"] = df_clinical["HB"] * df_clinical["PLT"] / 100

    # Clinical risk score
    df_clinical["clinical_risk_score"] = 0
    df_clinical["clinical_risk_score"] += df_clinical["cytopenia_score"]
    df_clinical.loc[df_clinical["BM_BLAST"] > 5, "clinical_risk_score"] += 1
    df_clinical.loc[df_clinical["BM_BLAST"] > 10, "clinical_risk_score"] += 1
    df_clinical.loc[df_clinical["BM_BLAST"] > 20, "clinical_risk_score"] += 2

    return df_clinical


def get_feature_lists():
    """Retourne les listes de features organisées par catégorie"""
    clinical_features = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    gene_mut_features = [f"has_{gene}_mutation" for gene in config.IMPORTANT_GENES]
    stat_features = ["Nmut", "vaf_mean", "vaf_max", "vaf_std"]
    cyto_feature_list = [
        "normal_karyotype",
        "complex_karyotype",
        "monosomy_7",
        "del_5q",
        "inv_3",
        "abn_17p",
        "trisomy_8",
        "t_9_11",
        "inv_16",
        "t_8_21",
    ]
    ratio_features = ["ANC_to_WBC_ratio", "MONO_to_WBC_ratio"]
    clinical_score_features = ["MONO_to_LYM_ratio", "cytopenia_score", "pancytopenia"]

    return {
        "clinical": clinical_features,
        "gene_mutations": gene_mut_features,
        "statistics": stat_features,
        "cytogenetic": cyto_feature_list,
        "ratios": ratio_features,
        "clinical_scores": clinical_score_features,
    }
