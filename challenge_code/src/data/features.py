"""
Feature engineering pour la modélisation de survie en leucémie myéloïde aiguë (LMA)

Ce module gère la création de features cliniquement pertinentes pour prédire
la survie globale chez des patients atteints de LMA.

Principes directeurs:
1. Features basées sur des connaissances médicales établies (ELN 2017, WHO)
2. Simplicité et interprétabilité pour les cliniciens
3. Robustesse aux données manquantes
4. Pertinence pour la modélisation de survie
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

from .. import config


def extract_cytogenetic_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extraction de features cytogénétiques basées sur la classification ELN 2017

    Créer des variables binaires pour les anomalies cytogénétiques les plus importantes
    en pronostic de survie pour la LMA selon les guidelines européennes (ELN 2017).

    Parameters:
    -----------
    df : pd.DataFrame avec colonne CYTOGENETICS

    Returns:
    --------
    pd.DataFrame : Features cytogénétiques (index = patients)
    """
    cyto_df = pd.DataFrame(index=df.index)

    # Gestion des valeurs manquantes - considérées comme cytogénétique normale
    cytogenetics_clean = df["CYTOGENETICS"].fillna("46,XX")  # Assume normal if missing

    # ===== ANOMALIES DE BON PRONOSTIC (Favorable) =====

    # t(8;21)(q22;q22) - CBF-AML, bon pronostic
    cyto_df["t_8_21"] = (
        cytogenetics_clean.str.contains(r"t\(8;21\)", regex=True, na=False)
    ).astype(int)

    # inv(16)(p13.1q22) ou t(16;16)(p13.1;q22) - CBF-AML, bon pronostic
    cyto_df["inv_16"] = (
        cytogenetics_clean.str.contains(r"inv\(16\)|t\(16;16\)", regex=True, na=False)
    ).astype(int)

    # t(15;17) - LAM3, très bon pronostic (mais rare dans cette cohorte MDS/sAML)
    cyto_df["t_15_17"] = (
        cytogenetics_clean.str.contains(r"t\(15;17\)", regex=True, na=False)
    ).astype(int)

    # ===== ANOMALIES DE PRONOSTIC INTERMÉDIAIRE =====

    # Cytogénétique normale (46,XX ou 46,XY)
    cyto_df["normal_karyotype"] = (
        cytogenetics_clean.str.match(r"^46,X[XY]$", na=False)
    ).astype(int)

    # Trisomie 8 - fréquente, pronostic intermédiaire
    cyto_df["trisomy_8"] = (
        cytogenetics_clean.str.contains(r"\+8\b", regex=True, na=False)
    ).astype(int)

    # ===== ANOMALIES DE PRONOSTIC DÉFAVORABLE (Adverse) =====

    # Caryotype complexe (≥3 anomalies) - très mauvais pronostic
    cyto_df["complex_karyotype"] = (cytogenetics_clean.str.count(",") >= 3).astype(int)

    # Monosomie 7 / délétion 7q - très mauvais pronostic
    cyto_df["monosomy_7"] = (
        cytogenetics_clean.str.contains(r"-7\b|del\(7q\)|\b7q-", regex=True, na=False)
    ).astype(int)

    # Délétion 5q / monosomie 5 - mauvais pronostic, fréquent en MDS
    cyto_df["del_5q"] = (
        cytogenetics_clean.str.contains(r"del\(5q\)|\b5q-|-5\b", regex=True, na=False)
    ).astype(int)

    # Anomalies de 3q - mauvais pronostic
    cyto_df["abn_3q"] = (
        cytogenetics_clean.str.contains(r"inv\(3\)|t\(3;3\)", regex=True, na=False)
    ).astype(int)

    # Anomalies de 17p (incluant TP53) - très mauvais pronostic
    cyto_df["abn_17p"] = (
        cytogenetics_clean.str.contains(r"del\(17p\)|17p-|\(17;", regex=True, na=False)
    ).astype(int)

    # t(6;9) - mauvais pronostic
    cyto_df["t_6_9"] = (
        cytogenetics_clean.str.contains(r"t\(6;9\)", regex=True, na=False)
    ).astype(int)

    # ===== SCORES DE RISQUE CYTOGÉNÉTIQUE =====

    # Score de risque ELN 2017 simplifié
    cyto_df["eln_cyto_risk"] = 1  # Par défaut: intermédiaire

    # Favorable (0)
    favorable_mask = (
        (cyto_df["t_8_21"] == 1) | (cyto_df["inv_16"] == 1) | (cyto_df["t_15_17"] == 1)
    )
    cyto_df.loc[favorable_mask, "eln_cyto_risk"] = 0

    # Défavorable (2)
    adverse_mask = (
        (cyto_df["complex_karyotype"] == 1)
        | (cyto_df["monosomy_7"] == 1)
        | (cyto_df["del_5q"] == 1)
        | (cyto_df["abn_3q"] == 1)
        | (cyto_df["abn_17p"] == 1)
        | (cyto_df["t_6_9"] == 1)
    )
    cyto_df.loc[adverse_mask, "eln_cyto_risk"] = 2

    # Nombre total d'anomalies cytogénétiques (proxy de complexité)
    cyto_df["num_cyto_abnormalities"] = cytogenetics_clean.str.count(",")
    cyto_df["num_cyto_abnormalities"] = (
        cyto_df["num_cyto_abnormalities"].fillna(0).astype(int)
    )

    return cyto_df


def extract_molecular_risk_features(
    df: pd.DataFrame, maf_df: pd.DataFrame, important_genes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Création de features moléculaires basées sur les mutations prognostiques

    Focus sur les gènes avec impact pronostique établi en LMA selon ELN 2017
    et les découvertes récentes de la littérature.

    Parameters:
    -----------
    df : pd.DataFrame avec colonnes patients (ID)
    maf_df : pd.DataFrame avec mutations (ID, GENE, VAF, EFFECT)
    important_genes : Liste des gènes à analyser (optionnel)

    Returns:
    --------
    pd.DataFrame : Features moléculaires (index = ID patients)
    """
    if important_genes is None:
        # Gènes avec impact pronostique établi en LMA
        important_genes = [
            # Bon pronostic
            "NPM1",
            "CEBPA",
            # Mauvais pronostic
            "TP53",
            "FLT3",
            "ASXL1",
            "RUNX1",
            "MECOM",
            # Pronostic intermédiaire/émergent
            "DNMT3A",
            "TET2",
            "IDH1",
            "IDH2",
            "SF3B1",
            "SRSF2",
            "U2AF1",
            "NRAS",
            "KRAS",
            "PTPN11",
        ]

    # DataFrame pour stocker les features par patient
    molecular_df = pd.DataFrame(index=df["ID"].unique())

    # ===== MUTATIONS BINAIRES PAR GÈNE =====

    for gene in important_genes:
        if gene in maf_df["GENE"].values:
            mutated_patients = maf_df[maf_df["GENE"] == gene]["ID"].unique()
            molecular_df[f"mut_{gene}"] = molecular_df.index.isin(
                mutated_patients
            ).astype(int)
        else:
            molecular_df[f"mut_{gene}"] = 0

    # ===== VAF MAXIMALE PAR GÈNE (pour gènes clés) =====

    # Pour certains gènes, la VAF peut être prognostique
    high_impact_genes = ["TP53", "FLT3", "NPM1", "CEBPA"]

    for gene in high_impact_genes:
        if gene in maf_df["GENE"].values:
            gene_vaf = maf_df[maf_df["GENE"] == gene].groupby("ID")["VAF"].max()
            molecular_df[f"vaf_max_{gene}"] = molecular_df.index.map(gene_vaf).fillna(0)
        else:
            molecular_df[f"vaf_max_{gene}"] = 0.0

    # ===== TYPES DE MUTATIONS POUR GÈNES CLÉS =====

    # Pour TP53: les mutations non-sens/frameshift sont plus graves
    if "TP53" in maf_df["GENE"].values:
        tp53_patients = maf_df[maf_df["GENE"] == "TP53"]

        # Mutations tronquantes (loss-of-function)
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

    # Pour FLT3: distinction ITD vs TKD (si informations disponibles)
    if "FLT3" in maf_df["GENE"].values:
        flt3_patients = maf_df[maf_df["GENE"] == "FLT3"]

        # FLT3-ITD souvent associé à VAF élevée
        flt3_high_vaf = flt3_patients[flt3_patients["VAF"] > 0.5]["ID"].unique()
        molecular_df["FLT3_high_VAF"] = molecular_df.index.isin(flt3_high_vaf).astype(
            int
        )
    else:
        molecular_df["FLT3_high_VAF"] = 0

    # ===== CO-MUTATIONS PROGNOSTIQUES =====

    # NPM1+/FLT3- : bon pronostic (si cytogénétique normale)
    molecular_df["NPM1_pos_FLT3_neg"] = (
        (molecular_df.get("mut_NPM1", 0) == 1) & (molecular_df.get("mut_FLT3", 0) == 0)
    ).astype(int)

    # TP53 + anomalies cytogénétiques complexes (synergie très défavorable)
    # Cette information sera combinée plus tard avec les données cytogénétiques

    # DNMT3A + NPM1 + FLT3 : triple mutation fréquente
    molecular_df["DNMT3A_NPM1_comut"] = (
        (molecular_df.get("mut_DNMT3A", 0) == 1)
        & (molecular_df.get("mut_NPM1", 0) == 1)
    ).astype(int)

    # ===== VOIES DE SIGNALISATION ALTÉRÉES =====

    # Voie de méthylation de l'ADN (épigénétique)
    methylation_genes = ["DNMT3A", "TET2", "IDH1", "IDH2"]
    molecular_df["methylation_pathway_altered"] = (
        molecular_df[
            [
                f"mut_{gene}"
                for gene in methylation_genes
                if f"mut_{gene}" in molecular_df.columns
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)

    # Voie du splicing (spliceosome)
    splicing_genes = ["SF3B1", "SRSF2", "U2AF1"]
    molecular_df["splicing_pathway_altered"] = (
        molecular_df[
            [
                f"mut_{gene}"
                for gene in splicing_genes
                if f"mut_{gene}" in molecular_df.columns
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)

    # Voie RAS (prolifération)
    ras_genes = ["NRAS", "KRAS", "PTPN11"]
    molecular_df["ras_pathway_altered"] = (
        molecular_df[
            [
                f"mut_{gene}"
                for gene in ras_genes
                if f"mut_{gene}" in molecular_df.columns
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)

    # ===== SCORE DE RISQUE MOLÉCULAIRE ELN 2017 =====

    molecular_df["eln_molecular_risk"] = 1  # Par défaut: intermédiaire

    # Favorable (0)
    favorable_molecular = (molecular_df.get("mut_NPM1", 0) == 1) & (
        molecular_df.get("mut_FLT3", 0) == 0
    ) | (  # NPM1+/FLT3-
        molecular_df.get("mut_CEBPA", 0) == 1
    )  # CEBPA biallelic (simplifié)
    molecular_df.loc[favorable_molecular, "eln_molecular_risk"] = 0

    # Défavorable (2)
    adverse_molecular = (
        (molecular_df.get("TP53_truncating", 0) == 1)
        | (molecular_df.get("mut_ASXL1", 0) == 1)
        | (molecular_df.get("mut_RUNX1", 0) == 1)
        | (molecular_df.get("FLT3_high_VAF", 0) == 1)  # FLT3-ITD à VAF élevée
    )
    molecular_df.loc[adverse_molecular, "eln_molecular_risk"] = 2

    return molecular_df.reset_index().rename(columns={"index": "ID"})


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

    # Proportion de mutations à VAF élevée (>0.4, possibles mutations germinales)
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

    # Ratio de mutations à VAF élevée
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

    # Assurer que les valeurs sont numériques
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

    # Ratio monocytes/globules blancs (élévation = pronostic défavorable)
    clinical_df["monocyte_ratio"] = clinical_df["MONOCYTES"] / clinical_df["WBC"]
    clinical_df["monocyte_ratio"] = clinical_df["monocyte_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    # Ratio plaquettes/globules blancs (mesure générale de l'hématopoïèse)
    clinical_df["platelet_wbc_ratio"] = clinical_df["PLT"] / clinical_df["WBC"]
    clinical_df["platelet_wbc_ratio"] = clinical_df["platelet_wbc_ratio"].replace(
        [np.inf, -np.inf], np.nan
    )

    # ===== SEUILS CLINIQUES (binaires) =====

    # Anémie (HB < 10 g/dL = modérée, < 8 g/dL = sévère)
    clinical_df["anemia_moderate"] = (clinical_df["HB"] < 10).astype(int)
    clinical_df["anemia_severe"] = (clinical_df["HB"] < 8).astype(int)

    # Thrombocytopénie (PLT < 100 = modérée, < 50 = sévère)
    clinical_df["thrombocytopenia_moderate"] = (clinical_df["PLT"] < 100).astype(int)
    clinical_df["thrombocytopenia_severe"] = (clinical_df["PLT"] < 50).astype(int)

    # Neutropénie (ANC < 1.5 = modérée, < 1.0 = sévère)
    clinical_df["neutropenia_moderate"] = (clinical_df["ANC"] < 1.5).astype(int)
    clinical_df["neutropenia_severe"] = (clinical_df["ANC"] < 1.0).astype(int)

    # Leucocytose (WBC > 30 = élevée)
    clinical_df["leukocytosis_high"] = (clinical_df["WBC"] > 30).astype(int)

    # Blastose médullaire élevée (>20% = LMA)
    clinical_df["high_blast_count"] = (clinical_df["BM_BLAST"] > 20).astype(int)

    # ===== SCORES COMPOSITES CLINIQUES =====

    # Score de cytopénie (0-3, basé sur anémie + thrombocytopénie + neutropénie)
    clinical_df["cytopenia_score"] = (
        clinical_df["anemia_moderate"]
        + clinical_df["thrombocytopenia_moderate"]
        + clinical_df["neutropenia_moderate"]
    )

    # Pancytopénie (toutes les lignées affectées)
    clinical_df["pancytopenia"] = (clinical_df["cytopenia_score"] == 3).astype(int)

    # Score de prolifération (blastose + leucocytose)
    clinical_df["proliferation_score"] = (
        clinical_df["high_blast_count"] + clinical_df["leukocytosis_high"]
    )

    # ===== TRANSFORMATIONS LOG (pour distributions asymétriques) =====

    # Log transformation pour les variables très asymétriques
    for col in ["WBC", "PLT", "ANC", "MONOCYTES"]:
        if col in clinical_df.columns:
            clinical_df[f"log_{col}"] = np.log1p(clinical_df[col].fillna(0))

    # ===== RATIOS ADDITIONNELS =====
    clinical_df["blast_platelet_ratio"] = clinical_df["BM_BLAST"] / clinical_df["PLT"]

    # Gérer les infinis
    clinical_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return clinical_df


def combine_all_features(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    burden_df: pd.DataFrame,
    cyto_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combiner toutes les features et créer des scores de risque intégrés

    Parameters:
    -----------
    clinical_df, molecular_df, burden_df, cyto_df : DataFrames avec features

    Returns:
    --------
    pd.DataFrame : Dataset final avec toutes les features
    """
    # Merger toutes les features sur l'ID patient
    final_df = clinical_df.copy()

    # S'assurer que tous les ID sont de même type
    final_df["ID"] = final_df["ID"].astype(str)

    # Merge molecular features
    if not molecular_df.empty:
        molecular_df["ID"] = molecular_df["ID"].astype(str)
        final_df = final_df.merge(molecular_df, on="ID", how="left")

    # Merge burden features
    if not burden_df.empty:
        burden_df["ID"] = burden_df["ID"].astype(str)
        final_df = final_df.merge(burden_df, on="ID", how="left")

        # Fill NaN pour les patients sans mutations
        mutation_cols = [
            "total_mutations",
            "high_vaf_mutations",
            "vaf_mean",
            "vaf_median",
            "vaf_max",
            "vaf_std",
            "high_vaf_ratio",
        ]
        for col in mutation_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(0)

    # Merge cytogenetic features
    if not cyto_df.empty:
        cyto_df_reset = cyto_df.reset_index()
        # S'assurer que la colonne ID existe et est du bon type
        if "index" in cyto_df_reset.columns:
            cyto_df_reset = cyto_df_reset.rename(columns={"index": "ID"})
        cyto_df_reset["ID"] = cyto_df_reset["ID"].astype(str)
        final_df = final_df.merge(cyto_df_reset, on="ID", how="left")

    # Fill NaN pour features cytogénétiques (= normal si manquant)
    cyto_cols = [
        col
        for col in final_df.columns
        if col.startswith(
            (
                "t_",
                "inv_",
                "del_",
                "complex_",
                "normal_",
                "trisomy_",
                "abn_",
                "monosomy_",
            )
        )
    ]
    for col in cyto_cols:
        final_df[col] = final_df[col].fillna(0).astype(int)

    if "eln_cyto_risk" in final_df.columns:
        final_df["eln_cyto_risk"] = final_df["eln_cyto_risk"].fillna(
            1
        )  # Intermédiaire si manquant

    # ===== SCORES DE RISQUE INTÉGRÉS =====

    # Score ELN 2017 combiné (cytogénétique + moléculaire)
    final_df["eln_integrated_risk"] = final_df.get("eln_cyto_risk", 1)

    # Ajuster selon le risque moléculaire
    if "eln_molecular_risk" in final_df.columns:
        # Si cytogénétique normale, utiliser le risque moléculaire
        normal_cyto_mask = final_df.get("normal_karyotype", 0) == 1
        final_df.loc[normal_cyto_mask, "eln_integrated_risk"] = final_df.loc[
            normal_cyto_mask, "eln_molecular_risk"
        ]

        # Pour les autres, prendre le plus mauvais des deux
        final_df["eln_integrated_risk"] = np.maximum(
            final_df["eln_integrated_risk"], final_df["eln_molecular_risk"]
        )

    # Score de charge mutationnelle (normalized)
    if "total_mutations" in final_df.columns:
        # Catégoriser le nombre de mutations (0: faible, 1: intermédiaire, 2: élevé)
        final_df["mutation_burden_category"] = pd.cut(
            final_df["total_mutations"], bins=[-np.inf, 2, 5, np.inf], labels=[0, 1, 2]
        ).astype(int)

    # Score clinique intégré
    final_df["clinical_risk_score"] = final_df.get("cytopenia_score", 0) + final_df.get(
        "proliferation_score", 0
    )

    return final_df


def get_clean_feature_lists() -> Dict[str, List[str]]:
    """
    Retourner les listes de features organisées par catégorie

    Version nettoyée avec seulement les features cliniquement pertinentes

    Returns:
    --------
    Dict : Features organisées par catégorie
    """

    # Features cliniques de base (mesures objectives)
    clinical_base = [
        "BM_BLAST",  # % de blastes médullaires
        "WBC",
        "ANC",  # Globules blancs et neutrophiles
        "MONOCYTES",  # Monocytes
        "HB",
        "PLT",  # Hémoglobine et plaquettes
    ]

    # Ratios cliniques (normalisés, robustes)
    clinical_ratios = [
        "neutrophil_ratio",  # ANC/WBC
        "monocyte_ratio",  # MONO/WBC
        "platelet_wbc_ratio",  # PLT/WBC
    ]

    # Features cliniques binaires (seuils établis)
    clinical_binary = [
        "anemia_severe",  # HB < 8
        "thrombocytopenia_severe",  # PLT < 50
        "neutropenia_severe",  # ANC < 1.0
        "leukocytosis_high",  # WBC > 30
        "high_blast_count",  # Blastes > 20%
    ]

    # Scores cliniques composites
    clinical_scores = [
        "cytopenia_score",  # 0-3, nombre de cytopénies
        "pancytopenia",  # Toutes lignées affectées
        "proliferation_score",  # Blastose + leucocytose
    ]

    # Features cytogénétiques (ELN 2017)
    cytogenetic = [
        "normal_karyotype",  # 46,XX/XY
        "t_8_21",
        "inv_16",  # Favorable
        "complex_karyotype",
        "monosomy_7",
        "del_5q",
        "abn_3q",
        "abn_17p",  # Adverse
        "eln_cyto_risk",  # Score 0-2
    ]

    # Mutations des gènes clés (binaire)
    molecular_mutations = [
        # Bon pronostic
        "mut_NPM1",
        "mut_CEBPA",
        # Mauvais pronostic
        "mut_TP53",
        "mut_FLT3",
        "mut_ASXL1",
        "mut_RUNX1",
        # Épigénétique
        "mut_DNMT3A",
        "mut_TET2",
        "mut_IDH1",
        "mut_IDH2",
        # Splicing
        "mut_SF3B1",
        "mut_SRSF2",
        "mut_U2AF1",
    ]

    # Features moléculaires dérivées
    molecular_derived = [
        "NPM1_pos_FLT3_neg",  # Co-mutation favorable
        "TP53_truncating",  # Mutations loss-of-function TP53
        "methylation_pathway_altered",  # Voie épigénétique
        "splicing_pathway_altered",  # Voie splicing
        "eln_molecular_risk",  # Score ELN moléculaire
    ]

    # Charge mutationnelle
    mutation_burden = [
        "total_mutations",  # Nombre total
        "vaf_mean",
        "vaf_max",  # Statistiques VAF
        "high_vaf_ratio",  # Proportion VAF > 0.4
        "mutation_burden_category",  # Catégorie 0-2
    ]

    # Scores intégrés
    integrated_scores = [
        "eln_integrated_risk",  # ELN 2017 complet
        "clinical_risk_score",  # Score clinique composite
    ]

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


# ===== COMPATIBILITÉ AVEC L'ANCIEN CODE =====


# Fonctions de compatibilité pour maintenir l'interface existante
def get_feature_lists():
    """Version de compatibilité avec l'ancien code"""
    new_lists = get_clean_feature_lists()

    # Mapping vers l'ancien format
    return {
        "clinical": new_lists["clinical_base"] + new_lists["clinical_ratios"],
        "gene_mutations": new_lists["molecular_mutations"],
        "statistics": new_lists["mutation_burden"][:4],  # vaf_mean, vaf_max, etc.
        "cytogenetic": new_lists["cytogenetic"][:-1],  # sans eln_cyto_risk
        "ratios": new_lists["clinical_ratios"],
        "clinical_scores": new_lists["clinical_scores"],
    }


def extract_advanced_cytogenetic_features(df):
    """Version de compatibilité avec l'ancien code"""
    return extract_cytogenetic_risk_features(df)


def create_advanced_molecular_features(df, maf_df, important_genes=None):
    """Version de compatibilité avec l'ancien code"""
    return extract_molecular_risk_features(df, maf_df, important_genes)


def create_molecular_stats_features(maf_df):
    """Version de compatibilité avec l'ancien code"""
    burden_df = create_molecular_burden_features(maf_df)

    # Reformater pour correspondre à l'ancien format
    mutation_counts = burden_df[["ID", "total_mutations"]].rename(
        columns={"total_mutations": "Nmut"}
    )

    vaf_stats = burden_df[["ID", "vaf_mean", "vaf_max", "vaf_std"]].copy()
    vaf_stats["vaf_min"] = 0  # Approximation

    effect_counts = pd.DataFrame()  # Placeholder, non utilisé dans le nouveau code

    return mutation_counts, vaf_stats, effect_counts


def create_advanced_clinical_features(df):
    """Version de compatibilité avec l'ancien code"""
    return create_clinical_features(df)
