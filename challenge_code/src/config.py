# Global project configuration
import os

# Seed for reproducibility
SEED = 42

# Relative paths from project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # challenge_code/
DATA_DIR = os.path.join(BASE_DIR, "datas")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Specific data paths
CLINICAL_TRAIN_PATH = os.path.join(DATA_DIR, "X_train", "clinical_train.csv")
CLINICAL_TEST_PATH = os.path.join(DATA_DIR, "X_test", "clinical_test.csv")
MOLECULAR_TRAIN_PATH = os.path.join(DATA_DIR, "X_train", "molecular_train.csv")
MOLECULAR_TEST_PATH = os.path.join(DATA_DIR, "X_test", "molecular_test.csv")
TARGET_TRAIN_PATH = os.path.join(DATA_DIR, "target_train.csv")

# ELN 2022 Cytogenetic Risk Classification
# Based on ELN 2022 recommendations for cytogenetic risk stratification

CYTOGENETIC_FAVORABLE = [
    r"t\(8;21\)",  # t(8;21)(q22;q22.1); RUNX1-RUNX1T1
    r"inv\(16\)",  # inv(16)(p13.1q22)
    r"t\(16;16\)",  # t(16;16)(p13.1;q22); CBFB-MYH11
    r"t\(15;17\)",  # t(15;17)(q24;q21.2); PML-RARA
]

CYTOGENETIC_ADVERSE = [
    r"-5\b|del\(5q\)",  # -5 or del(5q)
    r"-7\b|del\(7q\)",  # -7 or del(7q)
    r"del\(17p\)|17p-",  # del(17p) or i(17q)
    r"inv\(3\)|t\(3;3\)",  # inv(3)(q21q26.2) or t(3;3)(q21;q26.2); GATA2, MECOM
    r"t\(6;9\)",  # t(6;9)(p23;q34.1); DEK-NUP214
    r"t\(9;22\)",  # t(9;22)(q34.1;q11.2); BCR-ABL1
    # Complex karyotype (≥3 unrelated chromosome abnormalities)
]

CYTOGENETIC_INTERMEDIATE = [
    r"^46,X[XY]$",  # Normal karyotype
    r"\+8\b",  # Trisomy 8
    r"t\(9;11\)",  # t(9;11)(p21.3;q23.3); MLLT3-KMT2A
    r"11q23",  # Other KMT2A rearrangements
    # All other cytogenetic abnormalities not classified as favorable or adverse
]

# ELN 2022 Risk Classification Genes
# Based on ELN 2022 recommendations for AML genetic risk stratification

# Favorable prognosis genes
FAVORABLE_GENES = [
    "NPM1",  # NPM1 mutation (without adverse-risk genetic lesions)
    "CEBPA",  # Biallelic CEBPA mutations
]

# Adverse prognosis genes
ADVERSE_GENES = [
    "TP53",  # TP53 mutations
    "ASXL1",  # ASXL1 mutations
    "RUNX1",  # RUNX1 mutations
    "BCOR",  # BCOR mutations
    "EZH2",  # EZH2 mutations
    "SF3B1",  # SF3B1 mutations
    "SRSF2",  # SRSF2 mutations
    "STAG2",  # STAG2 mutations
    "U2AF1",  # U2AF1 mutations
    "ZRSR2",  # ZRSR2 mutations
]

# Intermediate prognosis genes (context-dependent)
INTERMEDIATE_GENES = [
    "FLT3",  # FLT3-ITD (context-dependent on NPM1 status and allelic ratio)
    "DNMT3A",  # DNMT3A mutations
    "TET2",  # TET2 mutations
    "IDH1",  # IDH1 mutations
    "IDH2",  # IDH2 mutations
    "KIT",  # KIT mutations (in CBF-AML)
]

# RAS pathway genes (emerging prognostic relevance)
RAS_PATHWAY_GENES = [
    "NRAS",  # NRAS mutations
    "KRAS",  # KRAS mutations
    "PTPN11",  # PTPN11 mutations
]

# All important genes combined (for comprehensive analysis)
ALL_IMPORTANT_GENES = (
    FAVORABLE_GENES + ADVERSE_GENES + INTERMEDIATE_GENES + RAS_PATHWAY_GENES
)

# Gene pathway definitions for functional analysis
GENE_PATHWAYS = {
    "DNA_methylation": ["DNMT3A", "TET2", "IDH1", "IDH2"],
    "RNA_splicing": ["SF3B1", "SRSF2", "U2AF1", "ZRSR2"],
    "Chromatin_modification": ["ASXL1", "EZH2", "BCOR"],
    "Transcription_factors": ["NPM1", "CEBPA", "RUNX1"],
    "Tumor_suppressor": ["TP53"],
    "Tyrosine_kinase": ["FLT3", "KIT"],
    "RAS_signaling": ["NRAS", "KRAS", "PTPN11"],
    "Cohesin_complex": ["STAG2"],
}

# Model parameters
TAU = 7  # For C-index calculation

# Survival model parameters
COX_PARAMS = {"alpha": 0.2, "n_iter": 10}


RSF_PARAMS = {
    "n_estimators": 300,
    "min_samples_split": 15,
    "min_samples_leaf": 15,
    "max_depth": None,
    "max_features": 0.2,
}


GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.02,
    "max_depth": 3,
    "subsample": 0.7,
    "min_samples_leaf": 30,
    "min_samples_split": 40,
}

"""
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": ,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "min_samples_leaf": 15,
    "min_samples_split": 30,
}"""
# 0,7545788740320561 best


# Parameters for CoxNet (Cox with regularization)
COXNET_PARAMS = {
    "l1_ratio": 0.5,  # Balance between L1 and L2 (0.9 = mainly L1)
    "alphas": None,  # Auto-determination
    "n_alphas": 100,
    "normalize": True,
    "max_iter": 10,
}

# Parameters for Extra Survival Trees
EXTRA_TREES_PARAMS = {
    "n_estimators": 30,
    "min_samples_split": 35,
    "min_samples_leaf": 8,
    "max_depth": 10,
}

# Parameters for Componentwise Gradient Boosting
COMPONENTWISE_GB_PARAMS = {
    "n_estimators": 5,
    "learning_rate": 0.05,
    "subsample": 1.0,
}

# Parameters for PyCox DeepSurv
PYCOX_DEEPSURV_PARAMS = {
    "hidden_layers": [128, 64, 32],  # Deeper network
    "dropout": 0.6,  # More dropout for regularization
    "batch_norm": True,
    "learning_rate": 0.001,  # Lower learning rate
    "batch_size": 128,  # Smaller batch size for more stability
    "epochs": 100,  # Optimized epochs
    "patience": 10,  # Generous patience
    "weight_decay": 0.005,  # Lower weight decay
    "optimizer": "Adam",
    "lr_scheduler": True,  # Learning rate scheduler
    "lr_factor": 0.5,  # LR reduction factor
    "lr_patience": 50,  # Patience for scheduler
}

# Advanced imputation parameters
IMPUTATION_PARAMS = {
    "knn_neighbors": 7,  # Increased for more robustness
    "cluster_n_clusters": 8,  # More clusters for better precision
    "rf_n_estimators": 100,  # More estimators for Random Forest
    "bagging_n_estimators": 1000,  # More estimators for bagging
    "outlier_multiplier": 1.8,  # Less restrictive on outliers
    "iterative_max_iter": 20,  # Number of iterations for iterative imputation
}


clinical_base = [
    "BM_BLAST",  # Bone marrow blast percentage
    "WBC",  # White blood cell count
    "ANC",  # Absolute neutrophil count
    "MONOCYTES",  # Monocyte count
    "HB",  # Hemoglobin level
    "PLT",  # Platelet count
    "AGE",  # Patient age
]

clinical_ratios = [
    "neutrophil_ratio",  # ANC/WBC ratio
    "monocyte_ratio",  # MONOCYTES/WBC ratio
    "platelet_wbc_ratio",  # PLT/WBC ratio
    "blast_platelet_ratio",  # BM_BLAST/PLT ratio
]

clinical_binary = [
    "anemia_severe",  # HB < 8 g/dL
    "thrombocytopenia_severe",  # PLT < 50 K/μL
    "neutropenia_severe",  # ANC < 1.0 K/μL
    "leukocytosis_high",  # WBC > 30 K/μL
    "high_blast_count",  # BM_BLAST > 20%
    "pancytopenia",  # All lineages affected
]

clinical_scores = [
    "cytopenia_score",  # 0-3, number of cytopenias
    "proliferation_score",  # Blast + leukocytosis score
    "clinical_risk_score",  # Composite clinical risk
]

# === CYTOGENETIC FEATURES (ELN 2022) ===
cytogenetic_basic = [
    "chromosome_count",  # Total chromosome number
    "sex_chromosomes",  # XX=0, XY=1, unknown=0.5
    "num_cyto_abnormalities",  # Number of cytogenetic changes
]

cytogenetic_favorable = [
    "t_8_21",  # t(8;21)(q22;q22.1)
    "inv_16",  # inv(16)(p13.1q22)
    "t_16_16",  # t(16;16)(p13.1;q22)
    "t_15_17",  # t(15;17)(q24;q21.2)
    "any_favorable_cyto",  # Any favorable abnormality
]

cytogenetic_intermediate = [
    "normal_karyotype",  # 46,XX/XY
    "trisomy_8",  # +8
    "t_9_11",  # t(9;11)(p21.3;q23.3)
    "kmt2a_rearranged",  # Other 11q23/KMT2A
]

cytogenetic_adverse = [
    "del_5q",  # -5/del(5q)
    "monosomy_7",  # -7/del(7q)
    "del_17p",  # del(17p)/i(17q)
    "abn_3q",  # inv(3)/t(3;3)
    "t_6_9",  # t(6;9)(p23;q34.1)
    "t_9_22",  # t(9;22)(q34.1;q11.2)
    "complex_karyotype",  # ≥3 unrelated abnormalities
    "monosomal_karyotype",  # Monosomal karyotype
    "any_adverse_cyto",  # Any adverse abnormality
]

cytogenetic_risk = [
    "eln_cyto_risk",  # ELN 2022 cytogenetic risk (0-2)
]

# === MOLECULAR FEATURES (ELN 2022) ===
molecular_favorable = [
    "mut_NPM1",  # NPM1 mutations
    "mut_CEBPA",  # CEBPA mutations
    "CEBPA_biallelic",  # Biallelic CEBPA
]

molecular_intermediate = [
    "mut_FLT3",  # FLT3 mutations
    "mut_DNMT3A",  # DNMT3A mutations
    "mut_TET2",  # TET2 mutations
    "mut_IDH1",  # IDH1 mutations
    "mut_IDH2",  # IDH2 mutations
    "mut_KIT",  # KIT mutations
]

molecular_adverse = [
    "mut_TP53",  # TP53 mutations
    "mut_ASXL1",  # ASXL1 mutations
    "mut_RUNX1",  # RUNX1 mutations
    "mut_BCOR",  # BCOR mutations
    "mut_EZH2",  # EZH2 mutations
    "mut_SF3B1",  # SF3B1 mutations
    "mut_SRSF2",  # SRSF2 mutations
    "mut_STAG2",  # STAG2 mutations
    "mut_U2AF1",  # U2AF1 mutations
    "mut_ZRSR2",  # ZRSR2 mutations
]

molecular_ras = [
    "mut_NRAS",  # NRAS mutations
    "mut_KRAS",  # KRAS mutations
    "mut_PTPN11",  # PTPN11 mutations
]

# === MOLECULAR DERIVED FEATURES ===
molecular_vaf = [
    "vaf_max_TP53",
    "TP53_high_VAF",
    "vaf_max_FLT3",
    "FLT3_high_VAF",
    "vaf_max_NPM1",
    "NPM1_high_VAF",
    "vaf_max_CEBPA",
    "CEBPA_high_VAF",
    "vaf_max_DNMT3A",
    "DNMT3A_high_VAF",
]

molecular_types = [
    "TP53_truncating",  # TP53 loss-of-function
    "CEBPA_biallelic",  # Biallelic CEBPA
]

molecular_comutations = [
    "NPM1_pos_FLT3_neg",  # NPM1+/FLT3- (favorable)
    "NPM1_pos_FLT3_low",  # NPM1+/FLT3+ low ratio
    "DNMT3A_NPM1_FLT3",  # Triple mutation
]

# === PATHWAY FEATURES ===
pathway_features = [
    "DNA_methylation_altered",
    "DNA_methylation_count",
    "RNA_splicing_altered",
    "RNA_splicing_count",
    "Chromatin_modification_altered",
    "Chromatin_modification_count",
    "Transcription_factors_altered",
    "Transcription_factors_count",
    "Tumor_suppressor_altered",
    "Tumor_suppressor_count",
    "Tyrosine_kinase_altered",
    "Tyrosine_kinase_count",
    "RAS_signaling_altered",
    "RAS_signaling_count",
    "Cohesin_complex_altered",
    "Cohesin_complex_count",
]

# === MUTATION BURDEN ===
mutation_burden = [
    "total_mutations",  # Total mutation count
    "vaf_mean",
    "vaf_median",
    "vaf_max",
    "vaf_std",  # VAF statistics
    "high_vaf_mutations",  # High VAF mutation count
    "high_vaf_ratio",  # High VAF proportion
    "mutation_burden_score",  # Continuous burden score
    "mutation_burden_category",  # Categorical burden (0-2)
]

# === INTEGRATED RISK SCORES ===
risk_scores = [
    "eln_molecular_risk",  # ELN 2022 molecular risk (0-2)
    "eln_integrated_risk",  # ELN 2022 combined risk (0-2)
    "comprehensive_risk_score",  # Multi-dimensional risk
]

# === INTERACTION FEATURES ===
interaction_features = [
    "TP53_complex_cyto",  # TP53 + complex karyotype
    "FLT3_NPM1_interaction",  # FLT3/NPM1 combinations
    "DNMT3A_age_interaction",  # DNMT3A × age interaction
]

FEATURE_LIST = {
    # Clinical features
    "clinical_base": clinical_base,
    "clinical_ratios": clinical_ratios,
    "clinical_binary": clinical_binary,
    "clinical_scores": clinical_scores,
    # Cytogenetic features (ELN 2022)
    "cytogenetic_basic": cytogenetic_basic,
    "cytogenetic_favorable": cytogenetic_favorable,
    "cytogenetic_intermediate": cytogenetic_intermediate,
    "cytogenetic_adverse": cytogenetic_adverse,
    "cytogenetic_risk": cytogenetic_risk,
    # Molecular features (ELN 2022)
    "molecular_favorable": molecular_favorable,
    "molecular_intermediate": molecular_intermediate,
    "molecular_adverse": molecular_adverse,
    "molecular_ras": molecular_ras,
    "molecular_vaf": molecular_vaf,
    "molecular_types": molecular_types,
    "molecular_comutations": molecular_comutations,
    # Pathway and burden features
    "pathway_features": pathway_features,
    "mutation_burden": mutation_burden,
    # Integrated scores
    "risk_scores": risk_scores,
    "interaction_features": interaction_features,
}


DATA_PREPARATION_CONFIG = {
    "pipeline": {
        "test_size": 0.3,  # Plus grande validation pour plus de robustesse
        "use_advanced_features": True,
        "include_molecular_burden": True,
        "include_cytogenetic_features": True,
        "include_interaction_features": True,
    },
    "imputation": {
        "strategy": "medical_informed",  # medical_informed, median, mean, knn, iterative, regression
        "fill_missing_with_zero": ["mutations", "molecular"],
        "fill_missing_with_median": ["clinical_numeric"],
        "fill_missing_with_mode": ["clinical_categorical"],
    },
    "feature_engineering": {
        "clinical": {
            "create_ratios": True,
            "create_thresholds": True,
            "create_composite_scores": True,
            "create_log_transforms": True,
        },
        "molecular": {
            "extract_binary_mutations": True,
            "extract_vaf_features": True,
            "extract_mutation_types": True,
            "extract_comutation_patterns": True,
            "extract_pathway_alterations": True,
        },
        "cytogenetic": {
            "extract_eln2022_abnormalities": True,
            "calculate_complexity": True,
            "extract_chromosome_features": True,
            "calculate_risk_scores": True,
        },
        "integrated": {
            "create_eln2022_risk_scores": True,
            "create_interaction_features": True,
            "create_comprehensive_scores": True,
        },
    },
    "quality_control": {
        "remove_low_variance_features": True,
        "variance_threshold": 0.01,
        "remove_highly_correlated": True,
        "correlation_threshold": 0.95,
        "handle_outliers": True,
        "outlier_method": "clip",  # clip, remove, transform
    },
    "output": {
        "save_datasets": True,
        "save_metadata": True,
        "save_feature_importance": True,
        "create_visualizations": True,
        "datasets_dir": "datasets",
        "models_dir": "models",
    },
}
