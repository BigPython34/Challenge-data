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
COX_PARAMS = {
    "alpha": 1.0,
}

RSF_PARAMS = {
    "n_estimators": 200,  # Augmentation pour plus de stabilité
    "min_samples_split": 20,  # Augmentation pour plus de régularisation
    "min_samples_leaf": 10,  # Augmentation pour plus de régularisation
    "max_depth": 10,  # Légère augmentation mais toujours contrôlé
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 1000,  # Réduction de 850 à 300
    "learning_rate": 0.013235192937383306,  # Augmentation pour plus de stabilité
    "max_depth": 6,
    "subsample": 0.4117261899723177,  # Sous-échantillonnage pour réduire l'overfitting
    "min_samples_leaf": 37,  # Augmentation pour plus de régularisation
    "min_samples_split": 160,  # Augmentation pour plus de régularisation
}
"""
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 300,  # Réduction de 850 à 300
    "learning_rate": 0.1,  # Augmentation pour plus de stabilité
    "max_depth": 3,
    "subsample": 0.8,  # Sous-échantillonnage pour réduire l'overfitting
    "min_samples_leaf": 15,  # Augmentation pour plus de régularisation
    "min_samples_split": 30,  # Augmentation pour plus de régularisation
}
0,7545788740320561 best
"""
# Parameters for CoxNet (Cox with regularization)
COXNET_PARAMS = {
    "l1_ratio": 0.9,  # Balance between L1 and L2 (0.9 = mainly L1)
    "alphas": None,  # Auto-determination
    "n_alphas": 100,
    "normalize": True,
    "max_iter": 100000,
}

# Parameters for Extra Survival Trees
EXTRA_TREES_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_depth": 10,
    "max_features": "sqrt",
}

# Parameters for Componentwise Gradient Boosting
COMPONENTWISE_GB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
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
