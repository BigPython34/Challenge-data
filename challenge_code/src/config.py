# Global project configuration
import os

# Seed for reproducibility
SEED = 42

# Relative paths from project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # challenge_code/
DATA_DIR = os.path.join(BASE_DIR, "datas")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


CLINICAL_RANGES = {
    "BM_BLAST": (0, 100),
    "WBC": (0, 400),
    "ANC": (0, 100),
    "MONOCYTES": (0, 100),
    "PLT": (0, 3000),
    "HB": (2, 25),
}
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


FIRST_TIER_DISCOVERED_GENES = [
    "CBL",
    "DDX41",
    "CUX1",
    "NF1",
    "PHF6",
    "SETBP1",
    "JAK2",
    "MLL",
    "WT1",
    "ETV6",
    "PPM1D",
]

# Gènes découverts au deuxième tour (pertinence clinique avérée)
SECOND_TIER_DISCOVERED_GENES = ["GATA2", "KMT2C", "BCORL1", "MPL", "SH2B3", "CSNK1A1"]

# --- LISTE FINALE ET COMPLÈTE ---
ALL_IMPORTANT_GENES = list(
    set(
        FAVORABLE_GENES
        + ADVERSE_GENES
        + INTERMEDIATE_GENES
        + RAS_PATHWAY_GENES
        + FIRST_TIER_DISCOVERED_GENES
        + SECOND_TIER_DISCOVERED_GENES
    )
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

RSF_PARAMS = {
    "n_estimators": 400,  # Assez d'arbres pour stabiliser les prédictions.
    "max_depth": 15,  # IMPORTANT: On limite la profondeur pour éviter l'overfitting.
    "min_samples_split": 20,  # Régularisation : ne pas splitter de petits nœuds.
    "min_samples_leaf": 10,  # Régularisation forte : chaque feuille doit être bien peuplée.
    "max_features": "sqrt",  # Standard pour la robustesse, force les arbres à être différents.
    "n_jobs": -1,  # Utilise tous les cœurs du CPU.
}

# --- Gradient Boosting Survival Analysis (LE PLUS PROMETTEUR) ---
# Souvent le plus performant. Les paramètres sont cruciaux.
# Arbres très peu profonds, apprentissage lent, forte régularisation.
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 500,  # Plus d'arbres car le learning_rate est faible.
    "learning_rate": 0.05,  # Un taux d'apprentissage plus faible est plus robuste.
    "max_depth": 3,  # RÈGLE D'OR : Garder les arbres très simples.
    "subsample": 0.7,  # Régularisation forte : chaque arbre ne voit que 70% des données.
    "min_samples_leaf": 20,  # Régularisation forte.
    "min_samples_split": 40,  # Régularisation forte.
    "loss": "coxph",  # La loss par défaut, très performante.
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.7,
    "min_samples_leaf": 20,
    "min_samples_split": 40,
    "loss": "coxph",
}
# --- Cox Proportional Hazards ---
# Notre baseline linéaire. Simple régularisation.
COX_PARAMS = {
    "alpha": 0.1,  # Une petite pénalité Ridge pour la stabilité numérique.
    "n_iter": 100,  # Plus d'itérations pour assurer la convergence.
}

# --- CoxNet (Cox avec régularisation ElasticNet) ---
# TRÈS INTÉRESSANT dans votre cas car il fait de la sélection de features.
COXNET_PARAMS = {
    "l1_ratio": 0.95,  # Privilégier fortement le Lasso (L1) pour mettre à zéro les features inutiles.
    "n_alphas": 100,  # Laisse scikit-survival trouver le meilleur alpha.
    "max_iter": 20000,  # Augmenter significativement pour garantir la convergence.
}

# --- Extra Survival Trees ---
# Alternative au RSF, parfois plus robuste.
EXTRA_TREES_PARAMS = {
    "n_estimators": 400,
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "n_jobs": -1,
}

# --- Component-wise Gradient Boosting ---
# Moins prioritaire, mais peut surprendre.
COMPONENTWISE_GB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "subsample": 0.7,
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
