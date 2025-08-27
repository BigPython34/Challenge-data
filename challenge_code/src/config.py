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

CREATE_LOG_COLUMNS = False
# Clinical feature engineering configuration (traceable)
CLINICAL_NUMERIC_COLUMNS = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
CLINICAL_RATIOS = {
    # ratio_name: (numerator, denominator)
    "neutrophil_ratio": ("ANC", "WBC"),
    "monocyte_ratio": ("MONOCYTES", "WBC"),
    "platelet_wbc_ratio": ("PLT", "WBC"),
    "blast_platelet_ratio": ("BM_BLAST", "PLT"),
}
CLINICAL_THRESHOLDS = {
    # feature_name: (column, operator, threshold)
    "anemia_moderate": ("HB", "<", 10),
    "anemia_severe": ("HB", "<", 8),
    "thrombocytopenia_moderate": ("PLT", "<", 100),
    "thrombocytopenia_severe": ("PLT", "<", 50),
    "neutropenia_moderate": ("ANC", "<", 1.5),
    "neutropenia_severe": ("ANC", "<", 1.0),
    "leukocytosis_high": ("WBC", ">", 30),
    "high_blast_count": ("BM_BLAST", ">", 20),
}
CLINICAL_LOG_COLUMNS = ["WBC", "PLT", "ANC", "MONOCYTES"]
MISSINGNESS_POLICY = {
    "create_indicators": True,
    # Keep only these indicators by default to limit drift while preserving clinically relevant signals
    "keep_columns": ["WBC", "HB", "PLT", "ANC", "MONOCYTES"],
    "drop_non_kept_indicators": True,
}
# ELN 2022 Cytogenetic Risk Classification
SPECIFIC_ABNORMALITIES_TO_FLAG = {
    # Favorables
    "has_t_8_21": r"t\s*\(\s*8\s*;\s*21\s*\)",
    "has_inv_16_or_t_16_16": r"inv\s*\(\s*16\s*\)|t\s*\(\s*16\s*;\s*16\s*\)",
    "has_t_15_17": r"t\s*\(\s*15\s*;\s*17\s*\)",
    # Intermédiaires
    "has_trisomy_8": r"\+\s*8\b",
    "has_t_9_11": r"t\s*\(\s*9\s*;\s*11\s*\)",
    "has_normal_karyotype": r"^\s*46\s*,\s*X[XY]\b(?!.)",  # Regex pour normal strict
    # Défavorables
    "has_mono_7_or_del_7q": r"-\s*7\b|del\s*\(\s*7\s*\)\s*\(q",
    "has_mono_5_or_del_5q": r"-\s*5\b|del\s*\(\s*5\s*\)\s*\(q",
    "has_abn_17p": r"del\s*\(\s*17\s*\)\s*\(p|i\s*\(\s*17\s*\)\s*\(q|17p-",
    "has_rearr_3q": r"inv\s*\(\s*3\s*\)\s*\(q|t\s*\(\s*3\s*;\s*3\s*\)\s*\(q",
    "has_t_6_9": r"t\s*\(\s*6\s*;\s*9\s*\)",
    "has_t_9_22": r"t\s*\(\s*9\s*;\s*22\s*\)",
}


# Complexity definition and ELN encoding controls
COMPLEX_KARYOTYPE_MIN_ABNORMALITIES = 3
ELN_CYTO_RISK_ENCODING = {
    # encode_as: "ordinal" -> a single numeric with weights; "one_hot" -> 3 binary cols
    "encode_as": "one_hot",
    # Weights used only when encode_as == "ordinal"
    "weights": {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0},
}
# ELN 2022 Risk Classification Genes


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

# Additional frequently mutated genes discovered in dataset but absent from the default config
# Sourced from exploratory analysis (top counts):
DISCOVERED_TOP_MISSING_GENES = [
    # From earlier exploration
    "ETNK1",
    "BRCC3",
    "CTCF",
    "EP300",
    "ZBTB33",
    "GNB1",
    "ASXL2",
    "ARID2",
    "PRPF8",
    "GNAS",
    "U2AF2",
    "KMT2D",
    "CREBBP",
    "NFE2",
    "CSF3R",
    # Added 2025-08-14 from train/test intersection (conservative)
    # Cohesin/related & chromatin modifiers frequently observed in both splits
    "RAD21",
    "SMC1A",
    "SMC3",
    "KDM6A",
    "SUZ12",
    "EED",
    # Signaling
    "STAT3",
]

# --- LISTE FINALE ET COMPLÈTE ---
ALL_IMPORTANT_GENES = list(
    set(
        FAVORABLE_GENES
        + ADVERSE_GENES
        + INTERMEDIATE_GENES
        + RAS_PATHWAY_GENES
        + FIRST_TIER_DISCOVERED_GENES
        + SECOND_TIER_DISCOVERED_GENES
        + DISCOVERED_TOP_MISSING_GENES
    )
)


# Gene pathway definitions for functional analysis
GENE_PATHWAYS = {
    "DNA_methylation": ["DNMT3A", "TET2", "IDH1", "IDH2"],
    "RNA_splicing": ["SF3B1", "SRSF2", "U2AF1", "ZRSR2"],
    "Chromatin_modification": ["ASXL1", "EZH2", "BCOR", "KDM6A", "SUZ12", "EED"],
    "Transcription_factors": ["NPM1", "CEBPA", "RUNX1"],
    "Tumor_suppressor": ["TP53"],
    "Tyrosine_kinase": ["FLT3", "KIT"],
    "RAS_signaling": ["NRAS", "KRAS", "PTPN11"],
    "Cohesin_complex": ["STAG2", "RAD21", "SMC1A", "SMC3"],
    # Optional: JAK/STAT
    "JAK_STAT": ["STAT3"],
}

# Common cyto events discovered (kept separate to avoid altering ELN rules)
CYTOGENETIC_COMMON_MONOSOMIES = [
    r"-7\b",
    r"-5\b",
    r"-18\b",
    r"-17\b",
    r"-20\b",
    r"-16\b",
    r"-13\b",
    r"-21\b",
    r"-12\b",
]
CYTOGENETIC_COMMON_TRISOMIES = [
    r"\+8\b",
    r"\+1\b",
    r"\+11\b",
    r"\+21\b",
    r"\+13\b",
    r"\+19\b",
    r"\+9\b",
    r"\+20\b",
]
CYTOGENETIC_SECONDARY_FLAGS = {
    "has_monosomy_17": r"-\s*17\b",
    "has_trisomy_21": r"\+\s*21\b",
    "has_trisomy_11": r"\+\s*11\b",
    "has_trisomy_13": r"\+\s*13\b",
}
# Toggle for including additional common cyto event features (binary flags)
CYTO_FEATURE_TOGGLES = {
    # Master switch for additional cytogenetic features beyond core ELN rules
    "extended_features": True,
    # Add mono_X / tri_Y columns for common events (also enabled when extended_features is True)
    "include_common_events": True,
    "main_clone_analysis": True,
}

# Frequency-based gene filtering for molecular features
# If enabled, restrict ALL_IMPORTANT_GENES to those observed with at least
# `min_total_count` occurrences in the chosen reference.
MOLECULAR_GENE_FREQ_FILTER = {
    "enabled": True,
    "min_total_count": 5,
    # reference: "reports" uses precomputed counts from data exploration (train+test),
    #            "current" uses counts from the provided MAF (per-run dataset)
    "reference": "reports",
    # default relative paths (from project root)
    "train_counts_path": os.path.join(
        BASE_DIR, "reports", "data_explore", "train", "molecular_gene_counts.csv"
    ),
    "test_counts_path": os.path.join(
        BASE_DIR, "reports", "data_explore", "test", "molecular_gene_counts.csv"
    ),
}

# Molecular feature toggles and thresholds
MOLECULAR_FEATURE_TOGGLES = {
    "pathway_features": {"binary": True, "count": True},
    "burden": True,
}
TP53_HIGH_VAF_THRESHOLD = 0.55
MOLECULAR_VAF_THRESHOLDS = {
    # Per-gene high VAF thresholds (used if present). TP53 falls back to TP53_HIGH_VAF_THRESHOLD.
    "FLT3": 0.25,
    "NPM1": 0.5,
    "CEBPA": 0.5,
    "DNMT3A": 0.5,
    "IDH1": 0.5,
    "IDH2": 0.5,
}
ELN_MOLECULAR_RISK_ENCODING = {
    "encode_as": "one_hot",
    "weights": {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0},
}

# Redundancy policy to prune overlapping or collinear features
REDUNDANCY_POLICY = {
    # Drop *_count when *_altered exists (e.g., Tumor_suppressor_count vs Tumor_suppressor_altered)
    "drop_count_when_binary_exists": False,
    # Also drop *_count when a matching any_* flag exists (e.g., foo_count vs any_foo)
    "drop_count_when_any_exists": False,
    # Drop numeric sex encoding if one-hot is present
    "drop_sex_numeric_if_ohe": True,
    # Prune most *_missing except those explicitly kept in MISSINGNESS_POLICY.keep_columns
    "prune_missingness_indicators": True,
    # Explicit drop list (can be adjusted per-run)
    "explicit_drop": [
        # Examples of redundant pairs observed as 1.0 correlated in reports
        # "Tumor_suppressor_count",  # keep binary altered by default
        # "Cohesin_complex_count",
        # "any_cosmic_is_fusion_gene",
        # "any_cosmic_has_translocation_common",
    ],
}
RARE_EVENT_PRUNING_TRESHOLD = 0.005
# Cap for total number of cyto abnormalities when computing derived features
COMPLEX_ABNORMALITIES_CAP = 12

# Centralized preprocessing controls
PREPROCESSING = {
    # Imputation strategy used inside the global preprocessing pipeline
    # accepted values by get_preprocessing_pipeline: "knn", "iterative", "simple"
    "imputer": "iterative",
    # Detailed KNN parameters
    "knn": {"n_neighbors": 4},
    # Iterative imputer parameters (for documentation/traceability)
    "iterative": {
        "max_iter": 350,
        "estimator": "RandomForest",
        "estimator_n_estimators": 70,
        "random_state": SEED,
    },
    # Supervised MONOCYTES imputer parameters (train-only model)
    # Used by data preparation when EXPERIMENT.use_monocyte_supervised is True
    "monocyte_imputer": {
        "model_path": os.path.join(MODEL_DIR, "monocyte_imputer.joblib"),
        # Predictors taken from clinical features
        "predictors": {
            "num": ["WBC", "ANC", "HB", "PLT", "BM_BLAST"],
            "cat": ["CENTER"],
        },
        # Preprocessing for numeric predictors
        "preprocessing": {
            "num_imputer": "median",  # median or mean
            "num_scaler": "standard",  # standard or robust
        },
        # HistGradientBoostingRegressor hyperparameters
        "regressor": {
            "type": "HistGradientBoostingRegressor",
            "learning_rate": 0.08,
            "max_depth": None,
            "max_iter": 400,
            "l2_regularization": 0.0,
            "random_state": SEED,
        },
        # Post-processing controls
        "clip_to_wbc": True,  # enforce MONOCYTES <= WBC when WBC available
        "winsorize_pct": 99.5,  # upper percentile cap for predictions
    },
    # Quantile clipping applied to continuous features
    "clip_quantiles": {"lower": 0.01, "upper": 0.99},
    # Numeric scaler policy inside the pipeline: "standard" or "robust"
    "numeric_scaler": "robust",
}

# Experiment metadata and toggles
EXPERIMENT = {
    "name": "baseline_supervised_mono",
    # Use supervised MONOCYTES imputation trained on train-only
    "use_monocyte_supervised": True,
    # If True, keeps MONOCYTES_missing indicator; else drop to reduce drift
    "keep_monocyte_indicator": False,
    # Centers differ at inference -> do not include CENTER OHE as features
    "use_center_ohe": False,
    # Downstream model family (for traceability only)
    "model_family": "rsf",
    # Random seed propagated to components
    "random_seed": SEED,
    # Optionally prune highly correlated features after preprocessing
    "prune_feature": False,
    # Correlation threshold used when prune_feature is True
    "prune_feature_threshold": 0.90,
}

# Model parameters
TAU = 7  # For C-index calculation

RSF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "n_jobs": -1,
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

# ----------------------
# External + Molecular FE
# ----------------------
# Controls for using external variant-level scores (e.g., CADD via myvariant.info)
MOLECULAR_EXTERNAL_SCORES = {
    "cadd": {
        "enabled": True,  # Network call; keep off by default for safety
        "snv_only": True,  # Query only SNVs (REF/ALT length == 1)
        "high_threshold": 20.0,  # CADD PHRED >= 20 usually considered deleterious
        # Aggregate patient-level features to compute from variant scores
        "features": ["max", "mean", "high_count"],
        # When running 1_prepare_data, should we proactively fetch scores?
        # Set to False to avoid re-downloading on every run (use existing cache only).
        "prefetch_on_prepare": False,
    },
    # Control MyVariant cache warm-up from 1_prepare_data
    "myvariant": {
        # Set to False to skip contacting the API during prepare; cache will be used as-is.
        "prefetch_on_prepare": False,
    },
}

# How to exploit COSMIC tiers at patient-level
COSMIC_TIER_FEATURES = {
    "enabled": True,  # Safe, no network
    # Count mutated genes per COSMIC tier (tier1_gene_count, tier2_gene_count, ...)
    "counts_by_tier": True,
    # Keep NaN for patients with no COSMIC-tiered genes; add an explicit indicator
    "keep_min_tier_na": True,
    "add_has_cosmic_tier": True,
}

# Driver-like features that match gene role and observed mutation effect
DRIVER_LIKE_FEATURES = {
    "enabled": True,
}

# Cross cytogenetic and molecular genomic location (using COSMIC chr band/arm)
CYTO_MOLECULAR_CROSS = {
    "enabled": True,
    # Arms of interest to cross with canonical adverse deletions/monosomies
    "arms": ["5q", "7q", "17p"],
}
