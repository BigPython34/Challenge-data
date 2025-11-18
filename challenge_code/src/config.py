import os
from pathlib import Path
from types import MappingProxyType

SEED = 42
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datas"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def _build_data_paths(base_dir: Path) -> dict[str, str]:
    data_dir = base_dir / "datas"
    return {
        "input_clinical_train": str(data_dir / "X_train" / "clinical_train.csv"),
        "input_molecular_train": str(data_dir / "X_train" / "molecular_train_filled.csv"),
        "input_target_train": str(data_dir / "target_train.csv"),
        "input_clinical_test": str(data_dir / "X_test" / "clinical_test.csv"),
        "input_molecular_test": str(data_dir / "X_test" / "molecular_test_filled.csv"),
        "output_dir": str(base_dir / "datasets_processed"),
        "oncokb_file": str(data_dir / "external" / "cancerGeneList.txt"),
        "cosmic_file": str(data_dir / "external" / "Cosmic_CancerGeneCensus_v102_GRCh38.tsv"),
    }


DATA_PATHS = _build_data_paths(BASE_DIR)

TARGET_COLUMNS = {
    "status": "OS_STATUS",
    "time": "OS_YEARS",
}

# Clinical configuration
ID_COLUMNS = {
    "patient": "ID",
    "center": "CENTER",
}

CLINICAL_RANGES = {
    "BM_BLAST": (0, 100),
    "WBC": (0, 400),
    "ANC": (0, 100),
    "MONOCYTES": (0, 100),
    "PLT": (0, 3000),
    "HB": (2, 25),
}



CLINICAL_NUMERIC_COLUMNS = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
CLINICAL_RATIOS = {
    "neutrophil_ratio": ("ANC", "WBC"),
    "monocyte_ratio": ("MONOCYTES", "WBC"),
    "platelet_wbc_ratio": ("PLT", "WBC"),
    "blast_platelet_ratio": ("BM_BLAST", "PLT"),
}
CLINICAL_THRESHOLDS = {
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
CREATE_LOG_COLUMNS = False
MISSINGNESS_POLICY = {
    "create_indicators": True,
    "keep_columns": ["WBC", "HB", "PLT", "ANC", "MONOCYTES"],
    "drop_non_kept_indicators": True,
}

CENTER_GROUPING = {
    "enabled": True,
    "rare_center_threshold": 40,
    "other_label": "CENTER_OTHER",
}


# Cytogenetic configuration
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

COMPLEX_KARYOTYPE_MIN_ABNORMALITIES = 3

ELN_CYTO_RISK_ENCODING = {
    "encode_as": "one_hot",
    "weights": {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0},
}

CYTO_FEATURE_TOGGLES = {
    "extended_features": True,
    "include_common_events": True,
    "main_clone_analysis": True,
}

FEATURE_ENGINEERING_TOGGLES = {
    "clinical": True,
    "cytogenetic": True,
    "molecular": True,
    "cyto_molecular_interaction": True,
}

###molecular
FAVORABLE_GENES = [
    "NPM1",  
    "CEBPA",  
]
ADVERSE_GENES = [
    "TP53", 
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
INTERMEDIATE_GENES = [
    "FLT3",  
    "DNMT3A",  
    "TET2",  
    "IDH1",  
    "IDH2",  
    "KIT",  
]


RAS_PATHWAY_GENES = [
    "NRAS",  
    "KRAS",  
    "PTPN11",  
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


SECOND_TIER_DISCOVERED_GENES = ["GATA2", "KMT2C", "BCORL1", "MPL", "SH2B3", "CSNK1A1"]

DISCOVERED_TOP_MISSING_GENES = [
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
    "RAD21",
    "SMC1A",
    "SMC3",
    "KDM6A",
    "SUZ12",
    "EED",
    "STAT3",
]


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


GENE_PATHWAYS = {
    "DNA_methylation": ["DNMT3A", "TET2", "IDH1", "IDH2"],
    "RNA_splicing": ["SF3B1", "SRSF2", "U2AF1", "ZRSR2"],
    "Chromatin_modification": ["ASXL1", "EZH2", "BCOR", "KDM6A", "SUZ12", "EED"],
    "Transcription_factors": ["NPM1", "CEBPA", "RUNX1"],
    "Tumor_suppressor": ["TP53"],
    "Tyrosine_kinase": ["FLT3", "KIT"],
    "RAS_signaling": ["NRAS", "KRAS", "PTPN11"],
    "Cohesin_complex": ["STAG2", "RAD21", "SMC1A", "SMC3"],
    "JAK_STAT": ["STAT3"],
}


MOLECULAR_GENE_FREQ_FILTER = {
    "enabled": True,
    "min_total_count": 5,
    "reference": "reports",
    "train_counts_path": os.path.join(
        BASE_DIR, "reports", "data_explore", "train", "molecular_gene_counts.csv"
    ),
    "test_counts_path": os.path.join(
        BASE_DIR, "reports", "data_explore", "test", "molecular_gene_counts.csv"
    ),
}

MOLECULAR_FEATURE_TOGGLES = {
    "pathway_features": {"binary": True, "count": True},
    "burden": True,
}

MOLECULAR_VAF_THRESHOLDS = {
   "TP53":0.55,
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


###feature eng

REDUNDANCY_POLICY = {
    "drop_count_when_binary_exists": False,
    "drop_count_when_any_exists": False,
    "drop_sex_numeric_if_ohe": True,
    "prune_missingness_indicators": True,
    "explicit_drop": ["CENTER"]
}
RARE_EVENT_PRUNING_TRESHOLD = 0.005
COMPLEX_ABNORMALITIES_CAP = 12


###preprocess
PREPROCESSING = {
    "imputer": "iterative",
    "knn": {"n_neighbors": 4},
    "iterative": {
        "max_iter": 100,
        "estimator": "BayesianRidge",
        "estimator_n_estimators": 250,
        "random_state": SEED,
    },
    "early_imputation": {
        "enabled": False,
        "strategy": "iterative",
        "columns": CLINICAL_NUMERIC_COLUMNS,
        "respect_ranges": True,
        "range_map": CLINICAL_RANGES,
        "artifact_path": os.path.join(MODEL_DIR, "early_continuous_imputer.joblib"),
    },
   
    "monocyte_imputer": {
        "model_path": os.path.join(MODEL_DIR, "monocyte_imputer.joblib"),
        "predictors": {
            "num": ["WBC", "ANC", "HB", "PLT", "BM_BLAST"],
            "cat": ["CENTER"],
        },
        "preprocessing": {
            "num_imputer": "median",  
            "num_scaler": "standard",  
        },
        
        "regressor": {
            "type": "HistGradientBoostingRegressor",
            "learning_rate": 0.08,
            "max_depth": None,
            "max_iter": 400,
            "l2_regularization": 0.0,
            "random_state": SEED,
        },
        "clip_to_wbc": True,  
        "winsorize_pct": 99.5,  
    },
    "clip_quantiles": {"lower": 0.01, "upper": 0.99},
    "numeric_scaler": "robust",
    "drop_zero_variance": True,
    "auto_detect_continuous_features": False,
    "continuous_threshold": 20,
    "continuous_features": CLINICAL_NUMERIC_COLUMNS
    + [
        "neutrophil_ratio",
        "monocyte_ratio",
        "platelet_wbc_ratio",
        "blast_platelet_ratio",
        "vaf_max_TP53",
        "vaf_max_FLT3",
        "vaf_max_NPM1",
        "vaf_max_CEBPA",
        "vaf_max_DNMT3A",
        "vaf_mean",
        "vaf_median",
        "vaf_max",
        "vaf_std",
        "high_vaf_ratio",
        "max_cadd_score",
        "mean_cadd_score",
        "max_gerp_score",
    ],
}

FLOAT32_POLICY = {
    "enabled": True,
    "columns": CLINICAL_NUMERIC_COLUMNS,
    "auto_detect_feature_frames": True,
    "auto_detect_processed_frames": True,
    "protected_columns": [
        ID_COLUMNS["patient"],
        ID_COLUMNS["center"],
        TARGET_COLUMNS["status"],
        TARGET_COLUMNS["time"],
        "CENTER_GROUP",
    ],
}

EXPERIMENT = {
    "name": "baseline_supervised_mono",
    "use_monocyte_supervised": True,
    "keep_monocyte_indicator": False,
    "use_center_ohe": False,
    "model_family": "rsf",
    "random_seed": SEED,
    "prune_feature": False,
    "prune_feature_threshold": 0.90,
}

# Model parameters
TAU = 7  

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

COX_PARAMS = {
    "alpha": 0.1, 
    "n_iter": 100,  
}


COXNET_PARAMS = {
    "l1_ratio": 0.95,  
    "n_alphas": 100, 
    "max_iter": 20000,  
}

EXTRA_TREES_PARAMS = {
    "n_estimators": 400,
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "n_jobs": -1,
}


COMPONENTWISE_GB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "subsample": 0.7,
}



PYCOX_DEEPSURV_PARAMS = {
    "hidden_layers": [128, 64, 32],  
    "dropout": 0.6,  #
    "batch_norm": True,
    "learning_rate": 0.001, 
    "batch_size": 128,  
    "epochs": 100,  
    "patience": 10,  
    "weight_decay": 0.005,  
    "optimizer": "Adam",
    "lr_scheduler": True,  
    "lr_factor": 0.5,  
    "lr_patience": 50,  
}


# External + Molecular FE

MOLECULAR_EXTERNAL_SCORES = {
    "cadd": {
        "enabled": True,  # Network call; keep off by default for safety
        "snv_only": True,  # Query only SNVs (REF/ALT length == 1)
        "high_threshold": 20.0,  # CADD PHRED >= 20 usually considered deleterious
        # Aggregate patient-level features to compute from variant scores
        "features": ["max", "mean", "high_count"],
        # Set to False to avoid re-downloading on every run (use existing cache only).
        "prefetch_on_prepare": False,
    },

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
    "specs": [
        {"arm": "5q", "cyto_col": "del_5q_or_mono5", "out_col": "mut_in_5q_and_del5q"},
        {"arm": "7q", "cyto_col": "monosomy_7_or_del7q", "out_col": "mut_in_7q_and_del7q"},
        {"arm": "17p", "cyto_col": "del_17p_or_i17q", "out_col": "mut_in_17p_and_del17p"},
    ]
}

FEATURE_INTERACTIONS = {
    "enabled": True,
    "cyto_normal_mol_favorable": {
        "enabled": True,
        "base_col": "normal_karyotype",
        "good_mol_cols": ["mut_NPM1", "CEBPA_biallelic"],
        "bad_mol_cols_for_good": ["FLT3_ITD"]
    },
    "cyto_normal_mol_adverse": {
        "enabled": True,
        "base_col": "normal_karyotype",
        "adverse_mol_cols": ["FLT3_ITD"]
    },
    "cyto_favorable_mol_adverse_kit": {
        "enabled": True,
        "base_col": "any_favorable_cyto",
        "adverse_mol_cols": ["mut_KIT"]
    },
    "cyto_complex_and_mol_tp53": {
        "enabled": True,
        "base_col": "complex_karyotype",
        "adverse_mol_cols": ["mut_TP53"]
    }
}

PRUNING_POLICY = {
    "rare_feature_threshold": RARE_EVENT_PRUNING_TRESHOLD,
    "correlation_threshold": 0.95,
    "default_id_cols": ["ID", "CENTER_GROUP"],
    "priority_rules": [
        {"keep": "mut_", "drop": "_altered"},
        {"keep": "mut_", "drop": "_count"},
        {"keep": "log_", "drop": ""}, # drop original if log exists
        {"keep": "mean", "drop": "median"}
    ]
}

CLINICAL_COMPOSITE_SCORES = {
    "cytopenia_score": {
        "components": [
            "anemia_moderate",
            "thrombocytopenia_moderate",
            "neutropenia_moderate",
        ],
        "output_col": "cytopenia_score",
    },
    "pancytopenia": {
        "score_col": "cytopenia_score", 
        "threshold": 3,
        "output_col": "pancytopenia"
    },
    "proliferation_score": {
        "components": ["high_blast_count", "leukocytosis_high"],
        "output_col": "proliferation_score",
    },
}

CYTOGENETIC_EVENT_PATTERNS = {
    "n_t": r"t\(",
    "n_del": r"del\(|del,",
    "n_inv": r"inv\(",
    "n_add": r"add\(",
    "n_der": r"der\(",
    "n_ins": r"ins\(",
    "n_i": r"i\(",
    "n_dic": r"dic\(",
    "n_ring": r"r\(",
    "n_mar": r"\+mar",
    "n_dmin": r"dmin",
    "n_plus": r"\+",
    "n_minus": r"-",
}

CYTOGENETIC_PARSING_REGEX = {
    "normal_string": r"normal",
    "complex_fallback": r">=3|>3|complex|multiple abnormalities",
    "clone_split": r"\s*/\s*|\s{2,}",
    "cell_count": r"\[\s*(?:cp)?\s*(\d+)(?:/\d+)?\s*\]$",
    "idem": r"idem",
    "event_text_prefix": r"^\s*\d{1,2}(?:-\d{1,2})?\s*,\s*X[XY]*",
    "chromosome_count": r"^\s*(\d+)",
}


CYTOGENETIC_PATTERNS = {
    "t(9;22)": r"t\s*\(\s*9\s*;\s*22\s*\)",
    "t(15;17)": r"t\s*\(\s*15\s*;\s*17\s*\)",
    "inv(16)": r"inv\s*\(\s*16\s*\)|t\s*\(\s*16\s*;\s*16\s*\)",
    "t(8;21)": r"t\s*\(\s*8\s*;\s*21\s*\)",
    "del5q_or_mono5": r"del\s*\(\s*5\s*\)\s*\(q|-\s*5\b",
    "del7q_or_mono7": r"del\s*\(\s*7\s*\)\s*\(q|-\s*7\b",
    "del17p": r"del\s*\(\s*17\s*\)\s*\(p",
    "del20q": r"del\s*\(\s*20\s*\)\s*\(q",
    "+8": r"\+\s*8\b",
    "idic(X)(q13)": r"idic\s*\(\s*X\s*\)\s*\(q13\)",
}

MOLECULAR_INPUT_COLUMNS = {
    "gene": "GENE",
    "vaf": "VAF",
    "effect": "EFFECT",
    "protein_change": "PROTEIN_CHANGE",
    "impact": "IMPACT",
    "hugo_symbol": "Hugo_Symbol",
}

MUTATION_TYPE_PATTERNS = {
    "TP53_truncating": {
        "gene": "TP53",
        "type": "pattern_match",
        "on_column": "effect",
        "pattern": r"nonsense|frameshift|splice_site|stop_gained",
    },
    "CEBPA_biallelic": {
        "gene": "CEBPA",
        "type": "count",
        "threshold": 2,
    },
    "FLT3_ITD": {
        "gene": "FLT3",
        "type": "pattern_match",
        "on_column": "effect",
        "pattern": r"ITD|internal tandem duplication",
    },
    "FLT3_TKD": {
        "gene": "FLT3",
        "type": "pattern_match",
        "on_column": "protein_change",
        "pattern": r"D835|I836",
    },
}

COMUTATION_PATTERNS = {
    "NPM1_pos_FLT3_neg": {
        "type": "co_occurrence",
        "genes": ["NPM1", "FLT3"],
        "status": [1, 0],
    },
    "double_hit_spliceosome": {
        "type": "multi_hit",
        "pathway": "RNA_splicing",
        "min_hits": 2,
    },
    "triple_hit_epigenetic": {
        "type": "multi_hit",
        "pathway": "DNA_methylation",
        "min_hits": 3,
    },
    "vaf_ratio_FLT3_NPM1": {
        "type": "vaf_ratio",
        "numerator": "FLT3",
        "denominator": "NPM1",
    },
}


MODELING = {
    "seed": SEED,
    "tau": 7,  # Time horizon for survival analysis
    "cv_folds": 4,
    "cv_repeats": 2,
    "models": {
        "RSF": {
            "enabled": True,
            "params": {
                "n_estimators": 400,
                "max_depth": 15,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
        },
        "GradientBoosting": {
            "enabled": True,
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.7,
                "min_samples_leaf": 20,
                "min_samples_split": 40,
                "loss": "coxph",
            },
        },
        "Cox": {"enabled": True, "params": {"alpha": 0.1, "n_iter": 100}},
        "CoxNet": {
            "enabled": True,
            "params": {"l1_ratio": 0.95, "n_alphas": 100, "max_iter": 20000},
        },
        "ExtraTrees": {
            "enabled": True,
            "params": {
                "n_estimators": 400,
                "max_depth": 15,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
        },
        "ComponentwiseGB": {
            "enabled": True,
            "params": {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "subsample": 0.7,
            },
        },
        "DeepSurv": {
            "enabled": False,  # Disabled by default as it requires special handling
            "params": {
                "hidden_layers": [128, 64, 32],
                "dropout": 0.6,
                "batch_norm": True,
                "learning_rate": 0.001,
                "batch_size": 128,
                "epochs": 100,
                "patience": 10,
                "weight_decay": 0.005,
                "optimizer": "Adam",
                "lr_scheduler": True,
                "lr_factor": 0.5,
                "lr_patience": 50,
            },
        },
    },
    "hyper_params_grids": {
        "RSF": {
            "n_estimators": [200, 400, 600],
            "max_depth": [10, 15, 20],
            "min_samples_leaf": [10, 20, 30],
        },
        "GradientBoosting": {
            "n_estimators": [300, 500, 700],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        },
        "CoxNet": {
            "l1_ratio": [0.5, 0.7, 0.9, 0.95, 1.0],
        },
        # Add other models as needed
    },
    "ensemble": {
        "max_models_in_ensemble": 3,
        "optimization_metric": "c_index", # or "brier_score"
        "score_time_point": 5, # for brier score
    },
    "prediction": {
        "ensemble_meta_path": os.path.join(MODEL_DIR, "ensemble_meta.json"),
        "preprocessor_path": os.path.join(MODEL_DIR, "preprocessor.joblib"),
    }
}