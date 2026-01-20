"""
Configuration centralisée pour le pipeline AML survival analysis.

Organisation du fichier:
========================
 1. PATHS & SEEDS           - Chemins, seed, répertoires
 2. DATA SOURCES            - Fichiers d'entrée, Beat AML fusion
 3. CLINICAL CONFIG         - Colonnes cliniques, ranges, ratios, seuils
 4. FEATURE LISTS           - CORE_FEATURES, EXPLORATORY_FEATURES
 5. CYTOGENETIC CONFIG      - Patterns, flags, parsing, ELN
 6. MOLECULAR CONFIG        - Gènes, pathways, VAF, external scores
 7. FEATURE ENGINEERING     - Toggles, interactions, redundancy, pruning
 8. PREPROCESSING           - Imputation (early + pipeline), scaling
 9. EXPERIMENT              - Flags expérimentaux
10. MODELING                - Modèles, hyperparams, stacking, ensemble

Tips de navigation:
-------------------
- Chercher "# ===" pour trouver les en-têtes de section
- Les dictionnaires principaux: PREPROCESSING, MODELING, FEATURE_SET_POLICY
- Pour l'imputation: voir PREPROCESSING["early_imputation"] et PREPROCESSING["imputer"]
"""
import os
from pathlib import Path

# =============================================================================
# 1. PATHS & SEEDS
# =============================================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datas"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def _build_data_paths(base_dir: Path) -> dict[str, str]:
    """Construit les chemins vers les fichiers de données."""
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
        "clinvar_vcf": str(data_dir / "external" / "clinvar.vcf"),
    }


DATA_PATHS = _build_data_paths(BASE_DIR)

# =============================================================================
# 2. DATA SOURCES
# =============================================================================

BEAT_AML_PATHS = {
    "clinical": str(BASE_DIR / "datasets_processed" / "clinical_beat_aml.csv"),
    "molecular": str(BASE_DIR / "datasets_processed" / "molecular_beat_aml.csv"),
    "target": str(BASE_DIR / "datasets_processed" / "target_beat_aml.csv"),
}

TCGA_PATHS = {
    "clinical": str(BASE_DIR / "datasets_processed" / "clinical_tcga.csv"),
    "molecular": str(BASE_DIR / "datasets_processed" / "molecular_tcga.csv"),
    "target": str(BASE_DIR / "datasets_processed" / "target_tcga.csv"),
}

DATA_FUSION = {
    "beat_aml": {
        "use_for_training": True,     # Ajouter Beat AML au jeu d'entraînement
        "use_for_imputation": False,   # Utiliser Beat AML uniquement pour ajuster les imputeurs
    },
    "tcga": {
        "use_for_training": True,    # Ajouter TCGA au jeu d'entraînement
        "use_for_imputation": False,  # Utiliser TCGA uniquement pour ajuster les imputeurs
    }
}

TARGET_COLUMNS = {
    "status": "OS_STATUS",
    "time": "OS_YEARS",
}

ID_COLUMNS = {
    "patient": "ID",
    "center": "CENTER",
}

# =============================================================================
# 3. CLINICAL CONFIG
# =============================================================================

CLINICAL_NUMERIC_COLUMNS = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

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
CLINICAL_RATIO_COLUMNS = list(CLINICAL_RATIOS.keys())

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

CLINICAL_COMPOSITE_SCORES = {
    "cytopenia_score": {
        "components": ["anemia_moderate", "thrombocytopenia_moderate", "neutropenia_moderate"],
        "output_col": "cytopenia_score",
    },
    "pancytopenia": {
        "score_col": "cytopenia_score",
        "threshold": 3,
        "output_col": "pancytopenia",
    },
    "proliferation_score": {
        "components": ["high_blast_count", "leukocytosis_high"],
        "output_col": "proliferation_score",
    },
}

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

# =============================================================================
# 4. FEATURE LISTS
# =============================================================================

CORE_FEATURES = [
    # Démographiques & Cliniques fondamentales
    "SEX_XY", "SEX_UNKNOWN",
    "BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT",
    "clinical_all_missing",
    # Cytogénétique haut niveau
    "cyto_is_missing", "complex_karyotype",
    "eln_cyto_favorable", "eln_cyto_intermediate", "eln_cyto_adverse",
    # Mutations majeures (top 15 gènes pronostiques)
    "mut_TP53", "mut_NPM1", "mut_FLT3", "mut_ASXL1", "mut_RUNX1",
    "mut_DNMT3A", "mut_IDH1", "mut_IDH2", "mut_TET2",
    "mut_SF3B1", "mut_SRSF2", "mut_U2AF1", "mut_ZRSR2",
    "mut_STAG2", "mut_EZH2",
    # Moléculaire composite
    "total_mutations", "num_oncogene_muts", "num_tsg_muts",
    "impact_HIGH", "impact_MODERATE",
    "vaf_max", "high_vaf_ratio", "TP53_high_VAF", "FLT3_ITD",
    "eln_mol_favorable", "eln_mol_intermediate", "eln_mol_adverse",
    # Interactions clés
    "cyto_normal_mol_adverse", "cyto_complex_and_mol_tp53", "NPM1_pos_FLT3_neg",
    "lucky_high_burden_survivor", "unlucky_low_burden_aggressive",
]

EXPLORATORY_FEATURES = list(set(CORE_FEATURES + [
    # Ratios cliniques
    "neutrophil_ratio", "monocyte_ratio", "platelet_wbc_ratio", "blast_platelet_ratio",
    # Seuils cliniques
    "anemia_severe", "thrombocytopenia_severe", "neutropenia_severe", "leukocytosis_high",
    # Cytogénétique détaillée
    "monosomal_karyotype", "num_cyto_abnormalities",
    "structural_count", "numerical_count",
    "has_derivative_chromosome", "has_marker_chromosome",
    # Mutations secondaires
    "mut_BCOR", "mut_BCORL1", "mut_CBL", "mut_CEBPA", "mut_CREBBP",
    "mut_CUX1", "mut_DDX41", "mut_GATA2", "mut_IDH1", "mut_IDH2", "mut_KIT",
    "mut_KRAS", "mut_NF1", "mut_NRAS", "mut_SETBP1",
    # Comptages pathways
    "DNA_methylation_count", "RNA_splicing_count", "Chromatin_modification_count",
    "Transcription_factors_count", "Tumor_suppressor_count", "Tyrosine_kinase_count",
    "RAS_signaling_count", "Cohesin_complex_count",
    # VAF détaillées
    "vaf_mean", "vaf_std", "high_vaf_mutations",
    # Interactions supplémentaires
    "double_hit_spliceosome", "triple_hit_epigenetic",
]))

FEATURE_SET_POLICY = {
    "mode": "default",   # "default" = toutes, "core" = CORE_FEATURES, "exploratory" = EXPLORATORY_FEATURES
    "warn_on_missing": True,
}

# =============================================================================
# 5. CYTOGENETIC CONFIG
# =============================================================================
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
    "monosomy_X": r"-\s*x(?![a-z0-9])|\bmono(?:somy)?\s*x",
    "trisomy_X": r"\+\s*x(?![a-z0-9])|\btris(?:omy|omia)?\s*x",
    "trisomy_Y": r"\+\s*y(?![a-z0-9])|\btris(?:omy|omia)?\s*y",
}

CYTOGENETIC_COMMON_MONOSOMIES = [r"-7\b", r"-5\b", r"-18\b", r"-17\b", r"-20\b", r"-16\b", r"-13\b", r"-21\b", r"-12\b"]
CYTOGENETIC_COMMON_TRISOMIES = [r"\+8\b", r"\+1\b", r"\+11\b", r"\+21\b", r"\+13\b", r"\+19\b", r"\+9\b", r"\+20\b"]

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

CYTOGENETIC_EVENT_PATTERNS = {
    "n_t": r"t\(",
    "n_del": r"del\(|del,|\bdel\b",
    "n_inv": r"inv\(",
    "n_add": r"add\(|\badd\b",
    "n_der": r"der\(",
    "n_ins": r"ins\(",
    "n_i": r"i\(",
    "n_dic": r"dic\(",
    "n_ring": r"(?:r\(|\+r[a-z0-9]*(?:\[[^\]]+\])?)",
    "n_mar": r"[+-]\s*m(?:ar)?[a-z0-9]*(?:\[[^\]]+\])?",
    "n_dmin": r"dmin",
    "n_plus": r"\+",
    "n_minus": r"-",
    "chromosome_range_flag": r"\b\d{2}\s*[-~]\s*\d{2}\b",
}

CYTOGENETIC_NORMALIZATION_RULES = [
    {"pattern": r"order\s*(?=\()", "replacement": "der"},
    {"pattern": r"ordel\s*(?=\()", "replacement": "del"},
    {"pattern": r"\b9ph\b", "replacement": "t(9;22)"},
    {"pattern": r"\bph\s*\+\b", "replacement": "t(9;22)"},
    {"pattern": r"(?i)([+-]\s*[xy])\s*\[[0-9]+\]", "replacement": r"\1"},
    {"pattern": r"(?i)([+-]\s*m(?:ar)?[a-z0-9]*)\s*\[[0-9]+\]", "replacement": r"\1"},
    {"pattern": r"(?i)([+-]\s*r[a-z0-9]*)\s*\[[0-9]+\]", "replacement": r"\1"},
]

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

# =============================================================================
# 6. MOLECULAR CONFIG
# =============================================================================

# Gènes par catégorie de risque ELN
FAVORABLE_GENES = ["NPM1", "CEBPA"]
ADVERSE_GENES = ["TP53", "ASXL1", "RUNX1", "BCOR", "EZH2", "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2"]
INTERMEDIATE_GENES = ["FLT3", "DNMT3A", "TET2", "IDH1", "IDH2", "KIT"]
RAS_PATHWAY_GENES = ["NRAS", "KRAS", "PTPN11"]

# Gènes découverts par analyse / complément
FIRST_TIER_DISCOVERED_GENES = ["CBL", "DDX41", "CUX1", "NF1", "PHF6", "SETBP1", "JAK2", "MLL", "WT1", "ETV6", "PPM1D"]
SECOND_TIER_DISCOVERED_GENES = ["GATA2", "KMT2C", "BCORL1", "MPL", "SH2B3", "CSNK1A1"]
DISCOVERED_TOP_MISSING_GENES = [
    "ETNK1", "BRCC3", "CTCF", "EP300", "ZBTB33", "GNB1", "ASXL2", "ARID2", "PRPF8", "GNAS",
    "U2AF2", "KMT2D", "CREBBP", "NFE2", "CSF3R", "RAD21", "SMC1A", "SMC3", "KDM6A", "SUZ12", "EED", "STAT3",
]

ALL_IMPORTANT_GENES = sorted(set(
    FAVORABLE_GENES + ADVERSE_GENES + INTERMEDIATE_GENES + RAS_PATHWAY_GENES +
    FIRST_TIER_DISCOVERED_GENES + SECOND_TIER_DISCOVERED_GENES + DISCOVERED_TOP_MISSING_GENES
))

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
    "train_counts_path": os.path.join(BASE_DIR, "reports", "data_explore", "train", "molecular_gene_counts.csv"),
    "test_counts_path": os.path.join(BASE_DIR, "reports", "data_explore", "test", "molecular_gene_counts.csv"),
}

MOLECULAR_FEATURE_TOGGLES = {
    "pathway_features": {"binary": True, "count": True},
    "burden": True,
}

MOLECULAR_VAF_THRESHOLDS = {
    "TP53": 0.55, "FLT3": 0.25, "NPM1": 0.5, "CEBPA": 0.5, "DNMT3A": 0.5, "IDH1": 0.5, "IDH2": 0.5,
}

ELN_MOLECULAR_RISK_ENCODING = {
    "encode_as": "one_hot",
    "weights": {"favorable": 0.0, "intermediate": 0.7, "adverse": 1.0},
}

MOLECULAR_INPUT_COLUMNS = {
    "gene": "GENE", "vaf": "VAF", "effect": "EFFECT",
    "protein_change": "PROTEIN_CHANGE", "impact": "IMPACT", "hugo_symbol": "Hugo_Symbol",
}

MUTATION_TYPE_PATTERNS = {
    "TP53_truncating": {"gene": "TP53", "type": "pattern_match", "on_column": "effect", "pattern": r"nonsense|frameshift|splice_site|stop_gained"},
    "CEBPA_biallelic": {"gene": "CEBPA", "type": "count", "threshold": 2},
    "FLT3_ITD": {"gene": "FLT3", "type": "pattern_match", "on_column": "effect", "pattern": r"ITD|internal tandem duplication"},
    "FLT3_TKD": {"gene": "FLT3", "type": "pattern_match", "on_column": "protein_change", "pattern": r"D835|I836"},
}

COMUTATION_PATTERNS = {
    "NPM1_pos_FLT3_neg": {"type": "co_occurrence", "genes": ["NPM1", "FLT3"], "status": [1, 0]},
    "double_hit_spliceosome": {"type": "multi_hit", "pathway": "RNA_splicing", "min_hits": 2},
    "triple_hit_epigenetic": {"type": "multi_hit", "pathway": "DNA_methylation", "min_hits": 3},
    "vaf_ratio_FLT3_NPM1": {"type": "vaf_ratio", "numerator": "FLT3", "denominator": "NPM1"},
}

# --- Scores externes (CADD, ClinVar, COSMIC) ---
MOLECULAR_EXTERNAL_SCORES = {
    "cadd": {
        "enabled": True, "snv_only": True, "high_threshold": 20.0,
        "features": ["max", "mean", "high_count"], "prefetch_on_prepare": False,
    },
    "myvariant": {"prefetch_on_prepare": False},
    "clinvar": {
        "enabled": True,
        "vcf_path": DATA_PATHS.get("clinvar_vcf"),
        "category_patterns": {
            "pathogenic": ["Pathogenic"], "likely_pathogenic": ["Likely_pathogenic"],
            "benign": ["Benign"], "uncertain": ["Uncertain"], "conflicting": ["Conflicting"],
        },
    },
}

COSMIC_TIER_FEATURES = {
    "enabled": True, "counts_by_tier": True, "keep_min_tier_na": True, "add_has_cosmic_tier": True,
}

DRIVER_LIKE_FEATURES = {"enabled": True}

CYTO_MOLECULAR_CROSS = {
    "enabled": True,
    "arms": ["5q", "7q", "17p"],
    "specs": [
        {"arm": "5q", "cyto_col": "del_5q_or_mono5", "out_col": "mut_in_5q_and_del5q"},
        {"arm": "7q", "cyto_col": "monosomy_7_or_del7q", "out_col": "mut_in_7q_and_del7q"},
        {"arm": "17p", "cyto_col": "del_17p_or_i17q", "out_col": "mut_in_17p_and_del17p"},
    ],
}

# =============================================================================
# 7. FEATURE ENGINEERING
# =============================================================================

FEATURE_ENGINEERING_TOGGLES = {
    "clinical": True,
    "cytogenetic": True,
    "molecular": True,
    "cyto_molecular_interaction": True,
}

FEATURE_INTERACTIONS = {
    "enabled": True,
    "cyto_normal_mol_favorable": {
        "enabled": True,
        "base_col": "normal_karyotype",
        "good_mol_cols": ["mut_NPM1", "CEBPA_biallelic"],
        "bad_mol_cols_for_good": ["FLT3_ITD"],
    },
    "cyto_normal_mol_adverse": {
        "enabled": True,
        "base_col": "normal_karyotype",
        "adverse_mol_cols": ["FLT3_ITD"],
    },
    "cyto_favorable_mol_adverse_kit": {
        "enabled": True,
        "base_col": "any_favorable_cyto",
        "adverse_mol_cols": ["mut_KIT"],
    },
    "cyto_complex_and_mol_tp53": {
        "enabled": True,
        "base_col": "complex_karyotype",
        "adverse_mol_cols": ["mut_TP53"],
    },
    # --- NOUVELLES INTERACTIONS (Error Analysis) ---
    "lucky_high_burden_survivor": {
        "enabled": False,
        "base_col": "mut_RUNX1",
        "other_cols": ["mut_ASXL1", "mut_STAG2"],
        "mode": "any", # RUNX1 AND (ASXL1 OR STAG2)
    },
    "unlucky_low_burden_aggressive": {
        "enabled": False,
        "base_col": "high_blast_count",
        "other_cols": ["thrombocytopenia_severe"],
        "mode": "and", # High Blast AND Low Platelets
    },
}

REDUNDANCY_POLICY = {
    "drop_count_when_binary_exists": False,
    "drop_count_when_any_exists": False,
    "drop_sex_numeric_if_ohe": True,
    "prune_missingness_indicators": False,
    "explicit_drop": [
        "cosmic_has_germline_evidence_count", "any_cosmic_has_germline_evidence",
        "molgen_dominant_count", "molgen_recessive_count", "any_molgen_dominant", "any_molgen_recessive",
        "chromosome_range_detected", "chromosome_count_min", "chromosome_count_max", "chromosome_range_span",  "n_ins", "n_i", "n_dic", "n_ring",
        "main_clone_cell_count", "total_cell_count", "incomplete_karyotype",
    ],
}

RARE_EVENT_PRUNING_THRESHOLD = 0.005
COMPLEX_ABNORMALITIES_CAP = 12

PRUNING_POLICY = {
    "rare_feature_threshold": RARE_EVENT_PRUNING_THRESHOLD,
    "correlation_threshold": 0.96,
    "default_id_cols": ["ID", "CENTER_GROUP"],
    "correlation_protected_features": [],
    "correlation_ignored_prefixes": ["CENTER_"],
    "priority_rules": [
        {"keep": "mut_", "drop": "_altered"},
        {"keep": "mut_", "drop": "_count"},
        {"keep": "log_", "drop": ""},
        {"keep": "mean", "drop": "median"},
    ],
    "rare_binary_protected_features": [
        "CEBPA_high_VAF", "IDH2_high_VAF", "FLT3_TKD", "FLT3_ITD",
        "triple_hit_epigenetic", "cyto_normal_mol_adverse", "MONOCYTES_missing",
    ],
    "rare_binary_protected_prefixes": [],
    "rare_feature_aggregations": [
        {
            "output_col": "has_rare_cyto_event",
            "features": ["incomplete_karyotype", "n_ins", "near_triploidy", "t_9_11"],
            "mode": "any",
        },
    ],
}

# =============================================================================
# 8. PREPROCESSING
# =============================================================================

PREPROCESSING = {
    # --- Mise à l'échelle de la cible (OS_YEARS) ---
    "target_time_multiplier": 1,  # Multiplicateur pour OS_YEARS (ex: 1.0 = aucune modif, 12.0 = conversion en mois)

    # --- Stratégie d'imputation globale ---
    "imputer": "iterative",   # "iterative", "knn", "simple"
    "knn": {"n_neighbors": 4},
    "iterative": {
        "max_iter": 100,
        "estimator": "BayesianRidge",
        "estimator_n_estimators": 250,
        "random_state": SEED,
    },

    # --- Imputation précoce (AVANT feature engineering) ---
    # Permet de remplir les colonnes cliniques brutes pour que les ratios soient calculables.
    #
    # NOUVEAU FLOW (1_prepare_data_v2.py):
    # 1. Créer les colonnes auxiliaires basiques (mut_*, eln_cyto_*) via auxiliary_features.py
    # 2. Utiliser ces colonnes pour guider l'IterativeImputer (use_auxiliary_columns=True)
    # 3. Imputer les colonnes cliniques (WBC, HB, PLT, etc.)
    # 4. Feature Engineering complet
    #
    # Les colonnes auxiliaires permettent à l'imputer d'apprendre des corrélations comme:
    # - mut_FLT3 → WBC élevé
    # - eln_cyto_adverse → HB bas
    "early_imputation": {
        "enabled": False,
        "strategy": "iterative",
        "columns": CLINICAL_NUMERIC_COLUMNS,
        "use_auxiliary_columns": True,  # Utiliser mut_*, eln_cyto_* pour guider l'imputation
        "respect_ranges": True,
        "range_map": CLINICAL_RANGES,
        "artifact_path": os.path.join(MODEL_DIR, "early_continuous_imputer.joblib"),
    },

    # --- Mode imputation unique ---
    # Si True ET early_imputation.enabled, la pipeline finale ne ré-impute PAS les colonnes continues.
    # Évite la double imputation (gaspillage de calcul + potentielle altération des valeurs).
    "single_imputation_mode": False,

    # --- Colonnes continues pour la pipeline finale ---
    "continuous_features": CLINICAL_NUMERIC_COLUMNS + CLINICAL_RATIO_COLUMNS,
    # --- Scope de fit des imputeurs ---
    "imputer_fit_scope": {
        "include_test_rows": True,   # Ajouter test dans le fit (pas de fuite de y)
    },

    # --- Features auxiliaires pour guider l'imputation ---
    # Ces colonnes sont utilisées de deux façons possibles:
    # 1. MODE EARLY (recommandé): Si early_imputation.use_auxiliary_columns=True,
    #    ces colonnes enrichissent la matrice de fit de l'IterativeImputer AVANT le FE.
    # 2. MODE LATE: Si early_imputation désactivé ou use_auxiliary_columns=False,
    #    ces colonnes sont injectées APRÈS le FE avec préfixe __aux_impute__ pour la pipeline finale.
    "imputer_auxiliary_features": {
        "enabled": True,
        "prefix": "__aux_impute__",
        "columns": [
            "eln_cyto_favorable", "eln_cyto_adverse",
            "mut_TP53", "mut_NPM1", "mut_FLT3", "mut_ASXL1", "mut_RUNX1",
            "RNA_splicing_altered", "is_clinical_profile_imputed",
        ],
    },

    # --- Monocytes (imputation dédiée optionnelle) ---
    "monocyte_mode": "separate",   # "separate" = imputeur supervisé, "joint" = pipeline globale
    "monocyte_imputer": {
        "model_path": os.path.join(MODEL_DIR, "monocyte_imputer.joblib"),
        "predictors": {"num": ["WBC", "ANC", "HB", "PLT", "BM_BLAST"], "cat": []},
        "preprocessing": {"num_imputer": "median", "num_scaler": "standard"},
        "regressor": {
            "type": "HistGradientBoostingRegressor",
            "learning_rate": 0.08, "max_depth": None, "max_iter": 400,
            "l2_regularization": 0.0, "random_state": SEED,
        },
        "clip_to_wbc": True,
        "winsorize_pct": 99.5,
    },

    # --- Transformations numériques ---
    "clip_quantiles": {"lower": 0.01, "upper": 0.99},
    "numeric_scaler": "robust",

    # --- Variance nulle ---
    "drop_zero_variance": True,
    "zero_variance_protected_columns": ["MONOCYTES_missing"],
    "zero_variance_protected_prefixes": ["CENTER_"],

    # --- Détection automatique des colonnes continues ---
    "auto_detect_continuous_features": False,
    "continuous_threshold": 20,

    # --- Helpers d'imputation (obsolète, désactivé par défaut) ---
    "imputation_helpers": {
        "enabled": False,
        "cyto_flags": ["eln_cyto_favorable", "eln_cyto_adverse"],
        "mutation_genes": ["TP53", "NPM1", "FLT3", "ASXL1", "RUNX1"],
        "pathway_flags": {"RNA_splicing_altered": "RNA_splicing"},
        "clinical_profile_indicator": {
            "enabled": True, "output_col": "is_clinical_profile_imputed", "columns": CLINICAL_NUMERIC_COLUMNS,
        },
        "fill_value": 0.0,
        "drop_after_imputation": True,
    },
}

FLOAT32_POLICY = {
    "enabled": True,
    "columns": CLINICAL_NUMERIC_COLUMNS,
    "auto_detect_feature_frames": True,
    "auto_detect_processed_frames": True,
    "protected_columns": [
        ID_COLUMNS["patient"], ID_COLUMNS["center"],
        TARGET_COLUMNS["status"], TARGET_COLUMNS["time"], "CENTER_GROUP",
    ],
}

# =============================================================================
# 9. EXPERIMENT
# =============================================================================

EXPERIMENT = {
    "name": "baseline",
    "use_monocyte_supervised": False,
    "keep_monocyte_indicator": True,
    "use_center_ohe": False,
    "include_center_group_feature": False,
    "model_family": "rsf",
    "random_seed": SEED,
    "prune_feature": True,
    "prune_feature_threshold": 0.96,
}

# =============================================================================
# 10. MODELING
# =============================================================================

TAU = 7  # Horizon temporel pour l'analyse de survie

# Paramètres legacy (pour compatibilité avec anciens scripts)
RSF_PARAMS = {"n_estimators": 400, "max_depth": 15, "min_samples_split": 20, "min_samples_leaf": 10, "max_features": "sqrt", "n_jobs": -1}
GRADIENT_BOOSTING_PARAMS = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.7, "min_samples_leaf": 20, "min_samples_split": 40, "loss": "coxph"}
COX_PARAMS = {"alpha": 0.1, "n_iter": 100}
COXNET_PARAMS = {"l1_ratio": 0.95, "n_alphas": 100, "max_iter": 20000}
EXTRA_TREES_PARAMS = {"n_estimators": 400, "max_depth": 15, "min_samples_split": 20, "min_samples_leaf": 10, "max_features": "sqrt", "n_jobs": -1}
COMPONENTWISE_GB_PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "subsample": 0.7}
PYCOX_DEEPSURV_PARAMS = {
    "hidden_layers": [128, 64, 32], "dropout": 0.6, "batch_norm": True,
    "learning_rate": 0.001, "batch_size": 128, "epochs": 100, "patience": 10,
    "weight_decay": 0.005, "optimizer": "Adam",
    "lr_scheduler": True, "lr_factor": 0.5, "lr_patience": 50,
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
                "n_estimators": 800,
                "max_depth": 20,
                "min_samples_split": 20,
                "min_samples_leaf": 30,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
        },
        "GradientBoosting": {
            "enabled": False,
            "params": {
                "n_estimators": 700,
                "learning_rate": 0.02,
                "max_depth": 4,
                "subsample": 0.7,
                "min_samples_leaf": 15,
                "min_samples_split": 30,
                "loss": "coxph",
            },
        },
        "Cox": {"enabled": False, "params": {"alpha": 0.1, "n_iter": 100}},
        "CoxNet": {
            "enabled": False,
            "params": {"l1_ratio": 0.95, "n_alphas": 100, "max_iter": 20000},
        },
        "ExtraTrees": {
            "enabled": False,
            "params": {
                "n_estimators": 600,
                "max_depth": 20,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "n_jobs": -1,
            },
        },
        "ComponentwiseGB": {
            "enabled": False,
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
    "stacking": {
        "enabled": False,
        "name": "Stacking",
        "base_models": ["RSF", "GradientBoosting", "ExtraTrees"],
        "meta_model": "RSF",
        "probabilities": True,
    },
    "prediction": {
        "ensemble_meta_path": os.path.join(MODEL_DIR, "ensemble_meta.json"),
        "preprocessor_path": os.path.join(MODEL_DIR, "preprocessor.joblib"),
    }
}

HYPERPARAM_OPTIMIZATION = {
    "models": {
        "RSF": {
            "enabled": True,
            "n_trials": 300,
            "optuna_workers": 1,
            "model_n_jobs": -1,
            "search_space": {
                "n_estimators": ("int", [200, 1000]),
                "max_depth": ("int", [8, 35]),
                "min_samples_split": ("int", [5, 50]),
                "min_samples_leaf": ("int", [3, 30]),
                "max_features": ("categorical", ["sqrt", 0.2, 0.3, 0.5]),
            },
        },
        "ExtraTrees": {
            "enabled": True,
            "n_trials": 300,
            "optuna_workers": 1,
            "model_n_jobs": -1,
            "search_space": {
                "n_estimators": ("int", [200, 800]),
                "max_depth": ("int", [10, 40]),
                "min_samples_split": ("int", [5, 50]),
                "min_samples_leaf": ("int", [3, 30]),
                "max_features": ("categorical", ["sqrt", 0.2, 0.3, 0.5]),
            },
        },
        "GradientBoosting": {
            "enabled": True,
            "n_trials": 400,
            "optuna_workers": 1,
            "search_space": {
                "n_estimators": ("int", [200, 1200]),
                "learning_rate": ("float", [0.01, 0.1, True]),
                "max_depth": ("int", [2, 6]),
                "subsample": ("float", [0.6, 0.9]),
                "min_samples_leaf": ("int", [15, 60]),
                "max_features": ("float", [0.6, 1.0]),
            },
        },
    }
}