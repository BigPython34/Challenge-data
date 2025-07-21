# Configuration globale du projet
import os

# Seed pour la reproductibilité
SEED = 42

# Chemins relatifs depuis la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # challenge_code/
DATA_DIR = os.path.join(BASE_DIR, "datas")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Chemins spécifiques aux données
CLINICAL_TRAIN_PATH = os.path.join(DATA_DIR, "X_train", "clinical_train.csv")
CLINICAL_TEST_PATH = os.path.join(DATA_DIR, "X_test", "clinical_test.csv")
MOLECULAR_TRAIN_PATH = os.path.join(DATA_DIR, "X_train", "molecular_train.csv")
MOLECULAR_TEST_PATH = os.path.join(DATA_DIR, "X_test", "molecular_test.csv")
TARGET_TRAIN_PATH = os.path.join(DATA_DIR, "target_train.csv")

# Gènes importants pour l'analyse
IMPORTANT_GENES = [
    "TP53",
    "ASXL1",
    "RUNX1",
    "FLT3",
    "NPM1",
    "IDH1",
    "IDH2",
    "DNMT3A",
    "TET2",
    "CEBPA",
    "SRSF2",
    "SF3B1",
]

# Paramètres des modèles
TAU = 7  # Pour le calcul du C-index
