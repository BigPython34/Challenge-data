# Configuration globale du projet
import os

# Seed pour la reproductibilite
SEED = 42

# Chemins relatifs depuis la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # challenge_code/
DATA_DIR = os.path.join(BASE_DIR, "datas")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Chemins specifiques aux donnees
CLINICAL_TRAIN_PATH = os.path.join(DATA_DIR, "X_train", "clinical_train.csv")
CLINICAL_TEST_PATH = os.path.join(DATA_DIR, "X_test", "clinical_test.csv")
MOLECULAR_TRAIN_PATH = os.path.join(DATA_DIR, "X_train", "molecular_train.csv")
MOLECULAR_TEST_PATH = os.path.join(DATA_DIR, "X_test", "molecular_test.csv")
TARGET_TRAIN_PATH = os.path.join(DATA_DIR, "target_train.csv")

# Genes importants pour l'analyse
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

# Parametres des modeles
TAU = 7  # Pour le calcul du C-index

# Parametres des modeles de survie
COX_PARAMS = {
    "alpha": 1.0,
}

RSF_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_depth": 5,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 850,
    "learning_rate": 0.03160736770883459,
    "max_depth": 3,
    "subsample": 0.8118022194771892,
    "min_samples_leaf": 4,
    "min_samples_split": 4,
}

# Parametres pour CoxNet (Cox avec regularisation)
COXNET_PARAMS = {
    "l1_ratio": 0.9,  # Equilibre entre L1 et L2 (0.9 = principalement L1)
    "alphas": None,  # Auto-determination
    "n_alphas": 100,
    "normalize": True,
    "max_iter": 100000,
}

# Parametres pour Extra Survival Trees
EXTRA_TREES_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_depth": 10,
    "max_features": "sqrt",
}

# Parametres pour Componentwise Gradient Boosting
COMPONENTWISE_GB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "subsample": 1.0,
}

# Parametres pour PyCox DeepSurv
PYCOX_DEEPSURV_PARAMS = {
    "hidden_layers": [128, 64, 32],  # Reseau plus profond
    "dropout": 0.6,  # Plus de dropout pour regularisation
    "batch_norm": True,
    "learning_rate": 0.001,  # Learning rate plus faible
    "batch_size": 128,  # Batch size plus petit pour plus de stabilite
    "epochs": 2000,  # Epochs optimises
    "patience": 400,  # Patience genereuse
    "weight_decay": 0.005,  # Weight decay plus faible
    "optimizer": "Adam",
    "lr_scheduler": True,  # Scheduler de learning rate
    "lr_factor": 0.5,  # Facteur de reduction du LR
    "lr_patience": 50,  # Patience pour le scheduler
}

# Parametres d'imputation avancee
IMPUTATION_PARAMS = {
    "knn_neighbors": 7,  # Augmente pour plus de robustesse
    "cluster_n_clusters": 8,  # Plus de clusters pour plus de precision
    "rf_n_estimators": 100,  # Plus d'estimateurs pour Random Forest
    "bagging_n_estimators": 1000,  # Plus d'estimateurs pour bagging
    "outlier_multiplier": 1.8,  # Moins restrictif sur les outliers
    "iterative_max_iter": 20,  # Nombre d'iterations pour imputation iterative
}
