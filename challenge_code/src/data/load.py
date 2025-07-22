# Fonctions de chargement des donnees
import pandas as pd
from .. import config


def load_clinical_data(path=None, train=True):
    """Charge les donnees cliniques"""
    if path is None:
        path = config.CLINICAL_TRAIN_PATH if train else config.CLINICAL_TEST_PATH
    return pd.read_csv(path)


def load_molecular_data(path=None, train=True):
    """Charge les donnees moleculaires"""
    if path is None:
        path = config.MOLECULAR_TRAIN_PATH if train else config.MOLECULAR_TEST_PATH
    return pd.read_csv(path)


def load_target_data(path=None):
    """Charge les donnees cibles (target)"""
    if path is None:
        path = config.TARGET_TRAIN_PATH
    return pd.read_csv(path)


def load_all_data():
    """Charge toutes les donnees d'entrainement"""
    clinical_train = load_clinical_data(train=True)
    clinical_test = load_clinical_data(train=False)
    molecular_train = load_molecular_data(train=True)
    molecular_test = load_molecular_data(train=False)
    target_train = load_target_data()

    return {
        "clinical_train": clinical_train,
        "clinical_test": clinical_test,
        "molecular_train": molecular_train,
        "molecular_test": molecular_test,
        "target_train": target_train,
    }
