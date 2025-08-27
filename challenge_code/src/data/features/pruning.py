import os
from typing import List, Optional, Sequence, Tuple
from ...config import MISSINGNESS_POLICY, REDUNDANCY_POLICY
import numpy as np
import pandas as pd


def _compute_upper_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation absolue et garde uniquement la partie supérieure.
    """
    corr_matrix = df.corr(numeric_only=True).abs()
    if corr_matrix.empty:
        return corr_matrix
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_matrix_upper = corr_matrix.where(upper)
    return corr_matrix_upper


def _apply_priority_rules(corr_upper: pd.DataFrame, threshold: float) -> List[str]:
    """
    Applique des règles de priorité pour décider quelles colonnes supprimer
    lorsqu'une forte corrélation est détectée.

    Règles implémentées (basées sur le script d'origine):
      - Garder les features mutation spécifiques (mut_*) vs *_altered ou *_count
      - Garder la version log en supprimant l'original (si corrélé)
      - Garder la moyenne vs supprimer la médiane (si corrélées)
    """
    to_drop: set[str] = set()

    for col in corr_upper.columns:
        if col not in corr_upper.columns:
            continue
        strong_corrs = corr_upper.index[corr_upper[col] > threshold].tolist()
        for correlated_col in strong_corrs:
            if correlated_col not in corr_upper.index:
                continue

            # Règle 1: mut_* prioritaire sur *_altered et *_count
            if ("_altered" in col or "_count" in col) and ("mut_" in correlated_col):
                to_drop.add(col)
            elif ("_altered" in correlated_col or "_count" in correlated_col) and (
                "mut_" in col
            ):
                to_drop.add(correlated_col)

            # Règle 2: garder la version log_*
            if f"log_{col}" == correlated_col:
                to_drop.add(col)
            elif f"log_{correlated_col}" == col:
                to_drop.add(correlated_col)

            # Règle 3: mean vs median -> garder mean
            if ("median" in col) and ("mean" in correlated_col):
                to_drop.add(col)
            elif ("mean" in col) and ("median" in correlated_col):
                to_drop.add(correlated_col)

    return list(to_drop)


def prune_highly_correlated_features(
    df: pd.DataFrame, threshold: float = 0.90, id_cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Supprime les features fortement corrélées d'un DataFrame unique, en excluant
    les colonnes d'identifiants. Utilisée pour compatibilité.
    """
    if id_cols is None:
        id_cols = ("ID", "CENTER_GROUP")

    print(f"\n[PRUNING] Élagage des features (seuil > {threshold}) sur un DataFrame...")
    df_to_prune = df.copy()

    # Exclure les colonnes d'identifiants du calcul de corrélation
    feature_cols = [c for c in df_to_prune.columns if c not in id_cols]
    work_df = df_to_prune[feature_cols]

    corr_upper = _compute_upper_corr(work_df)
    if corr_upper.empty:
        print(
            "   -> Aucune colonne numérique trouvée pour la corrélation. Rien à supprimer."
        )
        return df_to_prune

    # Règles de priorité
    to_drop = set(_apply_priority_rules(corr_upper, threshold))
    df_mid = work_df.drop(columns=list(to_drop), errors="ignore")
    print(f"   -> {len(to_drop)} features supprimées par règles de priorité.")

    # Méthode générale: supprimer la 2e des paires restantes au-dessus du seuil
    corr_upper2 = _compute_upper_corr(df_mid)
    to_drop_final: set[str] = set()
    for col in corr_upper2.columns:
        strong_corrs_final = corr_upper2.index[corr_upper2[col] > threshold].tolist()
        if strong_corrs_final:
            to_drop_final.update(strong_corrs_final)

    df_final = df_mid.drop(columns=list(to_drop_final), errors="ignore")
    print(
        f"   -> {len(to_drop_final)} features supplémentaires supprimées par la méthode générale."
    )

    # Réinsérer colonnes d'identifiants au bon endroit si besoin
    for id_col in reversed(list(id_cols)):
        if id_col in df_to_prune.columns and id_col not in df_final.columns:
            # Inserer au début par défaut
            df_final.insert(0, id_col, df_to_prune[id_col].values)

    # Conserver l'ordre initial autant que possible
    ordered_cols = [c for c in df_to_prune.columns if c in df_final.columns]
    return df_final[ordered_cols]


def prune_highly_correlated_features_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.90,
    id_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Variante à 2 DataFrames: le choix des colonnes à supprimer est décidé depuis le train,
    puis appliqué à train et test pour garder des schémas identiques.
    """
    if id_cols is None:
        id_cols = ("ID", "CENTER_GROUP")
    # Toujours travailler avec une liste pour l'indexation pandas
    id_cols_list = [c for c in id_cols]

    print(
        f"\n[PRUNING] Élagage des features corrélées (seuil > {threshold}) basé sur le train..."
    )

    # Exclure les colonnes d'identifiants
    feature_cols = [c for c in train_df.columns if c not in id_cols_list]
    train_feat = train_df[feature_cols]
    test_feat = test_df.reindex(columns=feature_cols)

    # Règles de priorité
    corr_upper = _compute_upper_corr(train_feat)
    if corr_upper.empty:
        print(
            "   -> Aucune colonne numérique trouvée pour la corrélation. Rien à supprimer."
        )
        return train_df.copy(), test_df.copy()

    to_drop_priority = set(_apply_priority_rules(corr_upper, threshold))
    train_mid = train_feat.drop(columns=list(to_drop_priority), errors="ignore")
    print(f"   -> {len(to_drop_priority)} features supprimées par règles de priorité.")

    # Méthode générale
    corr_upper2 = _compute_upper_corr(train_mid)
    to_drop_general: set[str] = set()
    for col in corr_upper2.columns:
        strong_corrs_final = corr_upper2.index[corr_upper2[col] > threshold].tolist()
        if strong_corrs_final:
            to_drop_general.update(strong_corrs_final)

    kept_cols = [c for c in train_mid.columns if c not in to_drop_general]
    print(
        f"   -> {len(to_drop_general)} features supplémentaires supprimées par la méthode générale."
    )
    print(list(to_drop_general))
    id_cols_present_train = [c for c in id_cols_list if c in train_df.columns]
    id_cols_present_test = [c for c in id_cols_list if c in test_df.columns]

    pruned_train = (
        pd.concat([train_df[id_cols_present_train].copy(), train_df[kept_cols]], axis=1)
        if id_cols_present_train
        else train_df[kept_cols]
    )
    pruned_test = (
        pd.concat([test_df[id_cols_present_test].copy(), test_feat[kept_cols]], axis=1)
        if id_cols_present_test
        else test_feat[kept_cols]
    )

    return pruned_train, pruned_test


def apply_pruning_to_processed_files(
    input_train_path: str,
    input_test_path: str,
    output_train_path: Optional[str] = None,
    output_test_path: Optional[str] = None,
    threshold: float = 0.90,
    id_cols: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """
    Utilitaire pour appliquer l'élagage directement sur des fichiers CSV 'processed'.
    Écrit par défaut par-dessus les fichiers d'entrée.
    """
    if output_train_path is None:
        output_train_path = input_train_path
    if output_test_path is None:
        output_test_path = input_test_path

    X_train = pd.read_csv(input_train_path)
    X_test = pd.read_csv(input_test_path)

    pruned_train, pruned_test = prune_highly_correlated_features_pair(
        X_train, X_test, threshold=threshold, id_cols=id_cols
    )

    os.makedirs(os.path.dirname(output_train_path) or ".", exist_ok=True)
    pruned_train.to_csv(output_train_path, index=False)
    pruned_test.to_csv(output_test_path, index=False)

    return output_train_path, output_test_path


__all__ = [
    "prune_highly_correlated_features",
    "prune_highly_correlated_features_pair",
    "apply_pruning_to_processed_files",
]


# -----------------
# Redundancy cleanup
# -----------------
def _apply_redundancy_policy(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    policy = REDUNDANCY_POLICY or {}
    drop_cols: List[str] = []

    # Drop numeric sex encoding when one-hot is present
    if policy.get("drop_sex_numeric_if_ohe", True):
        if {"SEX_XX", "SEX_XY"}.issubset(
            df.columns
        ) and "sex_chromosomes" in df.columns:
            drop_cols.append("sex_chromosomes")

    # Drop *_count when *_altered exists
    if policy.get("drop_count_when_binary_exists", True):
        for c in df.columns:
            if c.endswith("_count"):
                alt = c.replace("_count", "_altered")
                if alt in df.columns:
                    drop_cols.append(c)
    # Drop *_count when matching any_* exists
    if policy.get("drop_count_when_any_exists", True):
        for c in df.columns:
            if c.endswith("_count"):
                base = c[: -len("_count")]
                # Keep COSMIC counts even if an any_ flag exists; these counts
                # are valuable and should not be auto-dropped by the generic
                # redundancy policy.
                if base.startswith("cosmic_"):
                    continue
                any_col = f"any_{base}"
                if any_col in df.columns:
                    drop_cols.append(c)

    # Prune missingness indicators except whitelisted
    if policy.get("prune_missingness_indicators", True):
        keep = set(MISSINGNESS_POLICY.get("keep_columns", []))
        miss = [c for c in df.columns if c.endswith("_missing")]
        for c in miss:
            base = c[:-8]
            if base not in keep:
                drop_cols.append(c)

    # Explicit drops
    for c in policy.get("explicit_drop", []):
        if c in df.columns:
            drop_cols.append(c)

    if drop_cols:
        df = df.drop(columns=sorted(set(drop_cols)), errors="ignore")
    return df


def prune_rare_binary_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float,
    ignore_cols: List[str],
) -> (pd.DataFrame, pd.DataFrame):
    """
    Identifie et supprime les features binaires rares des jeux d'entraînement et de test.
    """
    print("Démarrage du pruning des features binaires rares...")

    # S'assurer que les dataframes sont des copies pour éviter les warnings
    train_df_pruned = train_df.copy()
    test_df_pruned = test_df.copy()

    cols_to_prune = []

    # Itérer sur toutes les colonnes du jeu d'entraînement
    for col in train_df_pruned.columns:
        if col in ignore_cols:
            continue

        # On ne s'intéresse qu'aux colonnes qui semblent binaires (0 ou 1)
        # On tolère les floats (ex: 0.0, 1.0) et les NaNs
        unique_vals = pd.unique(train_df_pruned[col].dropna())
        is_binary = np.all(np.isin(unique_vals, [0, 1]))

        if is_binary:
            # Calculer la prévalence (fréquence de la valeur 1)
            prevalence_train = train_df_pruned[col].mean()

            # Vérifier aussi la prévalence dans le jeu de test si la colonne existe
            prevalence_test = 0
            if col in test_df_pruned.columns:
                prevalence_test = test_df_pruned[col].mean()

            # On supprime si la feature est rare DANS L'UN OU L'AUTRE des jeux
            if prevalence_train < threshold or prevalence_test < threshold:
                cols_to_prune.append(col)

    if cols_to_prune:
        print(
            f"\n{len(cols_to_prune)} features rares identifiées pour suppression (prévalence < {threshold:.2%}):"
        )
        print(f"{cols_to_prune} will be supp")
        # Afficher par groupes pour une meilleure lisibilité
        groups = {}
        for col in sorted(cols_to_prune):
            prefix = col.split("_")[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(col)

        for prefix, features in groups.items():
            print(f"  - {prefix.upper()}: {features}")

        # Supprimer les colonnes des deux dataframes
        train_df_pruned.drop(columns=cols_to_prune, inplace=True, errors="ignore")
        test_df_pruned.drop(columns=cols_to_prune, inplace=True, errors="ignore")

        print(f"\nShape de X_train après pruning : {train_df_pruned.shape}")
        print(f"Shape de X_test après pruning  : {test_df_pruned.shape}")
    else:
        print(
            "\nAucune feature binaire rare n'a été trouvée. Aucune suppression nécessaire."
        )

    return train_df_pruned, test_df_pruned
