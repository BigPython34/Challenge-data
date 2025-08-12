import pandas as pd
import numpy as np
import os


def identify_unstable_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, variance_threshold=1e-8
) -> list:
    """
    Identifie les features instables entre les jeux d'entraînement et de test.
    Cible les features à variance nulle dans le test et celles liées aux 'CENTER'
    qui ont une forte dérive de distribution.
    """
    unstable_features_to_drop = set()

    # --- Critère 1: Variance nulle dans le test (obligatoire à supprimer) ---
    train_variance = X_train.var()
    test_variance = X_test.var()

    stable_in_train = train_variance > variance_threshold
    zero_in_test = test_variance < variance_threshold

    zero_variance_in_test = train_variance.index[
        stable_in_train & zero_in_test
    ].tolist()

    if zero_variance_in_test:
        print(
            f"\n[STABILITY] {len(zero_variance_in_test)} features avec variance nulle uniquement dans le test (seront supprimées) :"
        )
        print(sorted(zero_variance_in_test))
        unstable_features_to_drop.update(zero_variance_in_test)

    # --- Critère 2: Forte dérive de distribution ---
    train_describes = X_train.describe().T
    test_describes = X_test.describe().T

    drift_df = pd.concat(
        [train_describes["mean"], test_describes["mean"]],
        axis=1,
        keys=["train_mean", "test_mean"],
    )
    drift_df["drift_ratio"] = (drift_df["test_mean"] + 1e-6) / (
        drift_df["train_mean"] + 1e-6
    )

    high_drift_features = drift_df[
        ((drift_df["drift_ratio"] > 2) | (drift_df["drift_ratio"] < 0.5))
        & (drift_df["train_mean"] > 0.01)
    ].index.tolist()

    if high_drift_features:
        print(
            f"\n[STABILITY] {len(high_drift_features)} features avec une forte dérive de distribution détectée."
        )

        # Stratégie de suppression fine : ne supprimer que les features 'CENTER'
        center_features_with_drift = [f for f in high_drift_features if "CENTER" in f]

        if center_features_with_drift:
            print(
                f"   -> Parmi elles, {len(center_features_with_drift)} features 'CENTER' seront supprimées :"
            )
            print(sorted(center_features_with_drift))
            unstable_features_to_drop.update(center_features_with_drift)

    return list(unstable_features_to_drop)


def prune_redundant_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame, threshold: float = 0.90
) -> (pd.DataFrame, pd.DataFrame):
    """
    Identifie et supprime les features fortement corrélées de manière systématique.
    Pour chaque paire dépassant le seuil, une des deux features est supprimée.
    """
    print(
        f"\n[PRUNING] Recherche de features redondantes avec un seuil de corrélation > {threshold}..."
    )

    # Calculer la corrélation sur le jeu d'entraînement uniquement
    corr_matrix = df_train.corr().abs()

    # Créer un masque pour la partie supérieure de la matrice
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_matrix_upper = corr_matrix.where(upper)

    # Trouver les colonnes à supprimer
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                # Stratégie : garder la feature qui a la plus faible corrélation moyenne avec toutes les autres
                # C'est une heuristique pour garder la feature la moins "redondante" en général
                if corr_matrix[col_i].mean() > corr_matrix[col_j].mean():
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)

    if to_drop:
        print(
            f"   -> {len(to_drop)} features redondantes identifiées pour suppression :"
        )
        print(sorted(list(to_drop)))
        df_train_pruned = df_train.drop(columns=list(to_drop))
        df_test_pruned = df_test.drop(columns=list(to_drop), errors="ignore")
        return df_train_pruned, df_test_pruned
    else:
        print("   -> Aucune feature fortement redondante trouvée.")
        return df_train, df_test


def main():
    """Orchestre la sélection des features stables et l'élagage des features redondantes."""
    print("=" * 80)
    print(" SCRIPT 1b: SÉLECTION DES FEATURES STABLES ET ÉLAGAGE")
    print("=" * 80)

    # --- 1. CHARGER LES DATASETS PRÉTRAITÉS ---
    print("\n[STEP 1/4] Chargement des datasets prétraités...")
    try:
        X_train = pd.read_csv("datasets_processed/X_train_processed.csv")
        X_test = pd.read_csv("datasets_processed/X_test_processed.csv")
    except FileNotFoundError:
        print(
            "   [ERREUR] Fichiers _processed.csv non trouvés. Exécutez 1_prepare_data.py d'abord."
        )
        return

    # --- 2. SÉLECTION DES FEATURES STABLES ---
    print("\n[STEP 2/4] Identification et suppression des features instables...")
    train_ids = X_train["ID"].copy()
    test_ids = X_test["ID"].copy()
    features_to_drop_stable = identify_unstable_features(
        X_train.drop(columns=["ID"]), X_test.drop(columns=["ID"])
    )

    X_train_stable = X_train.drop(columns=features_to_drop_stable, errors="ignore")
    X_test_stable = X_test.drop(columns=features_to_drop_stable, errors="ignore")

    # --- 3. ÉLAGAGE DES FEATURES REDONDANTES ---
    # On passe les dataframes stables (sans ID) à la fonction d'élagage
    X_train_pruned, X_test_pruned = prune_redundant_features(
        X_train_stable.drop(columns=["ID"]), X_test_stable.drop(columns=["ID"])
    )

    # Réinsérer les IDs pour la sauvegarde
    X_train_pruned.insert(0, "ID", train_ids)
    X_test_pruned.insert(0, "ID", test_ids)

    # --- 4. SAUVEGARDE FINALE ---
    print(f"\n[STEP 4/4] Sauvegarde des datasets finaux (stables et élagués)...")
    output_dir = "datasets_final"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "X_train_final.csv")
    test_path = os.path.join(output_dir, "X_test_final.csv")

    X_train_pruned.to_csv(train_path, index=False)
    X_test_pruned.to_csv(test_path, index=False)

    print("\n--- RÉCAPITULATIF ---")
    print(f"   Nombre de features initial : {len(X_train.columns) - 1}")
    print(
        f"   Nombre de features après sélection de stabilité : {len(X_train_stable.columns) - 1}"
    )
    print(f"   Nombre de features après élagage : {len(X_train_pruned.columns) - 1}")
    print(
        f"   ✓ Fichiers finaux prêts pour l'entraînement dans le dossier '{output_dir}'"
    )


if __name__ == "__main__":
    main()
