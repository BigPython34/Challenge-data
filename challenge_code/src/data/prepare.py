# Nettoyage, split, imputation, etc.
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
import warnings

from .. import config
from ..config import SEED
from . import features


def clean_target_data(target_df):
    """Nettoie les données target"""
    # Drop rows where 'OS_YEARS' is NaN
    target_df.dropna(subset=["OS_YEARS", "OS_STATUS"], inplace=True)

    # Convert 'OS_YEARS' to numeric
    target_df["OS_YEARS"] = pd.to_numeric(target_df["OS_YEARS"], errors="coerce")

    # Ensure 'OS_STATUS' is boolean
    target_df["OS_STATUS"] = target_df["OS_STATUS"].astype(bool)

    return target_df


def prepare_enriched_dataset(
    clinical_df,
    molecular_df,
    target_df=None,
    imputer=None,
    advanced_imputation_method="knn",
    is_training=True,
):
    """Prépare un dataset enrichi avec toutes les features et imputation avancée"""
    print(
        f"\n🔧 Préparation du dataset enrichi ({'entraînement' if is_training else 'test'})..."
    )

    # 1. Création de features basées sur les gènes
    gene_features = pd.DataFrame(index=clinical_df["ID"].unique())

    for gene in config.IMPORTANT_GENES:
        mutated_patients = molecular_df[molecular_df["GENE"] == gene]["ID"].unique()
        gene_features[f"has_{gene}_mutation"] = gene_features.index.isin(
            mutated_patients
        ).astype(int)

    # 2. Statistiques moléculaires
    mutation_counts, vaf_stats, effect_counts = (
        features.create_molecular_stats_features(molecular_df)
    )

    # 3. Fusion des features moléculaires
    mol_features = gene_features.reset_index().rename(columns={"index": "ID"})
    mol_features = pd.merge(mol_features, mutation_counts, on="ID", how="outer")
    mol_features = pd.merge(mol_features, vaf_stats, on="ID", how="outer")
    mol_features = pd.merge(mol_features, effect_counts, on="ID", how="outer")

    # 4. Fusion avec les données cliniques
    df_enriched = clinical_df.merge(mol_features, on="ID", how="left")

    # 5. Features cytogénétiques
    cyto_features = features.extract_advanced_cytogenetic_features(clinical_df)
    df_enriched = pd.concat([df_enriched, cyto_features], axis=1)

    # 6. Features cliniques avancées
    df_enriched = features.create_advanced_clinical_features(df_enriched)

    # 7. Encoder le centre médical
    center_dummies = pd.get_dummies(df_enriched["CENTER"], prefix="center")
    df_enriched = pd.concat([df_enriched, center_dummies], axis=1)

    # 8. Gestion initiale des valeurs manquantes (features binaires/catégorielles)
    for col in df_enriched.columns:
        if col not in [
            "ID",
            "CENTER",
            "BM_BLAST",
            "WBC",
            "ANC",
            "MONOCYTES",
            "HB",
            "PLT",
            "CYTOGENETICS",
            "sex",  # Gérée par l'imputation avancée
        ]:
            df_enriched[col] = df_enriched[col].fillna(0)

    # 9. Imputation avancée
    print(f"📊 Valeurs manquantes avant imputation : {df_enriched.isna().sum().sum()}")

    if is_training:
        # Appliquer l'imputation avancée pour les données d'entraînement
        clinical_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "sex"]

        # Identifier les colonnes cliniques qui existent réellement
        existing_clinical_cols = [
            col for col in clinical_cols if col in df_enriched.columns
        ]

        if existing_clinical_cols:
            df_enriched, imputation_metadata = advanced_imputation(
                df_enriched,
                method=advanced_imputation_method,
                cluster_impute_cols=existing_clinical_cols,
                sex_impute=("sex" in df_enriched.columns),
            )

            # Fallback avec SimpleImputer pour garantir qu'il n'y a plus de NaN
            remaining_clinical_cols = [
                col for col in existing_clinical_cols if col != "sex"
            ]
            if remaining_clinical_cols:
                simple_imputer = SimpleImputer(strategy="median")
                df_enriched[remaining_clinical_cols] = simple_imputer.fit_transform(
                    df_enriched[remaining_clinical_cols]
                )
                imputation_metadata["simple_imputer_fallback"] = simple_imputer
        else:
            imputation_metadata = {}
            simple_imputer = SimpleImputer(strategy="median")
            df_enriched[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]] = (
                simple_imputer.fit_transform(
                    df_enriched[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]]
                )
            )
            imputation_metadata["simple_imputer_fallback"] = simple_imputer

        print(
            f"✅ Valeurs manquantes après imputation : {df_enriched.isna().sum().sum()}"
        )
        return df_enriched, imputation_metadata

    else:
        # Pour les données de test, utiliser les imputers entraînés
        if imputer is None:
            raise ValueError("Imputer must be provided for test data")

        # Appliquer l'imputation avec les métadonnées sauvegardées
        clinical_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
        existing_clinical_cols = [
            col for col in clinical_cols if col in df_enriched.columns
        ]

        # Utiliser l'imputer simple comme fallback si disponible
        if "simple_imputer_fallback" in imputer and existing_clinical_cols:
            df_enriched[existing_clinical_cols] = imputer[
                "simple_imputer_fallback"
            ].transform(df_enriched[existing_clinical_cols])

        # Imputation spéciale pour sex si disponible
        if "sex" in df_enriched.columns and df_enriched["sex"].isna().any():
            df_enriched = impute_sex_advanced(df_enriched)

        print(
            f"✅ Valeurs manquantes après imputation : {df_enriched.isna().sum().sum()}"
        )
        return df_enriched


def prepare_features_and_target(df_enriched, target_df, test_size=0.2):
    """Prépare les features et le target pour l'entraînement"""
    # Définir les features
    feature_lists = features.get_feature_lists()

    # Obtenir les colonnes center
    center_features = [col for col in df_enriched.columns if col.startswith("center_")]
    effect_features = [
        col
        for col in df_enriched.columns
        if col
        in [
            "frameshift_variant",
            "missense_variant",
            "nonsense",
            "splice_acceptor_variant",
            "splice_donor_variant",
        ]
    ]

    # Construire la liste finale des features (éviter le conflit de nom avec le module features)
    final_features = (
        feature_lists["clinical"]
        + feature_lists["gene_mutations"]
        + feature_lists["statistics"]
        + effect_features
        + feature_lists["cytogenetic"]
        + feature_lists["ratios"]
        + feature_lists["clinical_scores"]
        + center_features
    )

    # Créer X et y
    X = df_enriched.loc[df_enriched["ID"].isin(target_df["ID"]), final_features]
    y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", target_df)

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    # Gérer les valeurs manquantes
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    return X_train, X_test, y_train, y_test, final_features


def prepare_test_dataset(df_test_enriched, feature_list, center_columns_train):
    """Prépare le dataset de test pour les prédictions"""
    # Assurer la cohérence des colonnes center
    for col in center_columns_train:
        if col not in df_test_enriched.columns:
            df_test_enriched[col] = 0

    # Vérifier que toutes les features sont présentes
    for feature in feature_list:
        if feature not in df_test_enriched.columns:
            df_test_enriched[feature] = 0

    # S'assurer que les features sont dans le même ordre
    X_test = df_test_enriched.loc[:, feature_list]
    X_test.fillna(0, inplace=True)

    return X_test


def advanced_imputation(df, method="knn", cluster_impute_cols=None, sex_impute=False):
    """
    Imputation avancée des valeurs manquantes avec plusieurs stratégies

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec valeurs manquantes
    method : str
        Méthode d'imputation : 'knn', 'cluster', 'rf', 'bagging'
    cluster_impute_cols : list
        Colonnes pour l'imputation par clustering
    sex_impute : bool
        Si True, impute la colonne 'sex' avec un modèle de bagging

    Returns:
    --------
    pd.DataFrame : DataFrame avec valeurs imputées
    dict : Metadata sur l'imputation (imputers, scalers, etc.)
    """
    df_imputed = df.copy()
    imputation_metadata = {}

    # Séparer les colonnes numériques et catégorielles
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_imputed.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Exclure ID des colonnes à imputer
    if "ID" in numeric_cols:
        numeric_cols.remove("ID")
    if "ID" in categorical_cols:
        categorical_cols.remove("ID")

    print(f"🔧 Imputation avancée avec méthode : {method}")
    print(f"   • Colonnes numériques : {len(numeric_cols)}")
    print(f"   • Colonnes catégorielles : {len(categorical_cols)}")

    # 1. Imputation des colonnes numériques
    if numeric_cols and method == "knn":
        print("   📊 Imputation KNN pour les variables numériques...")
        knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
        df_imputed[numeric_cols] = knn_imputer.fit_transform(df_imputed[numeric_cols])
        imputation_metadata["knn_imputer"] = knn_imputer

    elif numeric_cols and method == "rf":
        print("   🌲 Imputation Random Forest pour les variables numériques...")
        df_imputed = random_forest_imputation(df_imputed, numeric_cols)

    elif numeric_cols and method == "cluster":
        print("   🎯 Imputation par clustering pour les variables numériques...")
        df_imputed, cluster_metadata = cluster_based_imputation(
            df_imputed, cluster_impute_cols or numeric_cols
        )
        imputation_metadata.update(cluster_metadata)

    elif numeric_cols:
        # Fallback vers l'imputation médiane simple
        print("   📈 Imputation médiane pour les variables numériques...")
        simple_imputer = SimpleImputer(strategy="median")
        df_imputed[numeric_cols] = simple_imputer.fit_transform(
            df_imputed[numeric_cols]
        )
        imputation_metadata["simple_imputer"] = simple_imputer

    # 2. Imputation des colonnes catégorielles
    if categorical_cols:
        print("   📝 Imputation mode pour les variables catégorielles...")
        for col in categorical_cols:
            if df_imputed[col].isna().any():
                mode_value = (
                    df_imputed[col].mode().iloc[0]
                    if len(df_imputed[col].mode()) > 0
                    else "unknown"
                )
                df_imputed[col] = df_imputed[col].fillna(mode_value)

    # 3. Imputation spéciale pour la colonne 'sex' avec modèle de bagging
    if sex_impute and "sex" in df_imputed.columns:
        print("   👤 Imputation spécialisée pour la variable 'sex'...")
        df_imputed = impute_sex_advanced(df_imputed)

    # 4. Gestion des outliers
    df_imputed = handle_outliers(df_imputed, numeric_cols, method="iqr", multiplier=2.0)

    print(
        f"   ✅ Imputation terminée : {df_imputed.isna().sum().sum()} valeurs manquantes restantes"
    )

    return df_imputed, imputation_metadata


def random_forest_imputation(df, numeric_cols, n_estimators=50):
    """Imputation avec Random Forest pour chaque colonne manquante"""
    df_rf = df.copy()

    for col in numeric_cols:
        if df_rf[col].isna().any():
            # Préparer les données pour l'entraînement
            train_data = df_rf[~df_rf[col].isna()]
            predict_data = df_rf[df_rf[col].isna()]

            if len(train_data) > 0 and len(predict_data) > 0:
                # Features pour prédire cette colonne
                feature_cols = [
                    c
                    for c in numeric_cols
                    if c != col and not train_data[c].isna().all()
                ]

                if len(feature_cols) > 0:
                    X_train = train_data[feature_cols].fillna(
                        0
                    )  # Remplissage temporaire
                    y_train = train_data[col]
                    X_predict = predict_data[feature_cols].fillna(0)

                    # Entraîner le modèle
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators, random_state=SEED
                    )
                    rf.fit(X_train, y_train)

                    # Prédire les valeurs manquantes
                    predictions = rf.predict(X_predict)
                    df_rf.loc[df_rf[col].isna(), col] = predictions

    return df_rf


def cluster_based_imputation(df, cluster_cols, n_clusters=5):
    """Imputation basée sur le clustering K-means"""
    df_cluster = df.copy()
    metadata = {}

    # Sélectionner les colonnes pour le clustering (sans valeurs manquantes)
    cluster_data = df_cluster[cluster_cols].copy()

    # Imputation simple préliminaire pour pouvoir faire le clustering
    temp_imputer = SimpleImputer(strategy="median")
    cluster_data_filled = pd.DataFrame(
        temp_imputer.fit_transform(cluster_data),
        columns=cluster_cols,
        index=cluster_data.index,
    )

    # Standardisation pour le clustering
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data_filled)

    # Clustering K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(cluster_data_scaled)
    df_cluster["cluster"] = clusters

    # Imputer les valeurs manquantes en utilisant la médiane de chaque cluster
    for col in cluster_cols:
        if df_cluster[col].isna().any():
            for cluster_id in range(n_clusters):
                cluster_mask = df_cluster["cluster"] == cluster_id
                cluster_median = df_cluster.loc[cluster_mask, col].median()

                # Si pas de valeur dans ce cluster, utiliser la médiane globale
                if pd.isna(cluster_median):
                    cluster_median = df_cluster[col].median()

                # Imputer les valeurs manquantes de ce cluster
                missing_mask = df_cluster[col].isna() & cluster_mask
                df_cluster.loc[missing_mask, col] = cluster_median

    # Supprimer la colonne temporaire de cluster
    df_cluster = df_cluster.drop("cluster", axis=1)
    metadata["scaler"] = scaler
    metadata["kmeans"] = kmeans

    return df_cluster, metadata


def impute_sex_advanced(df):
    """
    Impute les valeurs inconnues (0.5) de la colonne 'sex'
    en entraînant un modèle de bagging sur les observations dont le sexe est connu (0 ou 1).
    Basé sur la méthode de preprocess.py

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec une colonne 'sex' contenant des valeurs 0.5 à imputer

    Returns:
    --------
    pd.DataFrame : DataFrame avec la colonne 'sex' imputée
    """
    df_imputed = df.copy()

    # Vérifier que la colonne 'sex' existe
    if "sex" not in df_imputed.columns:
        warnings.warn("Colonne 'sex' non trouvée, pas d'imputation effectuée")
        return df_imputed

    # Sélectionner les observations où le sexe est connu (0 ou 1)
    known_mask = (df_imputed["sex"] != 0.5) & (~df_imputed["sex"].isna())
    unknown_mask = (df_imputed["sex"] == 0.5) | (df_imputed["sex"].isna())

    # Si aucune donnée connue n'est présente ou aucune donnée à imputer
    if not known_mask.any() or not unknown_mask.any():
        print("   ⚠️  Pas d'imputation possible pour 'sex': données insuffisantes")
        return df_imputed

    # Préparer les données d'entraînement pour prédire 'sex'
    known_data = df_imputed[known_mask]
    unknown_data = df_imputed[unknown_mask]

    # Sélectionner les features numériques (exclure ID, sex, et autres colonnes non-numériques)
    feature_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in ["ID", "sex"]]

    if len(feature_cols) == 0:
        print("   ⚠️  Pas de features numériques pour prédire 'sex'")
        return df_imputed

    # Préparer X et y pour l'entraînement
    X_train = known_data[feature_cols].fillna(0)  # Remplissage temporaire des NaN
    y_train = known_data["sex"]
    X_predict = unknown_data[feature_cols].fillna(0)

    # Création du modèle de bagging avec 500 estimateurs
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=500,
        oob_score=True,
        random_state=SEED,
    )

    try:
        model.fit(X_train, y_train)
        print(f"   📊 Modèle de bagging entraîné (OOB Score: {model.oob_score_:.3f})")

        # Prédire les valeurs manquantes
        predictions = model.predict(X_predict)
        df_imputed.loc[unknown_mask, "sex"] = predictions

        print(f"   ✅ {unknown_mask.sum()} valeurs 'sex' imputées")

    except (ValueError, RuntimeError) as e:
        print(f"   ❌ Erreur lors de l'imputation de 'sex': {e}")
        # En cas d'erreur, remplacer par la valeur la plus fréquente
        most_frequent = (
            known_data["sex"].mode().iloc[0] if len(known_data["sex"].mode()) > 0 else 0
        )
        df_imputed.loc[unknown_mask, "sex"] = most_frequent

    return df_imputed


def handle_outliers(df, numeric_cols, method="iqr", multiplier=1.5, threshold=0.25):
    """
    Traite les outliers dans un DataFrame selon différentes méthodes.
    Basé sur la méthode process_outliers de preprocess.py

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame à traiter
    numeric_cols : list
        Liste des colonnes numériques à traiter
    method : str
        Méthode de traitement : 'iqr', 'zscore', 'quantile'
    multiplier : float
        Facteur multiplicateur pour la méthode IQR
    threshold : float
        Seuil pour la méthode quantile (ex: 0.25 pour Q1/Q3)

    Returns:
    --------
    pd.DataFrame : DataFrame avec outliers traités
    """
    df_out = df.copy()

    if len(numeric_cols) == 0:
        return df_out

    outliers_treated = 0

    for col in numeric_cols:
        if col not in df_out.columns:
            continue

        original_values = df_out[col].notna().sum()
        if original_values == 0:
            continue

        if method == "iqr":
            # Méthode IQR (Interquartile Range)
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

        elif method == "quantile":
            # Méthode par quantiles
            lower_bound = df_out[col].quantile(threshold)
            upper_bound = df_out[col].quantile(1 - threshold)

        elif method == "zscore":
            # Méthode Z-score (nécessite une distribution normale)
            mean_val = df_out[col].mean()
            std_val = df_out[col].std()
            lower_bound = mean_val - multiplier * std_val
            upper_bound = mean_val + multiplier * std_val

        else:
            print(f"   ⚠️  Méthode '{method}' non reconnue pour {col}")
            continue

        # Compter les outliers avant traitement
        outliers_before = (
            (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        ).sum()

        # Limiter (clip) les valeurs en dehors de l'intervalle
        df_out[col] = df_out[col].clip(lower_bound, upper_bound)

        if outliers_before > 0:
            outliers_treated += outliers_before
            print(
                f"   🔧 {outliers_before} outliers traités pour '{col}' [{lower_bound:.2f}, {upper_bound:.2f}]"
            )

    if outliers_treated > 0:
        print(f"   ✅ Total: {outliers_treated} outliers traités")
    else:
        print("   ℹ️  Aucun outlier détecté")

    return df_out
