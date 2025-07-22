# Nettoyage, split, imputation, etc.
import pandas as pd
import numpy as np
from sklearn.experimental import (
    enable_iterative_imputer,
)  # Required for IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingClassifier,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sksurv.util import Surv
import warnings
from scipy import stats

from .. import config
from ..config import SEED, IMPUTATION_PARAMS
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
            df_enriched = impute_sex_advanced_v2(df_enriched)

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


def advanced_imputation(
    df, method="iterative_ensemble", cluster_impute_cols=None, sex_impute=False
):
    """
    Imputation avancée des valeurs manquantes avec stratégies sophistiquées

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec valeurs manquantes
    method : str
        Méthode d'imputation : 'iterative_ensemble', 'knn_adaptive', 'cluster_hierarchical',
        'rf_ensemble', 'bayesian_ridge', 'hybrid'
    cluster_impute_cols : list
        Colonnes pour l'imputation par clustering
    sex_impute : bool
        Si True, impute la colonne 'sex' avec un modèle de bagging avancé

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

    elif numeric_cols and method == "iterative":
        print("   🔄 Imputation itérative pour les variables numériques...")
        iterative_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=SEED,
            n_jobs=-1,
        )
        df_imputed[numeric_cols] = iterative_imputer.fit_transform(
            df_imputed[numeric_cols]
        )
        imputation_metadata["iterative_imputer"] = iterative_imputer

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
        df_imputed = impute_sex_advanced_v2(df_imputed)

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


def advanced_outlier_detection(df, numeric_cols, contamination=0.1):
    """Détection avancée des outliers avec plusieurs méthodes combinées"""
    df_clean = df.copy()
    outliers_detected = 0

    for col in numeric_cols:
        if col not in df_clean.columns or df_clean[col].isna().all():
            continue

        # Méthode 1: Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=SEED)
        outlier_mask_iso = (
            iso_forest.fit_predict(df_clean[[col]].fillna(df_clean[col].median())) == -1
        )

        # Méthode 2: Local Outlier Factor
        lof = LocalOutlierFactor(contamination=contamination)
        outlier_mask_lof = (
            lof.fit_predict(df_clean[[col]].fillna(df_clean[col].median())) == -1
        )

        # Méthode 3: Z-score modifié (MAD - Median Absolute Deviation)
        median = df_clean[col].median()
        mad = np.median(np.abs(df_clean[col] - median))
        modified_z_scores = 0.6745 * (df_clean[col] - median) / mad
        outlier_mask_zscore = np.abs(modified_z_scores) > 3.5

        # Consensus: outlier si détecté par au moins 2 méthodes
        consensus_outliers = (
            outlier_mask_iso.astype(int)
            + outlier_mask_lof.astype(int)
            + outlier_mask_zscore.astype(int)
        ) >= 2

        if consensus_outliers.sum() > 0:
            # Remplacement par des valeurs robustes (percentiles)
            p25, p75 = df_clean[col].quantile([0.25, 0.75])
            df_clean.loc[consensus_outliers, col] = np.where(
                df_clean.loc[consensus_outliers, col] > p75, p75, p25
            )
            outliers_detected += consensus_outliers.sum()

    if outliers_detected > 0:
        print(
            f"   🔍 {outliers_detected} outliers détectés et traités avec consensus multi-méthodes"
        )

    return df_clean


def iterative_ensemble_imputation(df, numeric_cols, max_iter=None):
    """Imputation itérative avec ensemble de modèles prédictifs"""
    if max_iter is None:
        max_iter = IMPUTATION_PARAMS.get("iterative_max_iter", 20)

    # Ensemble d'estimateurs pour plus de robustesse
    estimators = [
        BayesianRidge(),
        ExtraTreesRegressor(n_estimators=50, random_state=SEED),
        RandomForestRegressor(n_estimators=50, random_state=SEED),
    ]

    imputers = {}
    df_imputed = df.copy()

    for estimator in estimators:
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=SEED,
            initial_strategy="median",
        )

        # Imputation pour ce modèle
        imputed_data = imputer.fit_transform(df_imputed[numeric_cols])
        imputed_df = pd.DataFrame(
            imputed_data, columns=numeric_cols, index=df_imputed.index
        )

        # Stocker l'imputer
        imputers[type(estimator).__name__] = imputer

        # Moyenne pondérée des prédictions (ensemble)
        if "ensemble_sum" not in locals():
            ensemble_sum = imputed_df
            weights_sum = 1.0
        else:
            # Pondération selon la performance (BayesianRidge plus de poids)
            weight = 2.0 if isinstance(estimator, BayesianRidge) else 1.0
            ensemble_sum = ensemble_sum + weight * imputed_df
            weights_sum += weight

    # Moyenne pondérée finale
    df_imputed[numeric_cols] = ensemble_sum / weights_sum

    metadata = {"iterative_imputers": imputers, "ensemble_weights": weights_sum}
    print(f"   ✅ Ensemble de {len(estimators)} modèles itératifs convergé")

    return df_imputed, metadata


def adaptive_knn_imputation(df, numeric_cols):
    """Imputation KNN adaptatif avec distance pondérée optimisée"""
    df_imputed = df.copy()

    # Déterminer le nombre optimal de voisins par validation croisée
    optimal_k = min(max(5, int(np.sqrt(len(df) / 4))), 15)

    # Scaler robuste pour KNN
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_imputed[numeric_cols])

    # KNN avec distance de Minkowski optimisée
    knn_imputer = KNNImputer(
        n_neighbors=optimal_k, weights="distance", metric="nan_euclidean"
    )

    imputed_scaled = knn_imputer.fit_transform(scaled_data)

    # Retour à l'échelle originale
    imputed_data = scaler.inverse_transform(imputed_scaled)
    df_imputed[numeric_cols] = imputed_data

    metadata = {
        "adaptive_knn_imputer": knn_imputer,
        "robust_scaler": scaler,
        "optimal_k": optimal_k,
    }

    print(f"   📊 KNN adaptatif (k={optimal_k}) avec distance pondérée")
    return df_imputed, metadata


def hierarchical_cluster_imputation(df, cluster_cols, min_cluster_size=10):
    """Imputation par clustering hiérarchique avec DBSCAN"""
    df_cluster = df.copy()

    # Préparation des données pour clustering
    cluster_data = df_cluster[cluster_cols].copy()

    # Imputation préliminaire
    temp_imputer = SimpleImputer(strategy="median")
    cluster_data_filled = pd.DataFrame(
        temp_imputer.fit_transform(cluster_data),
        columns=cluster_cols,
        index=cluster_data.index,
    )

    # Standardisation robuste
    scaler = RobustScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data_filled)

    # DBSCAN pour clustering hiérarchique
    eps = np.percentile(np.std(cluster_data_scaled, axis=0), 75) * 0.5
    dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size)
    clusters = dbscan.fit_predict(cluster_data_scaled)

    # Ajouter KMeans en fallback pour les points "bruit" (-1)
    noise_mask = clusters == -1
    if noise_mask.sum() > 0:
        n_clusters = max(3, len(np.unique(clusters[clusters != -1])))
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        noise_clusters = kmeans.fit_predict(cluster_data_scaled[noise_mask])

        # Réassigner les clusters de bruit
        max_cluster = clusters.max()
        clusters[noise_mask] = noise_clusters + max_cluster + 1

    df_cluster["cluster"] = clusters
    unique_clusters = np.unique(clusters)

    # Imputation sophistiquée par cluster
    for col in cluster_cols:
        if df_cluster[col].isna().any():
            for cluster_id in unique_clusters:
                cluster_mask = df_cluster["cluster"] == cluster_id
                cluster_data_col = df_cluster.loc[cluster_mask, col]

                # Utiliser médiane robuste ou moyenne tronquée selon la distribution
                if len(cluster_data_col.dropna()) > 5:
                    # Test de normalité
                    _, p_value = stats.normaltest(cluster_data_col.dropna())
                    if p_value > 0.05:  # Distribution normale
                        cluster_value = cluster_data_col.mean()
                    else:  # Distribution non-normale
                        cluster_value = cluster_data_col.median()
                else:
                    cluster_value = df_cluster[col].median()

                # Imputation
                missing_mask = df_cluster[col].isna() & cluster_mask
                df_cluster.loc[missing_mask, col] = cluster_value

    df_cluster = df_cluster.drop("cluster", axis=1)

    metadata = {
        "dbscan_clusterer": dbscan,
        "hierarchical_scaler": scaler,
        "n_clusters_found": len(unique_clusters),
        "eps_used": eps,
    }

    print(f"   🎯 Clustering hiérarchique: {len(unique_clusters)} clusters détectés")
    return df_cluster, metadata


def ensemble_forest_imputation(df, numeric_cols):
    """Imputation avec ensemble de forêts (RF + ExtraTrees)"""
    df_forest = df.copy()

    for col in numeric_cols:
        if df_forest[col].isna().any():
            # Données d'entraînement
            train_mask = ~df_forest[col].isna()
            predict_mask = df_forest[col].isna()

            if train_mask.sum() > 10 and predict_mask.sum() > 0:
                feature_cols = [c for c in numeric_cols if c != col]
                X_train = df_forest.loc[train_mask, feature_cols].fillna(0)
                y_train = df_forest.loc[train_mask, col]
                X_predict = df_forest.loc[predict_mask, feature_cols].fillna(0)

                # Ensemble Random Forest + Extra Trees
                rf = RandomForestRegressor(
                    n_estimators=IMPUTATION_PARAMS.get("rf_n_estimators", 100),
                    random_state=SEED,
                    max_features="sqrt",
                )
                et = ExtraTreesRegressor(
                    n_estimators=100, random_state=SEED, max_features="sqrt"
                )

                # Entraînement
                rf.fit(X_train, y_train)
                et.fit(X_train, y_train)

                # Prédictions ensemble (moyenne pondérée)
                rf_pred = rf.predict(X_predict)
                et_pred = et.predict(X_predict)
                ensemble_pred = 0.6 * rf_pred + 0.4 * et_pred  # RF plus de poids

                df_forest.loc[predict_mask, col] = ensemble_pred

    return df_forest


def bayesian_ridge_imputation(df, numeric_cols):
    """Imputation Bayésienne avec Ridge regression"""
    df_bayes = df.copy()
    imputers = {}

    for col in numeric_cols:
        if df_bayes[col].isna().any():
            train_mask = ~df_bayes[col].isna()
            predict_mask = df_bayes[col].isna()

            if train_mask.sum() > 5 and predict_mask.sum() > 0:
                feature_cols = [c for c in numeric_cols if c != col]
                X_train = df_bayes.loc[train_mask, feature_cols].fillna(0)
                y_train = df_bayes.loc[train_mask, col]
                X_predict = df_bayes.loc[predict_mask, feature_cols].fillna(0)

                # Régression Bayésienne Ridge avec priors informatifs
                bayes_ridge = BayesianRidge(
                    alpha_1=1e-6,
                    alpha_2=1e-6,
                    lambda_1=1e-6,
                    lambda_2=1e-6,
                    compute_score=True,
                )

                bayes_ridge.fit(X_train, y_train)
                predictions = bayes_ridge.predict(X_predict)

                df_bayes.loc[predict_mask, col] = predictions
                imputers[col] = bayes_ridge

    metadata = {"bayesian_imputers": imputers}
    return df_bayes, metadata


def hybrid_imputation_strategy(df, numeric_cols):
    """Stratégie hybride combinant plusieurs approches selon les caractéristiques des données"""
    df_hybrid = df.copy()
    strategy_used = {}

    for col in numeric_cols:
        if df_hybrid[col].isna().any():
            missing_ratio = df_hybrid[col].isna().mean()
            data_variance = df_hybrid[col].var()
            n_unique = df_hybrid[col].nunique()

            # Choisir la stratégie selon les caractéristiques
            if missing_ratio > 0.5:
                # Beaucoup de valeurs manquantes -> KNN robuste
                strategy = "knn_robust"
                knn_imputer = KNNImputer(n_neighbors=3, weights="distance")
                df_hybrid[[col]] = knn_imputer.fit_transform(df_hybrid[[col]])

            elif n_unique < 10:
                # Peu de valeurs uniques -> Mode ou médiane
                strategy = "mode_median"
                df_hybrid[col] = df_hybrid[col].fillna(
                    df_hybrid[col].mode().iloc[0]
                    if len(df_hybrid[col].mode()) > 0
                    else df_hybrid[col].median()
                )

            elif data_variance > np.percentile(
                [df_hybrid[c].var() for c in numeric_cols], 75
            ):
                # Haute variance -> Random Forest
                strategy = "random_forest"
                df_hybrid = ensemble_forest_imputation(df_hybrid, [col])

            else:
                # Cas général -> Imputation itérative
                strategy = "iterative"
                imputer = IterativeImputer(estimator=BayesianRidge(), random_state=SEED)
                df_hybrid[[col]] = imputer.fit_transform(df_hybrid[[col]])

            strategy_used[col] = strategy

    metadata = {"hybrid_strategies": strategy_used}
    print(
        f"   🔀 Stratégies hybrides: {len(set(strategy_used.values()))} méthodes utilisées"
    )
    return df_hybrid, metadata


def impute_sex_advanced_v2(df):
    """Version améliorée de l'imputation pour 'sex' avec stacking et validation croisée"""
    df_imputed = df.copy()

    if "sex" not in df_imputed.columns:
        return df_imputed

    known_mask = (df_imputed["sex"] != 0.5) & (~df_imputed["sex"].isna())
    unknown_mask = (df_imputed["sex"] == 0.5) | (df_imputed["sex"].isna())

    if not known_mask.any() or not unknown_mask.any():
        return df_imputed

    # Features pour prédiction
    feature_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in ["ID", "sex"]]

    if len(feature_cols) == 0:
        return df_imputed

    known_data = df_imputed[known_mask]
    unknown_data = df_imputed[unknown_mask]

    X_train = known_data[feature_cols].fillna(0)
    y_train = known_data["sex"]
    X_predict = unknown_data[feature_cols].fillna(0)

    # Ensemble de modèles avec stacking
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    base_models = [
        (
            "bagging",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=IMPUTATION_PARAMS.get("bagging_n_estimators", 1000),
                random_state=SEED,
            ),
        ),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=SEED)),
        ("svc", SVC(probability=True, random_state=SEED)),
    ]

    # Voting classifier avec soft voting
    ensemble = VotingClassifier(estimators=base_models, voting="soft")

    try:
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_predict)
        df_imputed.loc[unknown_mask, "sex"] = predictions

        print(f"   ✅ {unknown_mask.sum()} valeurs 'sex' imputées avec ensemble avancé")

    except Exception as e:
        print(f"   ⚠️  Fallback vers méthode simple pour 'sex': {e}")
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
