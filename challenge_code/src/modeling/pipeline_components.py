import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ..data.data_cleaning.imputer import AdvancedImputer
from ..data.data_cleaning.cleaner import ClipQuantiles
from ..config import PREPROCESSING


def get_preprocessing_pipeline(
    X_train_df: pd.DataFrame, strategy: str = "iterative"
) -> ColumnTransformer:
    """
    Crée un pipeline de prétraitement robuste qui DÉTECTE AUTOMATIQUEMENT les types de colonnes.
    """
    print(
        "\n[PREPROCESSING] Création de la pipeline de prétraitement (version robuste)..."
    )

    # --- DÉTECTION AUTOMATIQUE DES TYPES DE FEATURES ---

    # 1. Identifier les colonnes catégorielles connues
    categorical_features = [
        col for col in ["CENTER", "CENTER_GROUP"] if col in X_train_df.columns
    ]

    # 2. Pour le reste, détecter le type en fonction du nombre de valeurs uniques
    continuous_features = []
    discrete_features = []

    # Seuil pour distinguer discret de continu (ajustable)
    CONTINUOUS_THRESHOLD = 20

    # Consider all non-categorical columns as potential features. Previously
    # columns containing the substring 'count' were excluded which caused
    # numeric count features (e.g., cosmic_*_count) to be dropped by the
    # ColumnTransformer (remainder='drop'). Keep them and let detection
    # decide continuous vs discrete.
    potential_features = [
        col for col in X_train_df.columns if col not in categorical_features
    ]

    for col in potential_features:
        # Si la colonne est numérique et a beaucoup de valeurs uniques -> Continue
        if (
            pd.api.types.is_numeric_dtype(X_train_df[col])
            and X_train_df[col].nunique() > CONTINUOUS_THRESHOLD
            and not "count" in col
        ):
            continuous_features.append(col)
        # Sinon (binaire, ou peu de valeurs uniques, ou non-numérique) -> Discrète
        else:
            discrete_features.append(col)

    print(
        f"   -> Features auto-détectées : {len(continuous_features)} continues, {len(discrete_features)} discrètes, {len(categorical_features)} catégorielles."
    )
    print(continuous_features)

    # --- DÉFINITION DES PIPELINES DE TRANSFORMATION (pilotées par la config) ---

    clip_lower = PREPROCESSING.get("clip_quantiles", {}).get("lower", 0.01)
    clip_upper = PREPROCESSING.get("clip_quantiles", {}).get("upper", 0.99)
    scaler_choice = PREPROCESSING.get("numeric_scaler", "robust")
    scaler = StandardScaler() if scaler_choice == "standard" else RobustScaler()

    adv_kwargs = {}
    if strategy == "knn":
        adv_kwargs["n_neighbors"] = PREPROCESSING.get("knn", {}).get("n_neighbors", 4)

    # Pipeline pour les variables continues
    continuous_transformer = Pipeline(
        steps=[
            ("clip", ClipQuantiles(lower=clip_lower, upper=clip_upper)),
            ("imputer", AdvancedImputer(strategy=strategy, **adv_kwargs)),
            ("scaler", scaler),
        ]
    )

    # Pipeline pour les variables discrètes (toutes les autres)
    # SimpleImputer est sûr pour les colonnes binaires/numériques à faible cardinalité
    discrete_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    # Pipeline pour les variables catégorielles
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # --- ASSEMBLAGE FINAL DU COLUMNTRANSFORMER ---

    # On ne crée une étape que si des colonnes de ce type ont été trouvées
    transformers = []
    if continuous_features:
        transformers.append(("continuous", continuous_transformer, continuous_features))
    if discrete_features:
        transformers.append(("discrete", discrete_transformer, discrete_features))
    if categorical_features:
        transformers.append(
            ("categorical", categorical_transformer, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Ignore les colonnes non spécifiées (sécurité)
        verbose_feature_names_out=False,
    )

    # Activation de la sortie en DataFrame Pandas
    preprocessor.set_output(transform="pandas")

    print(
        "[PREPROCESSING] Pipeline de prétraitement prête (sortie DataFrame activée).\n"
    )
    return preprocessor
