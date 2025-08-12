import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ..data.data_cleaning.imputer import AdvancedImputer
from ..data.data_cleaning.cleaner import ClipQuantiles


def get_preprocessing_pipeline(
    X_train_df: pd.DataFrame, strategy: str = "iterative"
) -> ColumnTransformer:
    # Cette fonction n'a pas besoin de changer.
    # Elle utilise les classes que nous venons de rendre compatibles.
    print(
        "\n[PREPROCESSING] Création de la pipeline de prétraitement (version finale compatible)..."
    )

    # Identification des features...
    all_columns = X_train_df.columns.tolist()
    categorical_features = [col for col in ["CENTER"] if col in all_columns]
    continuous_features = [
        "BM_BLAST",
        "WBC",
        "ANC",
        "MONOCYTES",
        "HB",
        "PLT",
        "chromosome_count",
        "neutrophil_ratio",
        "monocyte_ratio",
        "platelet_wbc_ratio",
        "blast_platelet_ratio",
        "log_WBC",
        "log_PLT",
        "log_ANC",
        "log_MONOCYTES",
        "vaf_max_TP53",
        "vaf_max_FLT3",
        "vaf_max_NPM1",
        "vaf_max_CEBPA",
        "vaf_max_DNMT3A",
        "total_mutations",
        "vaf_mean",
        "vaf_median",
        "vaf_max",
        "vaf_std",
        "high_vaf_ratio",
    ]
    continuous_features = [col for col in continuous_features if col in all_columns]
    discrete_features = [
        col
        for col in all_columns
        if col not in categorical_features and col not in continuous_features
    ]

    print(
        f"   -> Features identifiées : {len(continuous_features)} continues, {len(discrete_features)} discrètes, {len(categorical_features)} catégorielles."
    )

    # Définition des pipelines...
    continuous_transformer = Pipeline(
        steps=[
            ("clip", ClipQuantiles()),
            ("imputer", AdvancedImputer(strategy=strategy)),
            ("scaler", RobustScaler()),
        ]
    )

    discrete_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Assemblage final...
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, continuous_features),
            ("discrete", discrete_transformer, discrete_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor.set_output(transform="pandas")

    print(
        "[PREPROCESSING] Pipeline de prétraitement prête (sortie DataFrame activée).\n"
    )
    return preprocessor
