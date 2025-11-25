import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ..data.data_cleaning.imputer import AdvancedImputer
from ..data.data_cleaning.cleaner import ClipQuantiles
from ..config import PREPROCESSING, DATA_PROFILE_STRATEGY


def get_preprocessing_pipeline(
    X_train_df: pd.DataFrame, strategy: str = "iterative"
) -> ColumnTransformer:
    """
    Crée une pipeline de prétraitement robuste avec une imputation différenciée
    pour les features continues, binaires, ordinales et de comptage.
    """
    print("\n[PREPROCESSING] Création de la pipeline de prétraitement (avec imputation différenciée)...")


    

    categorical_features = [col for col in ["CENTER", "CENTER_GROUP"] if col in X_train_df.columns]
    profile_col = DATA_PROFILE_STRATEGY.get("profile_column")
    profile_cfg = DATA_PROFILE_STRATEGY.get("profile_feature", {})
    profile_enabled = profile_cfg.get("enabled") and profile_col and profile_col in X_train_df.columns
    treat_profile_as_cat = profile_enabled and profile_cfg.get("treat_as_categorical", True)
    if treat_profile_as_cat:
        categorical_features.append(profile_col)
    categorical_features = list(dict.fromkeys(categorical_features))


    continuous_features = PREPROCESSING["continuous_features"]
    continuous_features = [col for col in continuous_features if col in X_train_df.columns]

    # Identifier les autres types de colonnes
    binary_features = []
    count_features = []
    ordinal_features = []
    
    potential_discrete_features = [
        col for col in X_train_df.columns 
        if col not in categorical_features and col not in continuous_features
    ]
    
    for col in potential_discrete_features:
        # Est-ce une feature binaire (0/1) ? (y compris les _missing)
        if (
            X_train_df[col].nunique() <= 2
            and (
                'mut_' in col
                or '_missing' in col
                or '_altered' in col
                or 'any_' in col
                or 'has_' in col
                or (profile_enabled and not treat_profile_as_cat and col == profile_col)
            )
        ):
            binary_features.append(col)
        # Est-ce un comptage ?
        elif 'count' in col or 'num_' in col or 'total_mutations' in col:
            count_features.append(col)

        else:
            ordinal_features.append(col)

    print(f"   -> Features identifiées : {len(continuous_features)} continues, {len(binary_features)} binaires, {len(count_features)} comptages, {len(ordinal_features)} ordinales, {len(categorical_features)} catégorielles.")
    

    
    continuous_transformer = Pipeline(steps=[
        ('clip', ClipQuantiles(lower=0.01, upper=0.99)),
        ('imputer', AdvancedImputer(strategy=strategy)),
        ('scaler', RobustScaler())
    ])


    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    

    count_ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- 3. Assemblage final du ColumnTransformer ---
    transformers = []
    if continuous_features:
        transformers.append(("continuous", continuous_transformer, continuous_features))
    if binary_features:
        transformers.append(("binary", binary_transformer, binary_features))
    if count_features:
        transformers.append(("count", count_ordinal_transformer, count_features))
    if ordinal_features:
        transformers.append(("ordinal", count_ordinal_transformer, ordinal_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )
    preprocessor.set_output(transform="pandas")

    print("[PREPROCESSING] Pipeline prête (sortie DataFrame activée).\n")
    return preprocessor
