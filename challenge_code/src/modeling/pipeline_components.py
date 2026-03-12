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
    Crée une pipeline de prétraitement robuste avec une imputation différenciée
    pour les features continues, binaires, ordinales et de comptage.
    """
    print("\n[PREPROCESSING] Création de la pipeline de prétraitement (avec imputation différenciée)...")

    

    categorical_features = [col for col in ["CENTER", "CENTER_GROUP"] if col in X_train_df.columns]


    continuous_features = list(PREPROCESSING["continuous_features"])
    
    # Handle auxiliary features for imputation guidance
    aux_cfg = PREPROCESSING.get("imputer_auxiliary_features", {})
    aux_prefix = aux_cfg.get("prefix", "__aux_impute__")
    aux_cols = []
    if aux_cfg.get("enabled", False):
        aux_cols = [col for col in X_train_df.columns if col.startswith(aux_prefix)]
        # Auxiliary columns are added to continuous features to be used by IterativeImputer
        # They help guide imputation through correlations
        continuous_features.extend(aux_cols)
        if aux_cols:
            print(f"   -> {len(aux_cols)} auxiliary columns will guide imputation: {aux_cols[:3]}{'...' if len(aux_cols) > 3 else ''}")

    # remove duplicates preserving order
    if continuous_features:
        seen = set()
        deduped = []
        for col in continuous_features:
            if col not in seen:
                seen.add(col)
                deduped.append(col)
        continuous_features = deduped
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
        if X_train_df[col].nunique() <= 2 and ('mut_' in col or '_missing' in col or '_altered' in col or 'any_' in col or 'has_' in col):
            binary_features.append(col)
        # Est-ce un comptage ?
        elif 'count' in col or 'num_' in col or 'total_mutations' in col:
            count_features.append(col)

        else:
            ordinal_features.append(col)

    print(f"   -> Features identifiées : {len(continuous_features)} continues, {len(binary_features)} binaires, {len(count_features)} comptages, {len(ordinal_features)} ordinales, {len(categorical_features)} catégorielles.")
    

    # Determine if we should use advanced imputation or just simple median fallback
    single_impute_mode = PREPROCESSING.get("single_imputation_mode", False)
    early_enabled = PREPROCESSING.get("early_imputation", {}).get("enabled", False)
    use_simple_fallback = single_impute_mode and early_enabled
    
    if use_simple_fallback:
        # Early imputation already handled clinical columns, but ratios may still have NaN
        # Use SimpleImputer as fallback to handle any remaining NaN (e.g., from division by zero)
        print("   -> Mode imputation unique: SimpleImputer(median) comme fallback pour les NaN résiduels.")
        continuous_steps = [
            ('clip', ClipQuantiles(lower=0.01, upper=0.99)),
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),  # Fallback for ratios with NaN
            ('scaler', RobustScaler()),
        ]
    else:
        continuous_steps = [
            ('clip', ClipQuantiles(lower=0.01, upper=0.99)),
            ('imputer', AdvancedImputer(strategy=strategy)),
            ('scaler', RobustScaler()),
        ]

    continuous_transformer = Pipeline(steps=continuous_steps)


    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True))
    ])
    

    count_ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)),
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
