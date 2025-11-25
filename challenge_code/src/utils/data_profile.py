"""Utilities to manage the clinical profile indicator feature."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from ..config import (
    CLINICAL_NUMERIC_COLUMNS,
    DATA_PROFILE_STRATEGY,
    MODEL_DIR,
)

PROFILE_METADATA_PREFIX = "feature_subset"


def compute_profile_indicator(clinical_df: pd.DataFrame) -> pd.Series:
    """Return a binary Series (1 = fully imputed clinical profile)."""
    if clinical_df is None or clinical_df.empty:
        return pd.Series(dtype="int64")

    profile_col = DATA_PROFILE_STRATEGY.get("profile_column", "CLINICAL_PROFILE_IMPUTED")
    clinical_cols = DATA_PROFILE_STRATEGY.get("clinical_columns", CLINICAL_NUMERIC_COLUMNS)
    available_cols = [col for col in clinical_cols if col in clinical_df.columns]
    if not available_cols:
        indicator = pd.Series(0, index=clinical_df.index, dtype="int64")
    else:
        has_real_value = clinical_df[available_cols].notna().any(axis=1)
        indicator = (~has_real_value).astype("int64")
    indicator.name = profile_col
    return indicator


def annotate_profile_column(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame | None = None,
    dataset_name: str = "",
) -> pd.DataFrame:
    """Add/overwrite the binary clinical profile indicator column."""
    annotated = clinical_df.copy()
    indicator = compute_profile_indicator(clinical_df)
    if dataset_name:
        indicator = indicator.fillna(0)
    annotated[indicator.name] = indicator
    return annotated


def build_profile_masks(profile_series: pd.Series | None) -> Dict[str, pd.Series]:
    """Provide backward-compatible masks for dual-model logic (if ever re-enabled)."""
    if profile_series is None or profile_series.empty:
        return {}
    filled = profile_series.fillna(0).astype(int)
    return {
        "complete": filled == 0,
        "molecular_only": filled == 1,
        "fallback": filled == 0,
        "all": filled.notna(),
    }


def select_feature_subset(
    columns: Sequence[str],
    prefixes: Sequence[str] | None = None,
    exact: Sequence[str] | None = None,
) -> List[str]:
    """Filter a list of column names by prefix or explicit inclusion."""
    if not columns:
        return []
    prefixes = prefixes or []
    exact = exact or []
    selected: List[str] = []
    for col in columns:
        if col in exact:
            selected.append(col)
            continue
        if any(col.startswith(pref) for pref in prefixes):
            selected.append(col)
    # Ensure deterministic order
    return [c for c in columns if c in set(selected)]


def save_feature_subset(columns: Sequence[str], label: str, model_dir: str | None = None) -> str:
    """Persist the list of columns used for a specific profile-aware subset."""
    target_dir = model_dir or MODEL_DIR
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{PROFILE_METADATA_PREFIX}_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"columns": list(columns)}, f, indent=2)
    return path


def load_feature_subset(label: str, model_dir: str | None = None) -> List[str]:
    """Load the stored feature list for the requested subset label."""
    target_dir = model_dir or MODEL_DIR
    path = os.path.join(target_dir, f"{PROFILE_METADATA_PREFIX}_{label}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing feature subset metadata for '{label}'. Expected file: {path}."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("columns", [])
