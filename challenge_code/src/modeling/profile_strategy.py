"""Profile-aware training and inference helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from ..config import DATA_PROFILE_STRATEGY
from ..utils.data_profile import (
    build_profile_masks,
    select_feature_subset,
    save_feature_subset,
    load_feature_subset,
)


@dataclass
class DatasetSlice:
    X: pd.DataFrame
    y: np.ndarray
    groups: Optional[np.ndarray]
    label: str


def get_profile_mode() -> str:
    return DATA_PROFILE_STRATEGY.get("mode", "single_model")


def get_profile_column() -> str:
    return DATA_PROFILE_STRATEGY.get("profile_column", "DATA_PROFILE")


def should_use_profile_feature() -> bool:
    mode = get_profile_mode()
    profile_feature_cfg = DATA_PROFILE_STRATEGY.get("profile_feature", {})
    return mode == "profile_feature" and profile_feature_cfg.get("enabled", False)


def _select_molecular_columns(columns: Iterable[str]) -> List[str]:
    cfg = DATA_PROFILE_STRATEGY.get("dual_model", {})
    prefixes = cfg.get("molecular_feature_prefixes", [])
    manual_cols = cfg.get("molecular_feature_columns", [])
    selected = select_feature_subset(list(columns), prefixes=prefixes, exact=manual_cols)
    if not selected:
        raise ValueError(
            "Dual-model strategy requires at least one molecular feature; selection was empty."
        )
    return selected


def prepare_training_subsets(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, DatasetSlice]:
    """Return the datasets to train depending on the configured strategy."""
    mode = get_profile_mode()
    profile_col = get_profile_column()
    keep_profile_feature = should_use_profile_feature()

    subsets: Dict[str, DatasetSlice] = {}

    if mode != "dual_model":
        features = X.copy()
        if not keep_profile_feature:
            features = features.drop(columns=[profile_col], errors="ignore")
        subsets["default"] = DatasetSlice(X=features, y=y, groups=groups, label="default")
        return subsets

    if profile_col not in X.columns:
        raise ValueError(
            "Dual-model strategy requires the profile column in processed datasets."
        )

    base_features = X.copy()
    if not keep_profile_feature:
        base_features = base_features.drop(columns=[profile_col], errors="ignore")

    # Complete model sees every feature (minus profile column if not required)
    subsets["complete"] = DatasetSlice(
        X=base_features.copy(),
        y=y,
        groups=groups,
        label="complete",
    )

    # Molecular specialist uses every patient but restricts to the configured molecular columns
    molecular_cols = _select_molecular_columns(base_features.columns)
    molecular_X = base_features[molecular_cols].copy()
    subsets["molecular"] = DatasetSlice(
        X=molecular_X,
        y=y,
        groups=groups,
        label="molecular",
    )

    return subsets


def record_feature_subset(label: str, columns: Iterable[str]) -> str:
    """Persist the exact feature ordering used for a subset."""
    return save_feature_subset(list(columns), label=label)


def align_features_for_subset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Align a dataframe to the feature list recorded for the subset."""
    wanted_cols = load_feature_subset(label)
    if not wanted_cols:
        raise FileNotFoundError(
            f"No feature subset metadata found for '{label}'. Run training first."
        )
    missing = [c for c in wanted_cols if c not in df.columns]
    if missing:
        df = df.copy()
        for col in missing:
            df[col] = 0
    return df.reindex(columns=wanted_cols)


def route_inference_rows(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return boolean masks per subset for inference."""
    profile_col = get_profile_column()
    if profile_col not in df.columns:
        raise ValueError(
            "Processed inference data must include the profile column for routing."
        )
    masks = build_profile_masks(df[profile_col])
    routed: Dict[str, pd.Series] = {}
    complete_mask = masks.get("complete")
    fallback_mask = masks.get("fallback")
    molecular_mask = masks.get("molecular_only")

    if complete_mask is not None:
        routed["complete"] = complete_mask.copy()
    if fallback_mask is not None:
        if "complete" in routed:
            routed["complete"] = routed["complete"] | fallback_mask
        else:
            routed["complete"] = fallback_mask.copy()
    if molecular_mask is not None:
        routed["molecular"] = molecular_mask.copy()

    # Ensure every row is assigned to at least one subset by falling back to the complete bucket
    assigned = None
    for mask in routed.values():
        assigned = mask if assigned is None else (assigned | mask)

    if assigned is None:
        raise ValueError("No routable samples detected for dual-model inference.")

    unassigned = ~assigned
    if unassigned.any():
        fallback_target = "complete" if "complete" in routed else next(iter(routed))
        if fallback_target in routed:
            routed[fallback_target] = routed[fallback_target] | unassigned
        else:
            routed[fallback_target] = unassigned

    return routed
