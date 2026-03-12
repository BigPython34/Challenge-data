#!/usr/bin/env python3
"""
Fold heterogeneity analysis: quantify distribution drift per fold and relate it to fold IPCW performance.

What it does
- Loads processed training data (X_train_processed.csv, y_train_processed.csv)
- Builds the same CV strategy as training (GroupKFold by CENTER_GROUP if available; else KFold)
- For a baseline survival model (GradientBoosting by default), computes per-fold IPCW C-index
- For each fold, computes drift metrics between train and val:
    * Numeric: KS statistic, Wasserstein distance, mean/std deltas
    * Binary: prevalence shift (delta p)
    * Missingness: delta missing rate per column (if any NA) and via *_missing indicators
    * Group divergence: Jensen-Shannon divergence for CENTER_GROUP distribution (if present)
- Aggregates per-fold drift metrics and correlates them with fold IPCW
- Saves compact CSV reports under `reports/`

Outputs
- reports/fold_performance.csv
- reports/fold_drift_aggregates.csv
- reports/fold_drift_correlations.csv
- reports/fold_feature_drift_topN.csv (top drifted features per fold)
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Local config
try:
    from src.config import SEED, TAU, GRADIENT_BOOSTING_PARAMS
except (ImportError, ModuleNotFoundError):
    # Fallback for relative import issues
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.config import SEED, TAU, GRADIENT_BOOSTING_PARAMS


@dataclass
class FoldMetrics:
    fold: int
    ipcw: float
    n_train: int
    n_val: int
    center_js: float | None
    mean_ks: float
    median_ks: float
    top5_mean_ks: float
    mean_wass: float
    bin_abs_prevalence_sum: float
    mean_missing_delta: float


def is_binary(series: pd.Series) -> bool:
    vals = pd.unique(series.dropna())
    try:
        vals_set = set(pd.Series(vals).astype(float))
    except Exception:
        # If casting fails, it's not numeric-binary
        return False
    return len(vals_set) <= 2 and vals_set.issubset({0.0, 1.0})


def compute_center_js(train_groups: pd.Series, val_groups: pd.Series) -> float:
    # Jensen-Shannon distance between center distributions (symmetric, bounded)
    try:
        train_counts = train_groups.value_counts().sort_index()
        val_counts = val_groups.value_counts().sort_index()
        all_idx = train_counts.index.union(val_counts.index)
        p = train_counts.reindex(all_idx, fill_value=0).astype(float) + 1e-9
        q = val_counts.reindex(all_idx, fill_value=0).astype(float) + 1e-9
        p = (p / p.sum()).values
        q = (q / q.sum()).values
        return float(jensenshannon(p, q))
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return np.nan


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    # Robust JS distance computation with manual fallback
    try:
        return float(jensenshannon(p, q))
    except (FloatingPointError, ValueError, ZeroDivisionError):
        m = 0.5 * (p + q)

        def kl(a, b):
            a = np.clip(a, 1e-12, 1.0)
            b = np.clip(b, 1e-12, 1.0)
            return float(np.sum(a * np.log(a / b)))

        return float(np.sqrt(0.5 * (kl(p, m) + kl(q, m))))


def per_feature_drift_numeric(
    a: pd.Series, b: pd.Series
) -> Tuple[float, float, float, float]:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan, np.nan, np.nan)
    ks = ks_2samp(a, b).statistic
    wd = wasserstein_distance(a, b)
    d_mean = float(np.nan_to_num(b.mean() - a.mean()))
    d_std = float(np.nan_to_num(b.std(ddof=1) - a.std(ddof=1)))
    return ks, wd, d_mean, d_std


def main():
    os.makedirs("reports", exist_ok=True)

    # Load datasets
    X = pd.read_csv("datasets_processed/X_train_processed.csv")
    y_df = pd.read_csv("datasets_processed/y_train_processed.csv")

    # Build y structured array
    # Expect columns: OS_STATUS (1=event), OS_YEARS (duration)
    if not {"OS_STATUS", "OS_YEARS"}.issubset(y_df.columns):
        raise ValueError("y_train_processed.csv must contain OS_STATUS and OS_YEARS")
    y = Surv.from_arrays(
        event=y_df["OS_STATUS"].astype(bool).values,
        time=y_df["OS_YEARS"].astype(float).values,
    )

    groups = (
        X["CENTER_GROUP"].astype(str).values if "CENTER_GROUP" in X.columns else None
    )

    # Keep a copy of group labels for JS computation,
    # but drop non-features from X for modeling
    group_series = (
        X["CENTER_GROUP"].astype(str) if "CENTER_GROUP" in X.columns else None
    )
    for c in ["CENTER_GROUP", "ID"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Identify numeric vs binary features
    num_cols = [
        c
        for c in X.columns
        if np.issubdtype(X[c].dtype, np.number) and not is_binary(X[c])
    ]
    bin_cols = [c for c in X.columns if is_binary(X[c])]

    # Model: GradientBoosting with config params
    gb_params = dict(GRADIENT_BOOSTING_PARAMS)
    gb = GradientBoostingSurvivalAnalysis(random_state=SEED, **gb_params)

    # CV splitter
    N_SPLITS = 5
    if groups is not None:
        splitter = GroupKFold(n_splits=N_SPLITS)
        split_iter = splitter.split(X, groups=groups)
        cv_meta = {"type": "GroupKFold", "n_splits": N_SPLITS}
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X)
        cv_meta = {
            "type": "KFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
        }

    fold_metrics: List[FoldMetrics] = []
    top_rows: List[Dict] = []

    for i, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        centers_tr = group_series.iloc[tr_idx] if group_series is not None else None
        centers_va = group_series.iloc[va_idx] if group_series is not None else None

        # Train baseline model and compute IPCW
        try:
            gb.fit(X_tr, y_tr)
            preds = gb.predict(X_va)
            ipcw = float(concordance_index_ipcw(y_tr, y_va, preds, tau=TAU)[0])
        except (ValueError, RuntimeError):
            ipcw = np.nan

        # Compute drift metrics
        ks_list, wd_list = [], []
        for c in num_cols:
            ks, wd, d_mean, d_std = per_feature_drift_numeric(X_tr[c], X_va[c])
            if not np.isnan(ks):
                ks_list.append(ks)
                wd_list.append(wd)
            # Collect top drifted features by KS
        # Binary prevalence shift
        bin_abs_shift_sum = 0.0
        for c in bin_cols:
            p_tr = float(pd.to_numeric(X_tr[c], errors="coerce").mean())
            p_va = float(pd.to_numeric(X_va[c], errors="coerce").mean())
            if not (np.isnan(p_tr) or np.isnan(p_va)):
                bin_abs_shift_sum += abs(p_va - p_tr)

        # Missingness: raw NA deltas (should be low after preprocessing)
        miss_delta = []
        for c in X.columns:
            m_tr = float(X_tr[c].isna().mean())
            m_va = float(X_va[c].isna().mean())
            if m_tr > 0 or m_va > 0:
                miss_delta.append(abs(m_va - m_tr))
        mean_missing_delta = float(np.mean(miss_delta)) if miss_delta else 0.0

        # Center distribution JS divergence
        center_js = (
            compute_center_js(centers_tr, centers_va)
            if centers_tr is not None
            else np.nan
        )

        # Aggregate
        mean_ks = float(np.mean(ks_list)) if ks_list else np.nan
        median_ks = float(np.median(ks_list)) if ks_list else np.nan
        top5_mean_ks = (
            float(np.mean(sorted(ks_list, reverse=True)[:5]))
            if len(ks_list) >= 5
            else mean_ks
        )
        mean_wd = float(np.mean(wd_list)) if wd_list else np.nan

        fold_metrics.append(
            FoldMetrics(
                fold=i,
                ipcw=ipcw,
                n_train=len(tr_idx),
                n_val=len(va_idx),
                center_js=center_js,
                mean_ks=mean_ks,
                median_ks=median_ks,
                top5_mean_ks=top5_mean_ks,
                mean_wass=mean_wd,
                bin_abs_prevalence_sum=bin_abs_shift_sum,
                mean_missing_delta=mean_missing_delta,
            )
        )

        # Top drifted features - record top 20 by KS
        per_feat = []
        for c in num_cols:
            ks, wd, d_mean, d_std = per_feature_drift_numeric(X_tr[c], X_va[c])
            per_feat.append(
                {
                    "fold": i,
                    "feature": c,
                    "ks": ks,
                    "wasserstein": wd,
                    "mean_delta": d_mean,
                    "std_delta": d_std,
                }
            )
        if per_feat:
            per_feat_df = (
                pd.DataFrame(per_feat)
                .sort_values(["ks", "wasserstein"], ascending=False)
                .head(20)
            )
            top_rows.extend(per_feat_df.to_dict(orient="records"))

    # Save per-fold performance
    perf_df = pd.DataFrame([fm.__dict__ for fm in fold_metrics])
    perf_path = os.path.join("reports", "fold_performance.csv")
    perf_df.to_csv(perf_path, index=False)

    # Correlations between drift aggregates and IPCW
    agg_cols = [
        "center_js",
        "mean_ks",
        "median_ks",
        "top5_mean_ks",
        "mean_wass",
        "bin_abs_prevalence_sum",
        "mean_missing_delta",
    ]
    rows = []
    for col in agg_cols:
        s = perf_df[["ipcw", col]].dropna()
        if len(s) >= 3:
            pear = float(np.corrcoef(s["ipcw"], s[col])[0, 1])
            from scipy.stats import spearmanr

            rho, _ = spearmanr(s["ipcw"], s[col])
            rows.append(
                {
                    "metric": col,
                    "pearson": pear,
                    "spearman": float(rho),
                    "n": int(len(s)),
                }
            )
        else:
            rows.append(
                {"metric": col, "pearson": np.nan, "spearman": np.nan, "n": int(len(s))}
            )
    corr_df = pd.DataFrame(rows)
    corr_path = os.path.join("reports", "fold_drift_correlations.csv")
    corr_df.to_csv(corr_path, index=False)

    # Save top drifted features across folds
    if top_rows:
        top_df = pd.DataFrame(top_rows)
        top_path = os.path.join("reports", "fold_feature_drift_topN.csv")
        top_df.to_csv(top_path, index=False)

    # Save a compact meta report
    meta = {
        "cv": cv_meta,
        "n_features": int(X.shape[1]),
        "n_numeric": len(num_cols),
        "n_binary": len(bin_cols),
        "model": {
            "type": "GradientBoostingSurvivalAnalysis",
            "params": gb_params,
        },
        "tau": TAU,
        "seed": SEED,
        "outputs": {
            "performance_csv": perf_path,
            "correlations_csv": corr_path,
            "top_feature_drift_csv": top_path if top_rows else None,
        },
    }
    with open(
        os.path.join("reports", "fold_heterogeneity_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    for k, v in meta["outputs"].items():
        if v:
            print(f" - {k}: {v}")
    print("Done.")


if __name__ == "__main__":
    main()
