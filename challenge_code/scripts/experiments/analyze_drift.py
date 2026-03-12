import pandas as pd
import numpy as np
from collections import Counter


def describe_df(df: pd.DataFrame, name: str):
    print(f"\n===== {name} shape={df.shape} =====")
    print(df.head(3))


def kolmogorov_smirnov(a: pd.Series, b: pd.Series) -> float:
    try:
        from scipy.stats import ks_2samp

        a = a.dropna().values
        b = b.dropna().values
        if len(a) < 10 or len(b) < 10:
            return np.nan
        return ks_2samp(a, b).statistic
    except Exception:
        return np.nan


def main():
    Xtr = pd.read_csv("datasets_processed/X_train_processed.csv")
    Xte = pd.read_csv("datasets_processed/X_test_processed.csv")

    # Remove non-features if present
    for c in ["ID", "CENTER_GROUP"]:
        if c in Xtr.columns:
            Xtr = Xtr.drop(columns=[c])
        if c in Xte.columns:
            Xte = Xte.drop(columns=[c])

    common = [c for c in Xtr.columns if c in Xte.columns]
    print(f"Common columns: {len(common)}")

    # Rank columns by drift using KS statistic and mean/std deltas
    rows = []
    for c in common:
        if Xtr[c].dtype.kind in "biufc" and Xte[c].dtype.kind in "biufc":
            ks = kolmogorov_smirnov(Xtr[c], Xte[c])
            d_mean = float(np.nan_to_num(Xte[c].mean() - Xtr[c].mean()))
            d_std = float(np.nan_to_num(Xte[c].std() - Xtr[c].std()))
            rows.append((c, ks, d_mean, d_std))
    drift = pd.DataFrame(
        rows, columns=["feature", "ks", "d_mean", "d_std"]
    ).sort_values(by=["ks", "d_mean"], ascending=False)

    print("\nTop 20 drifted features (KS):")
    print(drift.head(20).to_string(index=False))
    try:
        drift.to_csv("reports/drift_numeric_ks.csv", index=False)
    except Exception:
        pass

    # Binary coverage differences
    bin_rows = []
    for c in common:
        if set(pd.unique(Xtr[c].dropna())) <= {0, 1}:
            p_tr = Xtr[c].mean()
            p_te = Xte[c].mean()
            bin_rows.append((c, p_tr, p_te, p_te - p_tr))
    bin_diff = (
        pd.DataFrame(bin_rows, columns=["feature", "p_train", "p_test", "delta"])
        .assign(abs_delta=lambda d: d["delta"].abs())
        .sort_values("abs_delta", ascending=False)
    )
    print("\nTop 20 binary prevalence shifts:")
    print(bin_diff.head(20).to_string(index=False))
    try:
        bin_diff.to_csv("reports/drift_binary_prevalence.csv", index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
