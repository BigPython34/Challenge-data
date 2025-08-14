import pandas as pd
import numpy as np
from pathlib import Path


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    Xtr = pd.read_csv("datasets_processed/X_train_processed.csv")
    Xte = pd.read_csv("datasets_processed/X_test_processed.csv")

    # Keep only common feature columns
    drop = [c for c in ["ID", "CENTER_GROUP"] if c in Xtr.columns]
    Xtr = Xtr.drop(columns=drop, errors="ignore")
    Xte = Xte.drop(columns=drop, errors="ignore")

    common = [c for c in Xtr.columns if c in Xte.columns]
    Xtr = Xtr[common]
    Xte = Xte[common]

    rows = []
    for c in common:
        s_tr = Xtr[c]
        s_te = Xte[c]
        # numeric mean for comparison; fall back to mode frequency for non-numeric
        if np.issubdtype(s_tr.dtype, np.number) and np.issubdtype(
            s_te.dtype, np.number
        ):
            mean_tr = float(np.nanmean(s_tr))
            mean_te = float(np.nanmean(s_te))
        else:
            mean_tr = float(
                s_tr.astype(str)
                .value_counts(normalize=True, dropna=False)
                .head(1)
                .values[0]
            )
            mean_te = float(
                s_te.astype(str)
                .value_counts(normalize=True, dropna=False)
                .head(1)
                .values[0]
            )
        rows.append(
            {
                "feature": c,
                "dtype": str(s_tr.dtype),
                "train_mean": mean_tr,
                "test_mean": mean_te,
                "abs_delta": abs(mean_te - mean_tr),
                "is_missing_indicator": c.endswith("_missing"),
            }
        )

    df = pd.DataFrame(rows).sort_values("abs_delta", ascending=False)
    df.to_csv(reports_dir / "mean_shift_overview.csv", index=False)

    # Save a focused view on missingness indicator columns
    miss_df = df[df["is_missing_indicator"]].copy()
    miss_df.to_csv(reports_dir / "missingness_indicator_shift.csv", index=False)

    print("Saved reports to:")
    print(f" - {reports_dir / 'mean_shift_overview.csv'}")
    print(f" - {reports_dir / 'missingness_indicator_shift.csv'}")


if __name__ == "__main__":
    main()
