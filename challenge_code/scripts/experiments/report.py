# Supposons que final_df est votre dataframe final avec toutes les features
print("===== FEATURE SANITY CHECK REPORT =====")


import pandas as pd




# === Rapport pour X_train_processed.csv ===
def feature_sanity_report(final_df, dataset_name=""):

    # Only consider one-hot encoded center columns (numeric), exclude the string group label
    import pandas.api.types as ptypes

    center_cols = [
        col
        for col in final_df.columns
        if col.startswith("CENTER_")
        and col != "CENTER_GROUP"
        and ptypes.is_numeric_dtype(final_df[col])
    ]
    if center_cols:
        print(f"\n===== NOMBRE DE PATIENTS PAR CENTRE (one-hot) [{dataset_name}] =====")
        for col in sorted(center_cols):
            center_name = col.replace("CENTER_", "")
            count = int(final_df[col].sum())
            print(f"{center_name}: {count}")
        print()

    feature_stats = []
    import pandas.api.types as ptypes

    for col in final_df.columns:
        if col in ["ID", "OS_YEARS", "OS_STATUS", "CENTER_GROUP"]:
            continue

        s = final_df[col]
        stats = {"feature": col, "dtype": s.dtype}


        if s.nunique(dropna=False) < 10:
            counts = s.value_counts(normalize=True, dropna=False)
            stats["value_counts (%)"] = {k: f"{v:.2%}" for k, v in counts.items()}
            if len(counts) == 1:
                stats["warning"] = "ZERO VARIANCE"
        else:
            if ptypes.is_numeric_dtype(s):
                desc = s.describe()

                stats["mean"] = f"{float(desc.get('mean', 0)):.2f}"
                stats["std"] = f"{float(desc.get('std', 0)):.2f}"
                stats["min"] = f"{float(desc.get('min', 0)):.2f}"
                stats["max"] = f"{float(desc.get('max', 0)):.2f}"
            else:

                top = s.astype(str).value_counts(dropna=False).head(5)
                stats["top_values"] = top.to_dict()

        stats["missing_percent"] = f"{s.isna().mean():.2%}"
        feature_stats.append(stats)

    report_df = pd.DataFrame(feature_stats)


    print(f"\n===== FEATURE SANITY CHECK REPORT [{dataset_name}] =====")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(report_df)

    if "warning" in report_df.columns:
        zero_variance_cols = report_df[report_df["warning"] == "ZERO VARIANCE"][
            "feature"
        ].tolist()
        if zero_variance_cols:
            print(
                f"\n\n=== ATTENTION : Features à variance nulle (devraient être supprimées) [{dataset_name}] ==="
            )
            print(zero_variance_cols)
        else:
            print(
                f"\n\n=== INFO : Aucune feature à variance nulle détectée. Excellent. [{dataset_name}] ==="
            )
    else:
        print(
            f"\n\n=== INFO : Aucune feature à variance nulle détectée. Excellent. [{dataset_name}] ==="
        )


# Rapport pour train
final_df_train_path = "datasets_processed/X_train_processed.csv"
final_df_train = pd.read_csv(final_df_train_path)
feature_sanity_report(final_df_train, dataset_name="X_train_processed.csv")

# Rapport pour test
final_df_test_path = "datasets_processed/X_test_processed.csv"
final_df_test = pd.read_csv(final_df_test_path)
feature_sanity_report(final_df_test, dataset_name="X_test_processed.csv")
