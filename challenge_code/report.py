# Supposons que final_df est votre dataframe final avec toutes les features
print("===== FEATURE SANITY CHECK REPORT =====")


import pandas as pd

# Créer une liste pour stocker les résultats


final_df_path = "datasets_processed\X_train_processed.csv"
final_df = pd.read_csv(final_df_path)

# Afficher le nombre de patients par centre à partir des colonnes one-hot CENTER_xxx
center_cols = [col for col in final_df.columns if col.startswith("CENTER_")]
if center_cols:
    print("\n===== NOMBRE DE PATIENTS PAR CENTRE (one-hot) =====")
    for col in sorted(center_cols):
        center_name = col.replace("CENTER_", "")
        count = int(final_df[col].sum())
        print(f"{center_name}: {count}")
    print()

feature_stats = []
for col in final_df.columns:
    if col in ["ID", "OS_YEARS", "OS_STATUS"]:
        continue

    stats = {"feature": col, "dtype": final_df[col].dtype}

    # Pour les features binaires ou catégorielles à faible cardinalité
    if final_df[col].nunique() < 10:
        counts = final_df[col].value_counts(normalize=True, dropna=False)
        stats["value_counts (%)"] = {k: f"{v:.2%}" for k, v in counts.to_dict().items()}
        # Vérifier la variance nulle (si une seule valeur existe)
        if len(counts) == 1:
            stats["warning"] = "ZERO VARIANCE"

    # Pour les features numériques
    else:
        desc = final_df[col].describe()
        stats["mean"] = f"{desc['mean']:.2f}"
        stats["std"] = f"{desc['std']:.2f}"
        stats["min"] = f"{desc['min']:.2f}"
        stats["max"] = f"{desc['max']:.2f}"

    # Compter les valeurs manquantes
    stats["missing_percent"] = f"{final_df[col].isna().mean():.2%}"

    feature_stats.append(stats)

report_df = pd.DataFrame(feature_stats)

# Afficher le rapport de manière lisible
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
            "\n\n=== ATTENTION : Features à variance nulle (devraient être supprimées) ==="
        )
        print(zero_variance_cols)
    else:
        print("\n\n=== INFO : Aucune feature à variance nulle détectée. Excellent. ===")
else:
    print("\n\n=== INFO : Aucune feature à variance nulle détectée. Excellent. ===")
