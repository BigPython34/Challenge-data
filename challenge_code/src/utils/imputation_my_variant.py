import pandas as pd
import json

print("=== SCRIPT D'IMPUTATION MYVARIANT AVEC COMPARAISONS ===")

# Charger les fichiers
print("Chargement des fichiers...")
try:
    train_df = pd.read_csv("datas/X_train/molecular_train.csv")
    test_df = pd.read_csv("datas/X_test/molecular_test.csv")
    print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")
except Exception as e:
    print(f"❌ Erreur chargement: {e}")
    exit(1)


def simplify_effect(effect):
    if pd.isna(effect):
        return effect
    effect = str(effect)
    if effect in [
        "splice_acceptor_variant&intron_variant",
        "splice_donor_variant&intron_variant",
        "splice_region_variant&intron_variant",
    ]:
        return "splice_site_variant"
    if effect == "missense_variant&splice_region_variant":
        return "missense_variant"
    if effect == "splice_region_variant&synonymous_variant":
        # Choix : focus sur splice_site_variant (adapter si besoin)
        return "splice_site_variant"
    return effect


for df in [train_df, test_df]:
    if "EFFECT" in df.columns:
        df["EFFECT"] = df["EFFECT"].apply(simplify_effect)

# Compteurs
imputation_counts = {"GENE": 0, "PROTEIN_CHANGE": 0, "EFFECT": 0, "VAF": 0, "DEPTH": 0}
comparison_counts = {
    field: {"same": 0, "different": 0, "total_existing": 0}
    for field in imputation_counts.keys()
}
different_examples = {field: [] for field in imputation_counts.keys()}
# Ensemble pour stocker les types de différences uniques
unique_differences = {field: set() for field in imputation_counts.keys()}

# Charger MyVariant
print("Chargement des données MyVariant...")
try:
    variant_data = {}
    with open("datas/variant_data_bis.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                vid, data = list(record.items())[0]
                variant_data[vid] = data
    print(f"✓ {len(variant_data)} variants chargés")
except Exception as e:
    print(f"❌ Erreur MyVariant: {e}")
    exit(1)


# Générer variant_id
def make_variant_id(row):
    try:
        return f"chr{row['CHR']}:g.{int(row['START'])}{row['REF']}>{row['ALT']}"
    except:
        return None


for df in [train_df, test_df]:
    if "PROTEIN_CHANGE" in df.columns:
        df["PROTEIN_CHANGE"] = (
            df["PROTEIN_CHANGE"].astype(str).str.replace(r"X$", "*", regex=True)
        )
    df["variant_id"] = df.apply(make_variant_id, axis=1)


# Fonction de comparaison
def compare_values(field_name, existing_value, myvariant_value, row_id):
    if pd.notna(existing_value):
        comparison_counts[field_name]["total_existing"] += 1

        if field_name in ["VAF", "DEPTH"]:
            # Comparaison numérique avec tolérance
            if isinstance(myvariant_value, (int, float)) and isinstance(
                existing_value, (int, float)
            ):
                tolerance = 0.01 if field_name == "VAF" else 1.0
                if abs(existing_value - myvariant_value) <= tolerance:
                    comparison_counts[field_name]["same"] += 1
                else:
                    comparison_counts[field_name]["different"] += 1
                    # Créer une clé unique pour ce type de différence
                    diff_key = f"{existing_value}->{myvariant_value}"
                    if diff_key not in unique_differences[field_name]:
                        unique_differences[field_name].add(diff_key)
                        if (
                            len(different_examples[field_name]) < 20
                        ):  # Augmenter la limite
                            different_examples[field_name].append(
                                {
                                    "ID": row_id,
                                    "existing": existing_value,
                                    "myvariant": myvariant_value,
                                }
                            )
            else:
                comparison_counts[field_name]["different"] += 1
                # Type de différence: incompatible types
                diff_key = f"type_mismatch:{type(existing_value).__name__}->{type(myvariant_value).__name__}"
                if diff_key not in unique_differences[field_name]:
                    unique_differences[field_name].add(diff_key)
                    if len(different_examples[field_name]) < 20:
                        different_examples[field_name].append(
                            {
                                "ID": row_id,
                                "existing": existing_value,
                                "myvariant": myvariant_value,
                            }
                        )
        else:
            # Comparaison textuelle
            if (
                str(existing_value).strip().lower()
                == str(myvariant_value).strip().lower()
            ):
                comparison_counts[field_name]["same"] += 1
            else:
                comparison_counts[field_name]["different"] += 1
                # Créer une clé unique pour ce type de différence textuelle
                diff_key = (
                    f"{str(existing_value).strip()}->{str(myvariant_value).strip()}"
                )
                if diff_key not in unique_differences[field_name]:
                    unique_differences[field_name].add(diff_key)
                    if len(different_examples[field_name]) < 20:  # Augmenter la limite
                        different_examples[field_name].append(
                            {
                                "ID": row_id,
                                "existing": existing_value,
                                "myvariant": myvariant_value,
                            }
                        )


# Fonction d'imputation
protein_imputation_report = []


def fill_missing_values(row):
    vid = row["variant_id"]
    data = variant_data.get(vid)

    if data is None:
        return row

    snpeff = data.get("snpeff", {}) or {}
    ann_raw = snpeff.get("ann", [])

    if isinstance(ann_raw, list) and len(ann_raw) > 0:
        ann = ann_raw[0]
    elif isinstance(ann_raw, dict):
        ann = ann_raw
    else:
        ann = {}

    vcf = data.get("vcf", {}) or {}

    # GENE
    if "gene_name" in ann:
        myvariant_gene = ann["gene_name"]
        if pd.isna(row.get("GENE")):
            row["GENE"] = myvariant_gene
            imputation_counts["GENE"] += 1
        else:
            compare_values("GENE", row.get("GENE"), myvariant_gene, row.get("ID"))

    # PROTEIN_CHANGE
    if "hgvs_p" in ann:
        myvariant_protein = ann["hgvs_p"]
        if pd.isna(row.get("PROTEIN_CHANGE")):
            row["PROTEIN_CHANGE"] = myvariant_protein
            imputation_counts["PROTEIN_CHANGE"] += 1
        else:
            # Ajoute au rapport si différent et si l'un des deux est abrégé et l'autre long
            existing = row.get("PROTEIN_CHANGE")
            if existing != myvariant_protein:
                # On considère abrégé si 1 lettre, long si 3 lettres
                import re

                def is_abbrev(s):
                    m = re.match(r"p\.([A-Z])\d+[A-Z\*]$", str(s))
                    return bool(m)

                def is_long(s):
                    m = re.match(r"p\.([A-Za-z]{3})\d+([A-Za-z]{3}|\*|X)$", str(s))
                    return bool(m)

                if (is_abbrev(existing) and is_long(myvariant_protein)) or (
                    is_long(existing) and is_abbrev(myvariant_protein)
                ):
                    protein_imputation_report.append(
                        (row.get("ID"), existing, myvariant_protein)
                    )
            compare_values(
                "PROTEIN_CHANGE",
                existing,
                myvariant_protein,
                row.get("ID"),
            )

    # EFFECT
    if "effect" in ann:
        myvariant_effect = ann["effect"]
        if pd.isna(row.get("EFFECT")):
            row["EFFECT"] = myvariant_effect
            imputation_counts["EFFECT"] += 1
        else:
            compare_values("EFFECT", row.get("EFFECT"), myvariant_effect, row.get("ID"))

    # VAF
    if isinstance(vcf.get("freq"), (float, int)):
        myvariant_vaf = vcf["freq"] / 100
        if pd.isna(row.get("VAF")):
            row["VAF"] = myvariant_vaf
            imputation_counts["VAF"] += 1
        else:
            compare_values("VAF", row.get("VAF"), myvariant_vaf, row.get("ID"))

    # DEPTH
    if isinstance(vcf.get("dp"), (float, int)):
        myvariant_depth = vcf["dp"]
        if pd.isna(row.get("DEPTH")):
            row["DEPTH"] = myvariant_depth
            imputation_counts["DEPTH"] += 1
        else:
            compare_values("DEPTH", row.get("DEPTH"), myvariant_depth, row.get("ID"))

    return row


# Appliquer l'imputation
print("Application de l'imputation avec comparaisons...")
train_filled = train_df.apply(fill_missing_values, axis=1)
test_filled = test_df.apply(fill_missing_values, axis=1)

# Sauvegarder
train_filled.drop(columns=["variant_id"]).to_csv(
    "datas/X_train/molecular_train_filled.csv", index=False
)
test_filled.drop(columns=["variant_id"]).to_csv(
    "datas/X_test/molecular_test_filled.csv", index=False
)

# Résultats
print("\n" + "=" * 60)
print("RÉCAPITULATIF DES IMPUTATIONS")
print("=" * 60)
for field, count in imputation_counts.items():
    print(f"- {field}: {count} valeurs imputées")

print("\n" + "=" * 60)
print("COMPARAISON AVEC LES VALEURS EXISTANTES")
print("=" * 60)
for field, counts in comparison_counts.items():
    total = counts["total_existing"]
    same = counts["same"]
    different = counts["different"]

    if total > 0:
        same_pct = (same / total) * 100
        different_pct = (different / total) * 100

        print(f"\n{field}:")
        print(f"  Total valeurs existantes : {total}")
        print(f"  Identiques à MyVariant   : {same} ({same_pct:.1f}%)")
        print(f"  Différentes de MyVariant : {different} ({different_pct:.1f}%)")

        if different > 0 and different_examples[field]:
            print(
                f"  Exemples de différences ({len(unique_differences[field])} types uniques):"
            )
            for i, ex in enumerate(different_examples[field], 1):
                print(
                    f"    {i}. ID {ex['ID']}: '{ex['existing']}' vs '{ex['myvariant']}'"
                )
    else:
        print(f"\n{field}: Aucune valeur existante à comparer")

# Rapport détaillé
with open("datas/comparison_report.txt", "w", encoding="utf-8") as f:
    f.write("RAPPORT DE COMPARAISON MYVARIANT\n")
    f.write("=" * 50 + "\n\n")

    for field, counts in comparison_counts.items():
        total = counts["total_existing"]
        same = counts["same"]
        different = counts["different"]

        f.write(f"{field}:\n")
        f.write(f"  Total valeurs existantes : {total}\n")
        f.write(f"  Identiques à MyVariant   : {same}\n")
        f.write(f"  Différentes de MyVariant : {different}\n")

        if total > 0:
            same_pct = (same / total) * 100
            different_pct = (different / total) * 100
            f.write(f"  Pourcentage identiques   : {same_pct:.1f}%\n")
            f.write(f"  Pourcentage différentes  : {different_pct:.1f}%\n")
            f.write(
                f"  Types de différences uniques : {len(unique_differences[field])}\n"
            )

        if different_examples[field]:
            f.write(
                f"  Exemples de différences ({len(unique_differences[field])} types uniques):\n"
            )
            for ex in different_examples[field]:
                f.write(
                    f"    ID {ex['ID']}: '{ex['existing']}' vs '{ex['myvariant']}'\n"
                )

            # Ajouter la liste de tous les types de différences uniques
            f.write(f"  Tous les types de différences pour {field}:\n")
            for i, diff_type in enumerate(sorted(unique_differences[field]), 1):
                f.write(f"    {i}. {diff_type}\n")

        f.write("\n")


# Rapport d'imputation des protéines abrégé/long
if protein_imputation_report:
    with open("datas/protein_imputation_report.txt", "w", encoding="utf-8") as f:
        for i, (pid, abbr, longf) in enumerate(protein_imputation_report, 1):
            f.write(f"{i}. ID {pid}: '{abbr}' vs '{longf}'\n")
    print(
        f"✓ Rapport d'imputation protéique abrégé/long: datas/protein_imputation_report.txt"
    )

print(f"\n✓ Rapport détaillé sauvegardé: datas/comparison_report.txt")
print(
    f"✓ Données enrichies sauvegardées: molecular_train_filled.csv, molecular_test_filled.csv"
)
print("\n" + "=" * 60)
