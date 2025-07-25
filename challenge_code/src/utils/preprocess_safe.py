"""
Module de preprocessing simplifie pour eviter les dependances problematiques
"""

import pandas as pd
import numpy as np
import json
import re
import math
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def safe_add_myvariant_data(df: pd.DataFrame, fields_list: list) -> pd.DataFrame:
    """
    Version simplifiee d'ajout des donnees MyVariant
    """
    if not fields_list:
        return df

    print(f"  → Tentative d'ajout de {len(fields_list)} champs MyVariant...")

    try:
        # Essayer de charger directement le fichier JSONL
        required_cols = ["CHR", "START", "REF", "ALT"]

        # Séparer les lignes valides et invalides
        valid_df = df.dropna(subset=required_cols).copy()
        invalid_df = df[df[required_cols].isna().any(axis=1)].copy()

        if valid_df.empty:
            print("  → Aucune ligne valide pour MyVariant")
            return df

        # Calculer variant_id pour les lignes valides
        valid_df["variant_id"] = valid_df.apply(
            lambda row: f"chr{row['CHR']}:g.{int(row['START'])}{row['REF']}>{row['ALT']}",
            axis=1,
        )

        # Charger les annotations MyVariant depuis le fichier JSONL
        variant_data = {}
        jsonl_path = "datas/variant_data.jsonl"

        try:
            with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
                for line in jsonl_file:
                    if line.strip():
                        entry = json.loads(line.strip())
                        # Le format est directement {variant_id: data}
                        for variant_id, variant_data_content in entry.items():
                            variant_data[variant_id] = variant_data_content

            print(f"  → {len(variant_data)} variantes chargees depuis {jsonl_path}")

            # Ajouter les champs demandes
            for field in fields_list:
                col_name = "_".join(field)
                valid_df[col_name] = valid_df["variant_id"].apply(
                    lambda vid: extract_field_nested(
                        variant_data.get(vid, {}), field, None
                    )
                )
                print(
                    f"    • Champ {col_name}: {valid_df[col_name].notna().sum()} valeurs"
                )

            # Fusionner les donnees enrichies avec les lignes non traitees
            enriched_df = pd.concat([valid_df, invalid_df], sort=False).reset_index(
                drop=True
            )

            # Supprimer la colonne temporaire variant_id si elle existe
            if "variant_id" in enriched_df.columns:
                enriched_df = enriched_df.drop(columns=["variant_id"])

            return enriched_df

        except FileNotFoundError:
            print(f"  ⚠ Fichier {jsonl_path} non trouve")
            return df
        except Exception as e:
            print(f"  ⚠ Erreur lors du chargement MyVariant: {e}")
            return df

    except Exception as e:
        print(f"  ⚠ Erreur MyVariant generale: {e}")
        return df


def extract_field_nested(variant, field_path, default=None):
    """
    Parcourt le dictionnaire variant en suivant le chemin indique par field_path.
    """
    try:
        val = variant
        for key in field_path:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return default


def safe_create_one_hot(
    df, id_col="ID", ref_col="GENE", min_count=5, rare_label="gene_other"
):
    """
    Version simplifiee du one-hot encoding des genes
    """
    try:
        if ref_col not in df.columns or id_col not in df.columns:
            print(f"  ⚠ Colonnes manquantes: {id_col} ou {ref_col}")
            return pd.DataFrame({id_col: df[id_col].unique()})

        # Comptage des occurrences par gene
        freq = df[ref_col].value_counts()

        # Identifier les genes rares
        rare_genes = freq[freq < min_count].index

        # Remplacer les genes rares
        df_copy = df.copy()
        df_copy[f"{ref_col.lower()}_aggreg"] = df_copy[ref_col].apply(
            lambda g: rare_label if g in rare_genes else g
        )

        # Realiser le pivot / crosstab
        pivoted = pd.crosstab(df_copy[id_col], df_copy[f"{ref_col.lower()}_aggreg"])

        # Renommer les colonnes
        pivoted.columns = [f"{ref_col.lower()}_{str(col)}" for col in pivoted.columns]

        # Remettre l'index comme colonne
        pivoted.reset_index(inplace=True)

        return pivoted

    except Exception as e:
        print(f"  ⚠ Erreur one-hot encoding: {e}")
        # Fallback simple: juste compter les occurrences par gene
        try:
            gene_counts = df.groupby([id_col, ref_col]).size().reset_index(name="count")
            gene_pivot = gene_counts.pivot_table(
                index=id_col, columns=ref_col, values="count", fill_value=0
            )

            # Limiter aux genes les plus frequents
            gene_freq = df[ref_col].value_counts()
            frequent_genes = gene_freq[gene_freq >= min_count].index[
                :20
            ]  # Top 20 genes

            # Filtrer les colonnes
            available_genes = [g for g in frequent_genes if g in gene_pivot.columns]
            if available_genes:
                result = gene_pivot[available_genes].reset_index()
                result.columns = [id_col] + [f"gene_{g}" for g in available_genes]
                return result
            else:
                return pd.DataFrame({id_col: df[id_col].unique()})
        except Exception as e2:
            print(f"  ⚠ Erreur fallback: {e2}")
            return pd.DataFrame({id_col: df[id_col].unique()})


def safe_count_bases_per_id(df, id_col="ID", ref_col="REF"):
    """
    Version simplifiee du comptage des bases
    """
    try:
        if ref_col not in df.columns or id_col not in df.columns:
            print(f"  ⚠ Colonnes manquantes: {id_col} ou {ref_col}")
            return pd.DataFrame({id_col: df[id_col].unique()})

        # Copie et nettoyage
        df_copy = df.copy()
        df_copy[ref_col] = df_copy[ref_col].astype(str).fillna("")

        # Grouper par ID et concatener les sequences
        grouped_strings = df_copy.groupby(id_col)[ref_col].apply(lambda x: "".join(x))

        # Fonction de comptage des bases
        def count_bases_simple(sequence):
            return pd.Series(
                {
                    f"{ref_col.lower()}_A": sequence.count("A"),
                    f"{ref_col.lower()}_G": sequence.count("G"),
                    f"{ref_col.lower()}_C": sequence.count("C"),
                    f"{ref_col.lower()}_T": sequence.count("T"),
                }
            )

        # Appliquer le comptage
        base_counts = grouped_strings.apply(count_bases_simple).reset_index()

        # S'assurer que tous les IDs sont representes
        distinct_ids = df_copy[[id_col]].drop_duplicates()
        result = distinct_ids.merge(base_counts, on=id_col, how="left")

        # Remplir les NaN par 0
        base_cols = [
            f"{ref_col.lower()}_A",
            f"{ref_col.lower()}_G",
            f"{ref_col.lower()}_C",
            f"{ref_col.lower()}_T",
        ]
        result[base_cols] = result[base_cols].fillna(0)

        return result

    except Exception as e:
        print(f"  ⚠ Erreur comptage bases: {e}")
        return pd.DataFrame({id_col: df[id_col].unique()})


def safe_parse_cytogenetics_v3(df, column_name="CYTOGENETICS"):
    """
    Version simplifiee du parsing cytogenetique
    """
    try:
        if column_name not in df.columns:
            print(f"  ⚠ Colonne {column_name} non trouvee")
            return df

        df_result = df.copy()

        # Features basiques extraites
        cyto_col = df_result[column_name].astype(str).fillna("")

        # Flags binaires
        df_result["cyto_has_complex"] = cyto_col.str.contains(
            "complex", case=False, na=False
        ).astype(int)
        df_result["cyto_has_del"] = cyto_col.str.contains(
            "del", case=False, na=False
        ).astype(int)
        df_result["cyto_has_t"] = cyto_col.str.contains(
            "t\\(", case=False, na=False
        ).astype(int)
        df_result["cyto_has_inv"] = cyto_col.str.contains(
            "inv", case=False, na=False
        ).astype(int)
        df_result["cyto_has_dup"] = cyto_col.str.contains(
            "dup", case=False, na=False
        ).astype(int)
        df_result["cyto_has_add"] = cyto_col.str.contains(
            "add", case=False, na=False
        ).astype(int)

        # Features numeriques
        df_result["cyto_length"] = cyto_col.str.len()
        df_result["cyto_num_commas"] = cyto_col.str.count(",")
        df_result["cyto_num_slashes"] = cyto_col.str.count("/")
        df_result["cyto_num_brackets"] = cyto_col.str.count("\\[")

        # Extraction du sexe
        def extract_sex(cyto_str):
            cyto_lower = cyto_str.lower()
            if "xy" in cyto_lower and "xx" not in cyto_lower:
                return 1.0  # Masculin
            elif "xx" in cyto_lower and "xy" not in cyto_lower:
                return 0.0  # Feminin
            else:
                return 0.5  # Indetermine

        df_result["sex"] = cyto_col.apply(extract_sex)

        # Supprimer la colonne originale
        df_result = df_result.drop(columns=[column_name])

        print(
            f"  → {len([c for c in df_result.columns if 'cyto' in c or c == 'sex'])} features cytogenetiques extraites"
        )

        return df_result

    except Exception as e:
        print(f"  ⚠ Erreur parsing cytogenetique: {e}")
        return df


def safe_log_transform(df, col):
    """
    Transformation logarithmique securisee
    """
    try:
        if col not in df.columns:
            print(f"  ⚠ Colonne {col} non trouvee pour transformation log")
            return pd.Series(0, index=df.index)
        return np.log(df[col] + 1)
    except Exception as e:
        print(f"  ⚠ Erreur transformation log pour {col}: {e}")
        return pd.Series(0, index=df.index)


def safe_process_outliers(df, threshold=0.05, multiplier=1.5):
    """
    Version simplifiee du traitement des outliers
    """
    try:
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns

        outliers_treated = 0
        for col in numeric_cols:
            if df_out[col].notna().sum() < 10:  # Pas assez de données
                continue

            Q1 = df_out[col].quantile(threshold)
            Q3 = df_out[col].quantile(1 - threshold)
            IQR = Q3 - Q1

            if IQR == 0:  # Pas de variabilité
                continue

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Compter les outliers avant traitement
            outliers_before = (
                (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
            ).sum()

            # Appliquer le clipping
            df_out[col] = df_out[col].clip(lower_bound, upper_bound)

            if outliers_before > 0:
                outliers_treated += 1

        print(f"  → Outliers traités pour {outliers_treated} colonnes")
        return df_out

    except Exception as e:
        print(f"  ⚠ Erreur traitement outliers: {e}")
        return df
