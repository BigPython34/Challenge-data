# molecular_feature_engineering.py

import pandas as pd
from ...config import (
    ALL_IMPORTANT_GENES,
    ADVERSE_GENES,  # Assurez-vous d'importer cette liste
    GENE_PATHWAYS,
    MOLECULAR_FEATURE_TOGGLES,
    ELN_MOLECULAR_RISK_ENCODING,
    TP53_HIGH_VAF_THRESHOLD,
    MOLECULAR_VAF_THRESHOLDS,
    MOLECULAR_GENE_FREQ_FILTER,
    MOLECULAR_EXTERNAL_SCORES,
    COSMIC_TIER_FEATURES,
    DRIVER_LIKE_FEATURES,
    CYTO_MOLECULAR_CROSS,
)
from src.data.data_extraction.external_data_manager import ExternalDataManager
from typing import Optional, List
import os
from .pruning import _apply_redundancy_policy


class MolecularFeatureExtraction:
    """Creates molecular features based on ELN 2022 prognostic mutations."""

    @staticmethod
    def get_frequent_genes(
        train_maf: pd.DataFrame, test_maf: pd.DataFrame
    ) -> List[str]:
        """
        Détermine la liste des gènes à utiliser en se basant sur les fréquences combinées
        des jeux d'entraînement et de test, comme défini dans la configuration.
        """
        config = MOLECULAR_GENE_FREQ_FILTER or {}
        if not config.get("enabled", False):
            return ALL_IMPORTANT_GENES

        min_total = int(config.get("min_total_count", 5))

        train_counts = train_maf["GENE"].value_counts()
        test_counts = test_maf["GENE"].value_counts()
        total_counts = train_counts.add(test_counts, fill_value=0)

        keep = set(total_counts[total_counts >= min_total].index)
        important_genes = [g for g in ALL_IMPORTANT_GENES if g in keep]

        print(
            f"[FE Mol.] Filtrage: {len(important_genes)} gènes sur {len(ALL_IMPORTANT_GENES)} conservés (fréq >= {min_total})."
        )
        return important_genes

    @staticmethod
    def extract_molecular_risk_features(
        df: pd.DataFrame,
        maf_df: pd.DataFrame,
        important_genes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        # Note: La logique de filtrage est maintenant gérée dans `create_all_molecular_features`
        important_genes = important_genes or ALL_IMPORTANT_GENES

        if df is None or df.empty:
            return pd.DataFrame()

        molecular_df = pd.DataFrame({"ID": df["ID"].astype(str).unique()})

        # === BINARY MUTATION STATUS ===
        molecular_df = MolecularFeatureExtraction._extract_binary_mutations(
            molecular_df, maf_df, important_genes
        )

        # === VAF-BASED FEATURES ===
        # ON PASSE MAINTENANT la liste `important_genes`
        molecular_df = MolecularFeatureExtraction._extract_vaf_features(
            molecular_df, maf_df, important_genes
        )

        # === MUTATION TYPE CLASSIFICATION ===
        molecular_df = MolecularFeatureExtraction._extract_mutation_types(
            molecular_df, maf_df
        )

        # === PATHWAY-LEVEL ALTERATIONS ===
        # ON PASSE MAINTENANT le `maf_df` complet pour utiliser tous les gènes
        molecular_df = MolecularFeatureExtraction._extract_pathway_alterations(
            molecular_df, maf_df
        )

        # === CLINICALLY RELEVANT CO-MUTATIONS ===
        molecular_df = MolecularFeatureExtraction._extract_comutation_patterns(
            molecular_df
        )

        # === ELN 2022 MOLECULAR RISK CLASSIFICATION ===
        # Cette fonction est maintenant dynamique
        molecular_df = MolecularFeatureExtraction._calculate_eln2022_molecular_risk(
            molecular_df
        )

        return molecular_df

    @staticmethod
    def _extract_binary_mutations(molecular_df, maf_df, important_genes):
        # VOTRE CODE ORIGINAL EST CONSERVÉ
        if molecular_df is None or molecular_df.empty:
            return molecular_df
        molecular_df["ID"] = molecular_df["ID"].astype(str)
        if maf_df is None or maf_df.empty:
            for gene in important_genes:
                molecular_df[f"mut_{gene}"] = 0
            return molecular_df
        gene_col = next(
            (c for c in ["GENE", "Hugo_Symbol"] if c in maf_df.columns), None
        )
        if gene_col is None:
            for gene in important_genes:
                molecular_df[f"mut_{gene}"] = 0
            return molecular_df
        maf = maf_df.copy()
        maf["ID"] = maf["ID"].astype(str)
        piv = (
            maf.loc[maf[gene_col].isin(important_genes), ["ID", gene_col]]
            .dropna()
            .assign(val=1)
            .drop_duplicates()
            .pivot_table(index="ID", columns=gene_col, values="val", fill_value=0)
            .reset_index()
        )
        piv.columns = ["ID", *[f"mut_{c}" for c in piv.columns if c != "ID"]]
        molecular_df = molecular_df.merge(piv, on="ID", how="left")
        for g in important_genes:
            col = f"mut_{g}"
            if col not in molecular_df.columns:
                molecular_df[col] = 0
        gene_cols = [f"mut_{g}" for g in important_genes]
        molecular_df[gene_cols] = molecular_df[gene_cols].fillna(0).astype(int)
        return molecular_df

    @staticmethod
    def _extract_vaf_features(
        molecular_df: pd.DataFrame, maf_df: pd.DataFrame, important_genes: List[str]
    ) -> pd.DataFrame:
        """MODIFIÉE : Utilise `important_genes` pour ne traiter que les gènes pertinents."""
        if maf_df is None or maf_df.empty:
            return molecular_df
        maf = maf_df.copy()
        maf["ID"] = maf["ID"].astype(str)
        vaf_col = next(
            (c for c in ["VAF", "Variant_Allele_Frequency"] if c in maf.columns), None
        )
        if vaf_col is None:
            return molecular_df
        maf[vaf_col] = pd.to_numeric(maf[vaf_col], errors="coerce")

        vaf_thresholds = dict(MOLECULAR_VAF_THRESHOLDS)
        vaf_thresholds["TP53"] = TP53_HIGH_VAF_THRESHOLD

        # Ne traiter que les gènes qui ont des règles ET qui sont dans la liste filtrée
        genes_to_process = [gene for gene in vaf_thresholds if gene in important_genes]

        for gene in genes_to_process:
            gene_rows = maf[maf["GENE"] == gene]
            if not gene_rows.empty:
                max_vaf = gene_rows.groupby("ID")[vaf_col].max()
                molecular_df[f"vaf_max_{gene}"] = (
                    molecular_df["ID"].map(max_vaf).fillna(0.0)
                )

                thr = vaf_thresholds[gene]
                high = molecular_df[f"vaf_max_{gene}"] >= thr

                if gene == "FLT3":
                    itd_ids = set(
                        gene_rows[
                            gene_rows["EFFECT"]
                            .astype(str)
                            .str.contains(
                                "ITD|internal tandem duplication", case=False, na=False
                            )
                        ]["ID"]
                    )
                    high |= molecular_df["ID"].isin(itd_ids)

                molecular_df[f"{gene}_high_VAF"] = high.astype(int)
            else:
                molecular_df[f"vaf_max_{gene}"] = 0.0
                molecular_df[f"{gene}_high_VAF"] = 0
        return molecular_df

    @staticmethod
    def _extract_mutation_types(
        molecular_df: pd.DataFrame, maf_df: pd.DataFrame
    ) -> pd.DataFrame:
        # VOTRE CODE ORIGINAL EST CONSERVÉ INTÉGRALEMENT
        if "TP53" in maf_df["GENE"].values:
            tp53_patients = maf_df[maf_df["GENE"] == "TP53"]
            truncating_pattern = r"nonsense|frameshift|splice_site|stop_gained"
            tp53_truncating_ids = tp53_patients[
                tp53_patients["EFFECT"]
                .astype(str)
                .str.contains(truncating_pattern, case=False, na=False)
            ]["ID"].unique()
            molecular_df["TP53_truncating"] = (
                molecular_df["ID"].astype(str).isin(tp53_truncating_ids).astype(int)
            )
        else:
            molecular_df["TP53_truncating"] = 0
        if "CEBPA" in maf_df["GENE"].values:
            cebpa_counts = maf_df[maf_df["GENE"] == "CEBPA"]["ID"].value_counts()
            biallelic_ids = cebpa_counts[cebpa_counts >= 2].index
            molecular_df["CEBPA_biallelic"] = (
                molecular_df["ID"].astype(str).isin(biallelic_ids).astype(int)
            )
        else:
            molecular_df["CEBPA_biallelic"] = 0
        if "FLT3" in maf_df["GENE"].values:
            flt3_rows = maf_df[maf_df["GENE"] == "FLT3"]
            itd_ids = set(
                flt3_rows[
                    flt3_rows["EFFECT"]
                    .astype(str)
                    .str.contains(
                        r"ITD|internal tandem duplication", case=False, na=False
                    )
                ]["ID"]
            )
            tkd_ids = set(
                flt3_rows[
                    flt3_rows["PROTEIN_CHANGE"]
                    .astype(str)
                    .str.contains(r"D835|I836", case=False, na=False)
                ]["ID"]
            )
            molecular_df["FLT3_ITD"] = (
                molecular_df["ID"].astype(str).isin(itd_ids).astype(int)
            )
            molecular_df["FLT3_TKD"] = (
                molecular_df["ID"].astype(str).isin(tkd_ids).astype(int)
            )
        else:
            molecular_df["FLT3_ITD"], molecular_df["FLT3_TKD"] = 0, 0
        return molecular_df

    @staticmethod
    def _extract_comutation_patterns(molecular_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichie pour extraire des interactions de gènes complexes et des ratios de VAF.
        """
        print("[FE Mol.] Création des patterns de co-mutation...")
        df_out = molecular_df.copy()

        # --- Feature existante ---
        if "mut_NPM1" in df_out.columns and "mut_FLT3" in df_out.columns:
            df_out["NPM1_pos_FLT3_neg"] = (
                (df_out["mut_NPM1"] == 1) & (df_out["mut_FLT3"] == 0)
            ).astype(int)

        # --- NOUVEAU: Double-hit sur les gènes du Spliceosome ---
        spliceosome_genes = GENE_PATHWAYS.get("RNA_splicing", [])
        spliceosome_cols = [
            f"mut_{g}" for g in spliceosome_genes if f"mut_{g}" in df_out.columns
        ]
        if len(spliceosome_cols) > 1:
            df_out["spliceosome_hit_count"] = df_out[spliceosome_cols].sum(axis=1)
            df_out["double_hit_spliceosome"] = (
                df_out["spliceosome_hit_count"] >= 2
            ).astype(int)

        # --- NOUVEAU: Triple-hit sur les gènes Epigénétiques (D/T/I) ---
        # On vérifie la présence d'au moins une mutation dans chaque sous-groupe
        has_dnmt3a = df_out.get("mut_DNMT3A", pd.Series(0, index=df_out.index))
        has_tet2 = df_out.get("mut_TET2", pd.Series(0, index=df_out.index))

        idh_cols = [c for c in ["mut_IDH1", "mut_IDH2"] if c in df_out.columns]
        has_idh = (
            (df_out[idh_cols].sum(axis=1) > 0).astype(int)
            if idh_cols
            else pd.Series(0, index=df_out.index)
        )

        df_out["epigenetic_hit_count"] = has_dnmt3a + has_tet2 + has_idh
        df_out["triple_hit_epigenetic"] = (df_out["epigenetic_hit_count"] >= 3).astype(
            int
        )

        # --- NOUVEAU: Ratio des VAF FLT3 / NPM1 ---
        if "vaf_max_FLT3" in df_out.columns and "vaf_max_NPM1" in df_out.columns:
            npm1_vaf = df_out["vaf_max_NPM1"]
            flt3_vaf = df_out["vaf_max_FLT3"]

            # Le ratio n'a de sens que si les deux mutations sont présentes
            df_out["vaf_ratio_FLT3_NPM1"] = (flt3_vaf / (npm1_vaf + 1e-6)).where(
                (npm1_vaf > 0) & (flt3_vaf > 0), 0
            )

        return df_out

    @staticmethod
    def _extract_pathway_alterations(
        molecular_df: pd.DataFrame, maf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        MODIFIÉE : Calcule les flags binaires et le score de diversité des pathways.
        """
        print("[FE Mol.] Création des features de pathways...")
        df_out = molecular_df.copy()

        toggles = MOLECULAR_FEATURE_TOGGLES.get(
            "pathway_features", {"binary": True, "count": False}
        )
        pathway_altered_cols = []

        for pathway_name, pathway_genes in GENE_PATHWAYS.items():
            pathway_maf = maf_df[maf_df["GENE"].isin(pathway_genes)]
            if not pathway_maf.empty:
                if toggles.get("binary", True):
                    altered_ids = pathway_maf["ID"].unique()
                    col_name = f"{pathway_name}_altered"
                    df_out[col_name] = df_out["ID"].isin(altered_ids).astype(int)
                    pathway_altered_cols.append(
                        col_name
                    )  # On garde en mémoire le nom de la colonne

                if toggles.get("count", False):
                    pathway_counts = pathway_maf.groupby("ID").size()
                    df_out[f"{pathway_name}_count"] = df_out["ID"].map(pathway_counts)

        # --- NOUVEAU: Score de Diversité des Pathways ---
        if pathway_altered_cols:
            df_out["pathway_diversity_score"] = df_out[pathway_altered_cols].sum(axis=1)

        return df_out

    @staticmethod
    def _calculate_eln2022_molecular_risk(molecular_df: pd.DataFrame) -> pd.DataFrame:
        """MODIFIÉE : Dynamique pour gérer les gènes filtrés."""
        favorable = (molecular_df.get("mut_NPM1", 0) == 1).astype(int)

        adverse_cols_present = [
            f"mut_{g}" for g in ADVERSE_GENES if f"mut_{g}" in molecular_df.columns
        ]
        adverse = (
            (molecular_df[adverse_cols_present].sum(axis=1) > 0).astype(int)
            if adverse_cols_present
            else pd.Series(0, index=molecular_df.index)
        )

        label = pd.Series(1, index=molecular_df.index)
        label = label.mask(favorable == 1, 0)
        label = label.mask(adverse == 1, 2)

        encoding = ELN_MOLECULAR_RISK_ENCODING or {}
        if encoding.get("encode_as") == "one_hot":
            molecular_df["eln_mol_favorable"] = (label == 0).astype(int)
            molecular_df["eln_mol_intermediate"] = (label == 1).astype(int)
            molecular_df["eln_mol_adverse"] = (label == 2).astype(int)
        else:
            weights = encoding.get("weights", {})
            mapping = {
                0: weights.get("favorable"),
                1: weights.get("intermediate"),
                2: weights.get("adverse"),
            }
            molecular_df["eln_mol_risk"] = label.map(mapping)
        return molecular_df

    @staticmethod
    def create_molecular_burden_features(maf_df: pd.DataFrame) -> pd.DataFrame:
        mutation_counts = (
            maf_df.groupby("ID").size().reset_index(name="total_mutations")
        )
        if "VAF" in maf_df.columns:
            maf = maf_df.copy()
            maf["VAF"] = pd.to_numeric(maf["VAF"], errors="coerce")
            vaf_stats = (
                maf.groupby("ID")["VAF"]
                .agg(
                    [
                        ("vaf_mean", "mean"),
                        ("vaf_median", "median"),
                        ("vaf_max", "max"),
                        ("vaf_std", "std"),
                    ]
                )
                .reset_index()
            )
            vaf_stats["vaf_std"] = vaf_stats["vaf_std"].fillna(0)
            high_vaf_counts = (
                maf[maf["VAF"] > 0.4]
                .groupby("ID")
                .size()
                .reset_index(name="high_vaf_mutations")
            )
        else:
            ids = mutation_counts["ID"].unique()
            vaf_stats = pd.DataFrame(
                {
                    "ID": ids,
                    "vaf_mean": 0.0,
                    "vaf_median": 0.0,
                    "vaf_max": 0.0,
                    "vaf_std": 0.0,
                }
            )
            high_vaf_counts = pd.DataFrame({"ID": ids, "high_vaf_mutations": 0})
        burden_df = mutation_counts.merge(vaf_stats, on="ID", how="left")
        burden_df = burden_df.merge(high_vaf_counts, on="ID", how="left")
        burden_df["high_vaf_mutations"] = (
            burden_df["high_vaf_mutations"].fillna(0).astype(int)
        )
        burden_df["high_vaf_ratio"] = (
            burden_df["high_vaf_mutations"] / burden_df["total_mutations"]
        ).fillna(0)
        return burden_df

    @staticmethod
    def get_favorable_molecular_mask(molecular_df: pd.DataFrame) -> pd.Series:
        # VOTRE CODE ORIGINAL EST CONSERVÉ INTÉGRALEMENT
        favorable_conditions = []
        if "mut_NPM1" in molecular_df.columns:
            npm1_favorable = (molecular_df["mut_NPM1"] == 1) & (
                molecular_df.get("mut_FLT3", 0) == 0
            )
            favorable_conditions.append(npm1_favorable)
        if "CEBPA_biallelic" in molecular_df.columns:
            favorable_conditions.append(molecular_df["CEBPA_biallelic"] == 1)
        return (
            pd.concat(favorable_conditions, axis=1).any(axis=1)
            if favorable_conditions
            else pd.Series(False, index=molecular_df.index)
        )

    @staticmethod
    def get_adverse_molecular_mask(molecular_df: pd.DataFrame) -> pd.Series:
        # VOTRE CODE ORIGINAL EST CONSERVÉ INTÉGRALEMENT
        adverse_conditions = []
        adverse_genes = ["TP53", "ASXL1", "RUNX1", "BCOR", "EZH2"]
        for gene in adverse_genes:
            if f"mut_{gene}" in molecular_df.columns:
                adverse_conditions.append(molecular_df[f"mut_{gene}"] == 1)
        if "FLT3_high_VAF" in molecular_df.columns:
            adverse_conditions.append(molecular_df["FLT3_high_VAF"] == 1)
        return (
            pd.concat(adverse_conditions, axis=1).any(axis=1)
            if adverse_conditions
            else pd.Series(False, index=molecular_df.index)
        )

    @staticmethod
    def _extract_impact_features(maf_df: pd.DataFrame) -> pd.DataFrame:
        if "IMPACT" not in maf_df.columns:
            return pd.DataFrame()
        impact_dummies = pd.get_dummies(
            maf_df["IMPACT"], prefix="impact", dummy_na=False
        )
        impact_with_id = pd.concat([maf_df["ID"], impact_dummies], axis=1)
        impact_features = impact_with_id.groupby("ID").sum()
        return impact_features.reset_index()

    @staticmethod
    def create_all_molecular_features(
        base_df: pd.DataFrame,
        maf_df: pd.DataFrame,
        important_genes: List[str],  # <--- ACCEPTE MAINTENANT LA LISTE PRÉ-CALCULÉE
        external_data_manager: ExternalDataManager,
    ) -> pd.DataFrame:
        """
        Orchestrateur qui crée toutes les features pour un jeu de données donné (train ou test)
        en utilisant une liste de gènes fixe pour garantir la cohérence.
        """
        # --- Étape 1: Enrichissement du MAF ---
        if not external_data_manager.gene_info_data.empty:
            maf_df = maf_df.merge(
                external_data_manager.get_gene_info(),
                left_on="GENE",
                right_index=True,
                how="left",
            ).fillna({"is_oncogene": 0, "is_tumor_suppressor": 0})
        # Ensure ID types align for all groupby/join operations (string keys)
        if "ID" in maf_df.columns:
            maf_df["ID"] = maf_df["ID"].astype(str)

        risk_features = MolecularFeatureExtraction.extract_molecular_risk_features(
            base_df, maf_df, important_genes=important_genes
        )

        burden_features = MolecularFeatureExtraction.create_molecular_burden_features(
            maf_df
        )
        impact_features = MolecularFeatureExtraction._extract_impact_features(maf_df)
        external_features = pd.DataFrame(index=base_df["ID"].unique())
        if "is_oncogene" in maf_df.columns:
            oncogene_counts = maf_df.groupby("ID")["is_oncogene"].sum()
            tsg_counts = maf_df.groupby("ID")["is_tumor_suppressor"].sum()
            external_features["num_oncogene_muts"] = external_features.index.map(
                oncogene_counts
            )
            external_features["num_tsg_muts"] = external_features.index.map(tsg_counts)
            external_features["any_oncogene_mut"] = (
                external_features["num_oncogene_muts"].fillna(0) > 0
            ).astype(int)
            external_features["any_tsg_mut"] = (
                external_features["num_tsg_muts"].fillna(0) > 0
            ).astype(int)

        # Agrégations génériques sur les colonnes COSMIC/MOLGEN si présentes
        cosmic_bool_cols = [
            c
            for c in maf_df.columns
            if (c.startswith("cosmic_") or c.startswith("molgen_"))
            and c != "cosmic_tier_min"
        ]
        cosmic_bool_cols = [
            c for c in cosmic_bool_cols if pd.api.types.is_numeric_dtype(maf_df[c])
        ]
        if cosmic_bool_cols:
            maf_cosmic = maf_df[["ID", *cosmic_bool_cols]].copy()
            maf_cosmic[cosmic_bool_cols] = maf_cosmic[cosmic_bool_cols].fillna(0)
            grp = maf_cosmic.groupby("ID")[cosmic_bool_cols]
            counts = grp.sum()
            any_flags = (counts > 0).astype(int)
            counts = counts.add_suffix("_count")
            any_flags = any_flags.add_prefix("any_")
            external_features = external_features.join(counts, how="left")
            external_features = external_features.join(any_flags, how="left")

        # ------------------------------------------------------------------
        # Robust COSMIC counts aggregation (guarantee presence of *_count cols)
        # We aggregate directly from the maf_df (qui a été enrichi avec
        # gene_info via external_data_manager) to avoid relying on earlier
        # selections that may have filtered numeric types.
        try:
            cosmic_flag_list = [
                "cosmic_is_fusion_gene",
                "cosmic_has_missense_common",
                "cosmic_has_nonsense_common",
                "cosmic_has_frameshift_common",
                "cosmic_has_translocation_common",
                "cosmic_has_splice_common",
                "cosmic_has_amplification_common",
                "cosmic_has_deletion_common",
                "cosmic_has_somatic_evidence",
                "cosmic_has_germline_evidence",
                "cosmic_in_aml",
                "cosmic_in_leukemia_lymphoma",
            ]
            present_flags = [c for c in cosmic_flag_list if c in maf_df.columns]
            if present_flags:
                mf = maf_df[["ID", *present_flags]].copy()
                # ensure numeric
                for c in present_flags:
                    mf[c] = pd.to_numeric(mf[c], errors="coerce").fillna(0).astype(int)
                flag_counts = mf.groupby("ID")[present_flags].sum()
                flag_counts = flag_counts.add_suffix("_count")
                # join counts (safe: external_features indexed by ID)
                external_features = external_features.join(flag_counts, how="left")
                # fill and cast
                for col in flag_counts.columns:
                    if col in external_features.columns:
                        external_features[col] = (
                            external_features[col].fillna(0).astype(int)
                        )

            # Also add a simple per-patient cosmic_gene_count: number of unique
            # mutated genes that are present in the combined gene info.
            gene_info = None
            try:
                gene_info = external_data_manager.get_gene_info()
            except Exception:
                gene_info = None
            if gene_info is not None and not gene_info.empty:
                # count unique mutated genes per patient that exist in gene_info
                tmp = maf_df[["ID", "GENE"]].dropna()
                tmp["GENE"] = tmp["GENE"].astype(str)
                cosmic_genes = set(gene_info.index)
                tmp = tmp[tmp["GENE"].isin(cosmic_genes)]
                if not tmp.empty:
                    cg = tmp.groupby("ID")["GENE"].nunique().rename("cosmic_gene_count")
                    external_features = external_features.join(cg, how="left")
                    external_features["cosmic_gene_count"] = (
                        external_features["cosmic_gene_count"].fillna(0).astype(int)
                    )
                else:
                    # ensure column exists (all zeros) for traceability
                    external_features["cosmic_gene_count"] = 0
        except Exception:
            # Non-blocking: keep pipeline running if any unexpected issue
            pass

        # cosmic_tier_min
        if "cosmic_tier_min" in maf_df.columns:
            tier_series = maf_df.groupby("ID")["cosmic_tier_min"].min()
            external_features["cosmic_min_tier_mut_genes"] = (
                external_features.index.map(tier_series)
            )
            if COSMIC_TIER_FEATURES.get("counts_by_tier", True):
                tmp = maf_df[["ID", "cosmic_tier_min"]].copy()
                tmp["tier_int"] = (
                    pd.to_numeric(tmp["cosmic_tier_min"], errors="coerce")
                    .round()
                    .astype("Int64")
                )
                tier_counts = tmp.dropna(subset=["tier_int"]).pivot_table(
                    index="ID",
                    columns="tier_int",
                    values="cosmic_tier_min",
                    aggfunc="count",
                    fill_value=0,
                )
                tier_counts.columns = [
                    f"tier{int(c)}_gene_count" for c in tier_counts.columns
                ]
                external_features = external_features.join(tier_counts, how="left")
            if COSMIC_TIER_FEATURES.get("add_has_cosmic_tier", True):
                external_features["has_cosmic_tier"] = (
                    ~external_features["cosmic_min_tier_mut_genes"].isna()
                ).astype(int)

        external_features = external_features.reset_index().rename(
            columns={"index": "ID"}
        )
        if COSMIC_TIER_FEATURES.get("keep_min_tier_na", True):
            num_cols = [
                c
                for c in external_features.columns
                if c not in {"ID", "cosmic_min_tier_mut_genes"}
            ]
            external_features[num_cols] = external_features[num_cols].fillna(0)
        else:
            external_features = external_features.fillna(0)

        # Driver-like features
        if DRIVER_LIKE_FEATURES.get("enabled", True) and "GENE" in maf_df.columns:
            m2 = maf_df.copy()
            # Robust EFFECT access and parsing
            eff = (
                m2.get("EFFECT", pd.Series(index=m2.index, dtype=object))
                .astype(str)
                .str.lower()
            )
            m2["is_missense_obs"] = eff.str.contains("missense", na=False).astype(int)
            m2["is_trunc_obs"] = eff.str.contains(
                r"nonsense|frameshift|splice|stop_gained|stop_lost|start_lost",
                na=False,
            ).astype(int)

            def _safe_bool_series(df, col: str) -> pd.Series:
                if col in df.columns:
                    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                return pd.Series(0, index=df.index, dtype=int)

            exp_miss = _safe_bool_series(m2, "cosmic_has_missense_common")
            exp_trunc = (
                _safe_bool_series(m2, "cosmic_has_nonsense_common")
                | _safe_bool_series(m2, "cosmic_has_frameshift_common")
                | _safe_bool_series(m2, "cosmic_has_splice_common")
            )
            onc = _safe_bool_series(m2, "is_oncogene")
            tsg = _safe_bool_series(m2, "is_tumor_suppressor")
            m2["oncogene_missense_match"] = (
                (onc == 1) & (exp_miss == 1) & (m2["is_missense_obs"] == 1)
            ).astype(int)
            m2["tsg_truncating_match"] = (
                (tsg == 1) & (exp_trunc == 1) & (m2["is_trunc_obs"] == 1)
            ).astype(int)
            dl_counts = m2.groupby("ID")[
                ["oncogene_missense_match", "tsg_truncating_match"]
            ].sum()
            external_features = (
                external_features.set_index("ID")
                .join(dl_counts, how="left")
                .reset_index()
            )
            for col in ["oncogene_missense_match", "tsg_truncating_match"]:
                if col in external_features.columns:
                    external_features[col] = (
                        external_features[col].fillna(0).astype(int)
                    )

        # Per-arm mutation flags
        if (
            CYTO_MOLECULAR_CROSS.get("enabled", False)
            and "cosmic_chr_arm" in maf_df.columns
        ):
            arms = CYTO_MOLECULAR_CROSS.get("arms", ["5q", "7q", "17p"])
            by_id_arm = (
                maf_df.dropna(subset=["cosmic_chr_arm"])
                .groupby("ID")["cosmic_chr_arm"]
                .apply(lambda s: set(s.astype(str)))
            )
            arm_map = (
                external_features.set_index("ID")
                .index.to_series()
                .map(by_id_arm)
                .apply(lambda x: x if isinstance(x, set) else set())
            )
            arm_flags = pd.DataFrame(index=arm_map.index)
            for a in arms:
                arm_flags[f"mutated_arm_{a}"] = arm_map.apply(
                    lambda s, _a=a: _a in s
                ).astype(int)
            external_features = (
                external_features.set_index("ID")
                .join(arm_flags, how="left")
                .reset_index()
            )
            for c in arm_flags.columns:
                external_features[c] = external_features[c].fillna(0).astype(int)

        # Variant-level pathogenicity scores (CADD)
        cadd_cfg = (
            MOLECULAR_EXTERNAL_SCORES.get("cadd", {})
            if MOLECULAR_EXTERNAL_SCORES
            else {}
        )
        if cadd_cfg.get("enabled", False):
            m3 = maf_df.copy()
            if cadd_cfg.get("snv_only", True):
                snv_mask = (m3.get("REF").astype(str).str.len() == 1) & (
                    m3.get("ALT").astype(str).str.len() == 1
                )
                if "END" in m3.columns and "START" in m3.columns:
                    snv_mask &= m3["START"].astype(str) == m3["END"].astype(str)
                m3 = m3.loc[snv_mask].copy()
            if {"CHR", "START", "REF", "ALT"}.issubset(m3.columns):
                # Normalize chromosome like in the cache fetch (remove leading 'chr')
                m3 = m3.copy()
                m3["CHR"] = (
                    m3["CHR"]
                    .astype(str)
                    .str.replace(r"^chr", "", regex=True)
                    .str.strip()
                )
                m3["hgv_id"] = m3.apply(
                    lambda r: f"chr{str(r['CHR']).strip()}:g.{int(r['START'])}{str(r['REF']).strip()}>{str(r['ALT']).strip()}",
                    axis=1,
                )
                scores = external_data_manager.get_variant_scores()
                m3["CADD_PHRED"] = m3["hgv_id"].map(scores)
                agg = m3.groupby("ID")["CADD_PHRED"].agg(["max", "mean"])
                agg.rename(
                    columns={"max": "cadd_max", "mean": "cadd_mean"}, inplace=True
                )
                high_thr = float(cadd_cfg.get("high_threshold", 20.0))
                high_count = (
                    (m3["CADD_PHRED"] >= high_thr)
                    .groupby(m3["ID"])
                    .sum()
                    .rename("cadd_high_count")
                )
                agg = agg.join(high_count, how="left")
                external_features = (
                    external_features.set_index("ID")
                    .join(agg, how="left")
                    .reset_index()
                )
                for c in ["cadd_max", "cadd_mean", "cadd_high_count"]:
                    if c in external_features.columns:
                        fillv = 0 if c.endswith("_count") else 0.0
                        external_features[c] = external_features[c].fillna(fillv)
        all_molecular_df = risk_features
        for df_to_merge in [burden_features, external_features, impact_features]:
            if not df_to_merge.empty:
                all_molecular_df = pd.merge(
                    all_molecular_df, df_to_merge, on="ID", how="outer"
                )

        all_molecular_df = all_molecular_df.fillna(0)
        all_molecular_df = _apply_redundancy_policy(all_molecular_df)

        print(
            f"[FE Mol.] Feature Engineering Moléculaire terminé. Shape: {all_molecular_df.shape}"
        )
        return all_molecular_df
