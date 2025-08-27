"""
Feature Engineering module for AML survival analysis.

All numeric thresholds, gene sets, toggles and encodings are configured in src.config
for full experiment traceability.
"""

import pandas as pd
from typing import Optional
from .clinical_feature_engineering import (
    ClinicalFeatureEngineering,
    CytogeneticFeatureExtraction,
)
from .molecular_feature_engineering import MolecularFeatureExtraction
from .pruning import _apply_redundancy_policy
from ...config import CYTO_MOLECULAR_CROSS


class IntegratedFeatureEngineering:
    @staticmethod
    def combine_all_features(
        clinical_df: pd.DataFrame,
        molecular_df: Optional[pd.DataFrame] = None,
        burden_df: Optional[pd.DataFrame] = None,
        cyto_df: Optional[pd.DataFrame] = None,
        use_center_ohe: bool = False,
    ) -> pd.DataFrame:

        final_df = clinical_df.copy()
        final_df["ID"] = final_df["ID"].astype(str)

        # Fusionner les dataframes
        if molecular_df is not None and not molecular_df.empty:
            molecular_df["ID"] = molecular_df["ID"].astype(str)
            final_df = final_df.merge(molecular_df, on="ID", how="left")

        if burden_df is not None and not burden_df.empty:
            burden_df["ID"] = burden_df["ID"].astype(str)
            final_df = final_df.merge(burden_df, on="ID", how="left")

        if cyto_df is not None and not cyto_df.empty:
            # L'index de cyto_df est l'ID patient
            cyto_df.index = cyto_df.index.astype(str)
            final_df = final_df.merge(
                cyto_df, left_on="ID", right_index=True, how="left"
            )
        # Optional cross of cyto deletions and mutated arms
        if CYTO_MOLECULAR_CROSS.get("enabled", False):
            cross_specs = [
                ("5q", "del_5q_or_mono5", "mut_in_5q_and_del5q"),
                ("7q", "monosomy_7_or_del7q", "mut_in_7q_and_del7q"),
                ("17p", "del_17p_or_i17q", "mut_in_17p_and_del17p"),
            ]
            for arm, cyto_col, out_col in cross_specs:
                mut_col = f"mutated_arm_{arm}"
                if mut_col in final_df.columns and cyto_col in final_df.columns:
                    final_df[out_col] = (
                        (final_df[mut_col].fillna(0).astype(int) == 1)
                        & (final_df[cyto_col].fillna(0).astype(int) == 1)
                    ).astype(int)

        mutation_cols = [
            col
            for col in final_df.columns
            if col.startswith(
                ("mut_", "vaf_", "CEBPA_", "TP53_", "pathway_", "eln_molecular_risk")
            )
        ]
        final_df[mutation_cols] = final_df[mutation_cols].fillna(0)

        # S'assurer que les colonnes de comptage sont des entiers
        count_cols = [
            col
            for col in final_df.columns
            if "_count" in col or "total_mutations" in col
        ]
        final_df[count_cols] = final_df[count_cols].fillna(0).astype(int)

        # Optionally include CENTER one-hot if requested; otherwise drop CENTER for safety
        if use_center_ohe and "CENTER" in final_df.columns:
            final_df = ClinicalFeatureEngineering.create_center_one_hot_encoding(
                final_df
            )
        final_df = final_df.drop(columns=["CYTOGENETICS", "CENTER"], errors="ignore")

        # Final redundancy pruning
        final_df = _apply_redundancy_policy(final_df)
        return final_df


class CytoMolecularInteractionFeatures:
    """
    Crée des features d'interaction entre les profils cytogénétiques et moléculaires,
    en se basant sur les règles pronostiques établies (ex: ELN 2022).

    Cette classe doit être appelée APRÈS la création de toutes les features
    cytogénétiques et moléculaires de base.
    """

    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Point d'entrée principal. Ajoute toutes les features d'interaction au dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Le dataframe contenant déjà TOUTES les features cliniques,
            cytogénétiques et moléculaires.

        Returns
        -------
        pd.DataFrame
            Le dataframe enrichi avec les nouvelles colonnes d'interaction.
        """
        if df.empty:
            return df

        df_out = df.copy()

        print("[FE Inter] Création des features d'interaction cyto-moléculaires...")

        # --- Interaction 1 : Caryotype Normal stratifié par le profil moléculaire ---
        if "normal_karyotype" in df_out.columns:
            # Cas favorable : caryotype normal AVEC mutation NPM1 (sans FLT3-ITD) OU CEBPA biallélique
            # C'est un signal très fort de bon pronostic.
            is_good_mol = (
                (df_out.get("mut_NPM1", 0) == 1) & (df_out.get("FLT3_ITD", 0) == 0)
            ) | (df_out.get("CEBPA_biallelic", 0) == 1)

            df_out["cyto_normal_mol_favorable"] = (
                (df_out["normal_karyotype"] == 1) & (is_good_mol)
            ).astype(int)

            # Cas défavorable : caryotype normal AVEC FLT3-ITD à haute charge allélique
            # C'est un signal très fort de mauvais pronostic.
            is_bad_mol = (
                df_out.get("FLT3_ITD", 0) == 1
            )  # On peut aussi utiliser FLT3_high_VAF si disponible

            df_out["cyto_normal_mol_adverse"] = (
                (df_out["normal_karyotype"] == 1) & (is_bad_mol)
            ).astype(int)

        # --- Interaction 2 : Caryotype Favorable "annulé" par une mutation défavorable ---
        if "any_favorable_cyto" in df_out.columns and "mut_KIT" in df_out.columns:
            # Cas où un bon caryotype (CBF-AML) est contrebalancé par une mutation de KIT.
            df_out["cyto_favorable_mol_adverse_kit"] = (
                (df_out["any_favorable_cyto"] == 1) & (df_out.get("mut_KIT", 0) == 1)
            ).astype(int)

        # --- Interaction 3 : La "double-peine" -> Caryotype Complexe ET mutation TP53 ---
        if "complex_karyotype" in df_out.columns and "mut_TP53" in df_out.columns:
            # C'est l'un des pires scénarios pronostiques possibles.
            df_out["cyto_complex_and_mol_tp53"] = (
                (df_out["complex_karyotype"] == 1) & (df_out.get("mut_TP53", 0) == 1)
            ).astype(int)

        print(
            f"[FE Inter] {len(df_out.columns) - len(df.columns)} features d'interaction ajoutées."
        )
        return df_out
