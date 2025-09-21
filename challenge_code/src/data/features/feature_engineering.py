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
from ...config import CYTO_MOLECULAR_CROSS, FEATURE_INTERACTIONS, ID_COLUMNS


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
            cross_specs = CYTO_MOLECULAR_CROSS.get("specs", [])
            for spec in cross_specs:
                arm = spec["arm"]
                cyto_col = spec["cyto_col"]
                out_col = spec["out_col"]
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
        if df.empty or not FEATURE_INTERACTIONS.get("enabled", False):
            return df

        df_out = df.copy()
        print("[FE Inter] Création des features d'interaction cyto-moléculaires...")
        
        # --- Interaction 1 : Caryotype Normal stratifié par le profil moléculaire ---
        config_fav = FEATURE_INTERACTIONS.get("cyto_normal_mol_favorable", {})
        if config_fav.get("enabled") and config_fav.get("base_col") in df_out.columns:
            base_col = config_fav["base_col"]
            good_mol_cols = [c for c in config_fav.get("good_mol_cols", []) if c in df_out.columns]
            bad_mol_cols = [c for c in config_fav.get("bad_mol_cols_for_good", []) if c in df_out.columns]
            
            is_good_mol = pd.Series(False, index=df_out.index)
            if good_mol_cols:
                is_good_mol = (df_out[good_mol_cols].sum(axis=1) > 0)
            
            if bad_mol_cols:
                is_good_mol &= (df_out[bad_mol_cols].sum(axis=1) == 0)

            df_out["cyto_normal_mol_favorable"] = ((df_out[base_col] == 1) & is_good_mol).astype(int)

        # --- Interaction 2 : Caryotype normal AVEC mutation adverse ---
        config_adv = FEATURE_INTERACTIONS.get("cyto_normal_mol_adverse", {})
        if config_adv.get("enabled") and config_adv.get("base_col") in df_out.columns:
            base_col = config_adv["base_col"]
            adverse_cols = [c for c in config_adv.get("adverse_mol_cols", []) if c in df_out.columns]
            if adverse_cols:
                is_bad_mol = (df_out[adverse_cols].sum(axis=1) > 0)
                df_out["cyto_normal_mol_adverse"] = ((df_out[base_col] == 1) & is_bad_mol).astype(int)

        # --- Interaction 3 : Caryotype Favorable "annulé" par une mutation défavorable ---
        config_kit = FEATURE_INTERACTIONS.get("cyto_favorable_mol_adverse_kit", {})
        if config_kit.get("enabled") and config_kit.get("base_col") in df_out.columns:
            base_col = config_kit["base_col"]
            adverse_cols = [c for c in config_kit.get("adverse_mol_cols", []) if c in df_out.columns]
            if adverse_cols:
                 df_out["cyto_favorable_mol_adverse_kit"] = ((df_out[base_col] == 1) & (df_out[adverse_cols].sum(axis=1) > 0)).astype(int)

        # --- Interaction 4 : La "double-peine" -> Caryotype Complexe ET mutation TP53 ---
        config_tp53 = FEATURE_INTERACTIONS.get("cyto_complex_and_mol_tp53", {})
        if config_tp53.get("enabled") and config_tp53.get("base_col") in df_out.columns:
            base_col = config_tp53["base_col"]
            adverse_cols = [c for c in config_tp53.get("adverse_mol_cols", []) if c in df_out.columns]
            if adverse_cols:
                df_out["cyto_complex_and_mol_tp53"] = ((df_out[base_col] == 1) & (df_out[adverse_cols].sum(axis=1) > 0)).astype(int)

        print(
            f"[FE Inter] {len(df_out.columns) - len(df.columns)} features d'interaction ajoutées."
        )
        return df_out
