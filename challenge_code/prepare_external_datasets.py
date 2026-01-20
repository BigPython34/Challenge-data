"""
Script de préparation des datasets externes (Beat AML / OHSU 2022 et TCGA LAML).
Génère des fichiers standardisés (clinique, moléculaire, cibles) pour le pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "datasets_processed"

# OHSU 2022 (Beat AML)
OHSU_DIR = ROOT_DIR / "datas" / "external" / "aml_ohsu_2022"
OHSU_PATHS = {
    "sample": OHSU_DIR / "data_clinical_sample.txt",
    "blood": OHSU_DIR / "data_blood_cell_percentages.txt",
    "mutations": OHSU_DIR / "data_mutations.txt",
    "structural": OHSU_DIR / "data_sv.txt",
    "patient": OHSU_DIR / "data_clinical_patient.txt",
}

# TCGA LAML
TCGA_DIR = ROOT_DIR / "datas" / "external" / "laml_tcga_pub"
TCGA_PATHS = {
    "sample": TCGA_DIR / "data_clinical_sample.txt",
    "patient": TCGA_DIR / "data_clinical_patient.txt",
    "mutations": TCGA_DIR / "data_mutations.txt",
    "structural": TCGA_DIR / "data_sv.txt",
}

# =============================================================================
# SHARED UTILS
# =============================================================================

def save_dataset(df: pd.DataFrame, path: Path, name: str):
    """Save dataframe to CSV and print status."""
    df.to_csv(path, index=False)
    print(f"[{name}] Saved {len(df)} rows to {path.name}")

# =============================================================================
# OHSU 2022 (BEAT AML) PROCESSING
# =============================================================================

def process_ohsu_2022():
    print("\n" + "="*50)
    print("PROCESSING OHSU 2022 (BEAT AML)")
    print("="*50)

    # --- 1. Load Samples ---
    sample_df = pd.read_csv(
        OHSU_PATHS["sample"],
        sep="\t",
        comment="#",
        dtype={"CENTER_ID": "string"},
        na_values=["NA", ""],
    )
    sample_df["_time"] = pd.to_numeric(
        sample_df["TIME_OF_SAMPLE_COLLECTION_RELATIVE_TO_INCLUSION"],
        errors="coerce",
    ).fillna(0)
    sample_df = sample_df.sort_values(["PATIENT_ID", "_time"])
    sample_df = sample_df.drop_duplicates(subset="PATIENT_ID", keep="first")
    sample_df["CENTER"] = (
        sample_df["CENTER_ID"]
        .fillna("UNKNOWN")
        .astype(str)
        .map(lambda x: x if x.startswith("CENTER_") else f"CENTER_{x}")
    )
    sample_df = sample_df[["PATIENT_ID", "SAMPLE_ID", "CENTER", "KARYOTYPE"]]

    # --- 2. Load Blood Metrics ---
    blood_df = pd.read_csv(OHSU_PATHS["blood"], sep="\t")
    blood_df = blood_df.set_index("ENTITY_STABLE_ID")
    blood_df = blood_df.drop(columns=["NAME", "DESCRIPTION"], errors="ignore")

    metric_map = {
        "BM_BLAST": "PERC_BLASTS_IN_BM",
        "WBC": "WBC_COUNT",
        "PERC_NEUT": "PERC_NEUTROPHILS_IN_PB",
        "PERC_MONO": "PERC_MONOCYTES_IN_PB",
        "HB": "HEMOGLOBIN",
        "PLT": "PLATELET_COUNT",
    }
    
    # Transpose and rename
    labs = blood_df.loc[list(metric_map.values())].T
    labs.index.name = "SAMPLE_ID"
    labs = labs.rename(columns={v: k for k, v in metric_map.items()})
    labs = labs.reset_index()

    for col in labs.columns:
        if col != "SAMPLE_ID":
            labs[col] = pd.to_numeric(labs[col], errors="coerce")

    labs["ANC"] = labs["WBC"] * labs["PERC_NEUT"] / 100.0
    labs["MONOCYTES"] = labs["WBC"] * labs["PERC_MONO"] / 100.0
    labs = labs.drop(columns=["PERC_NEUT", "PERC_MONO"])

    # --- 3. Build Clinical ---
    clinical = sample_df.merge(labs, on="SAMPLE_ID", how="left")
    clinical["CYTOGENETICS"] = clinical["KARYOTYPE"].fillna("")
    clinical = clinical.rename(columns={"PATIENT_ID": "ID"})
    
    # --- 4. Structural Variants (for Cytogenetics refinement) ---
    sv_df = pd.read_csv(OHSU_PATHS["structural"], sep="\t", comment="#")
    matcher = sv_df["Sample_Id"].str.extract(r"_(BA\d+)$")
    sv_df["SAMPLE_ID"] = matcher[0]
    
    # Map sample to patient
    sample_lookup = sample_df.set_index("SAMPLE_ID")["PATIENT_ID"]
    sv_df["PATIENT_ID"] = sv_df["SAMPLE_ID"].map(sample_lookup)
    # Fallback extraction
    fallback = sv_df["Sample_Id"].str.extract(r"^(.*)_BA")[0]
    sv_df["PATIENT_ID"] = sv_df["PATIENT_ID"].fillna(fallback)
    
    structural = sv_df.dropna(subset=["PATIENT_ID"]).copy()
    structural = structural.rename(columns={
        "PATIENT_ID": "ID",
        "Site1_Hugo_Symbol": "SITE1_GENE",
        "Site1_Description": "SITE1_DESC",
        "Site2_Hugo_Symbol": "SITE2_GENE",
        "SV_Status": "SV_STATUS",
        "Event_Info": "EVENT_INFO",
    })

    # Merge SV info into Cytogenetics if missing
    sv_events = (
        structural.groupby("ID")["EVENT_INFO"]
        .apply(lambda values: "; ".join(sorted(set(values))))
        .rename("SV_EVENTS")
        .reset_index()
    )
    clinical = clinical.merge(sv_events, on="ID", how="left")
    mask = clinical["CYTOGENETICS"].eq("") & clinical["SV_EVENTS"].notna()
    clinical.loc[mask, "CYTOGENETICS"] = clinical.loc[mask, "SV_EVENTS"]
    clinical = clinical.drop(columns=["SV_EVENTS"])

    # Final Clinical Columns
    clinical_cols = ["ID", "CENTER", "BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "CYTOGENETICS"]
    clinical = clinical[clinical_cols]

    # --- 5. Mutations ---
    mut_df = pd.read_csv(OHSU_PATHS["mutations"], sep="\t")
    mut_df = mut_df.merge(
        sample_df[["SAMPLE_ID", "PATIENT_ID"]],
        left_on="Tumor_Sample_Barcode",
        right_on="SAMPLE_ID",
        how="left",
    )
    mut_df = mut_df.dropna(subset=["PATIENT_ID"])

    # VAF Calculation
    t_ref = pd.to_numeric(mut_df["t_ref_count"], errors="coerce")
    t_alt = pd.to_numeric(mut_df["t_alt_count"], errors="coerce")
    depth = t_ref.add(t_alt, fill_value=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vaf = np.divide(t_alt, depth)
    vaf = np.where(np.isfinite(vaf), vaf, np.nan)

    # Alt Allele
    alt = mut_df["Tumor_Seq_Allele2"].fillna("")
    mask = alt.eq("") | alt.eq("-")
    alt = alt.mask(mask, mut_df["Tumor_Seq_Allele1"].fillna(""))

    # Protein
    protein = (
        mut_df["HGVSp_Short"].fillna("")
        .where(lambda s: s.ne(""), mut_df["HGVSp"].fillna(""))
    )

    molecular = pd.DataFrame({
        "ID": mut_df["PATIENT_ID"],
        "CHR": mut_df["Chromosome"].astype(str),
        "START": pd.to_numeric(mut_df["Start_Position"], errors="coerce"),
        "END": pd.to_numeric(mut_df["End_Position"], errors="coerce"),
        "REF": mut_df["Reference_Allele"],
        "ALT": alt,
        "GENE": mut_df["Hugo_Symbol"],
        "PROTEIN_CHANGE": protein,
        "EFFECT": mut_df["Consequence"],
        "VAF": vaf,
        "DEPTH": depth,
    })
    molecular = molecular.dropna(subset=["ID", "CHR", "START", "END", "REF", "ALT", "GENE"])
    molecular["VAF"] = molecular["VAF"].round(4)
    molecular = molecular.drop_duplicates()

    # --- 6. Target ---
    patient_df = pd.read_csv(OHSU_PATHS["patient"], sep="\t", comment="#")
    target = patient_df[["PATIENT_ID", "OS_STATUS", "OS_MONTHS"]].copy()
    target = target.dropna(subset=["OS_STATUS", "OS_MONTHS"])
    target["OS_STATUS"] = target["OS_STATUS"].astype(str).str.split(":").str[0].astype(int)
    target["OS_YEARS"] = target["OS_MONTHS"] / 12.0
    target = target.rename(columns={"PATIENT_ID": "ID"})
    target = target[["ID", "OS_YEARS", "OS_STATUS"]]

    # --- Save ---
    save_dataset(clinical, OUTPUT_DIR / "clinical_beat_aml.csv", "OHSU Clinical")
    save_dataset(molecular, OUTPUT_DIR / "molecular_beat_aml.csv", "OHSU Molecular")
    save_dataset(structural, OUTPUT_DIR / "structural_beat_aml.csv", "OHSU Structural")
    save_dataset(target, OUTPUT_DIR / "target_beat_aml.csv", "OHSU Target")


# =============================================================================
# TCGA LAML PROCESSING
# =============================================================================

def process_tcga_laml():
    print("\n" + "="*50)
    print("PROCESSING TCGA LAML")
    print("="*50)

    # --- 1. Load Patient & Sample Info ---
    patient_df = pd.read_csv(TCGA_PATHS["patient"], sep="\t", comment="#")
    sample_df = pd.read_csv(TCGA_PATHS["sample"], sep="\t", comment="#")

    # Keep only Primary Tumor samples usually (03 in TCGA barcode often means primary blood derived cancer)
    # But let's just take the first sample per patient to be safe, or filter by SAMPLE_TYPE
    # TCGA-AB-2803-03 -> 03 is Primary Blood Derived Cancer - Peripheral Blood
    
    # Merge to ensure we have patient-sample link
    # In TCGA LAML, PATIENT_ID is in both.
    
    # --- 2. Build Clinical ---
    # TCGA Patient file has most clinical info
    clinical = patient_df.copy()
    clinical = clinical.rename(columns={
        "PATIENT_ID": "ID",
        "BM_BLAST_PERCENTAGE": "BM_BLAST",
        "WBC": "WBC",
        "CYTOGENETICS": "CYTOGENETICS"
    })

    # Add missing columns with NaNs
    clinical["CENTER"] = "TCGA"
    clinical["ANC"] = np.nan
    clinical["MONOCYTES"] = np.nan
    clinical["HB"] = np.nan
    clinical["PLT"] = np.nan

    # Clean numeric columns
    for col in ["BM_BLAST", "WBC"]:
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")

    # --- 3. Structural Variants ---
    sv_df = pd.read_csv(TCGA_PATHS["structural"], sep="\t", comment="#")
    # TCGA SV file has Sample_Id. We need to map to Patient ID.
    # Sample_Id format: TCGA-AB-2908-03
    # Patient_Id format: TCGA-AB-2908
    sv_df["ID"] = sv_df["Sample_Id"].str.rsplit("-", n=1).str[0]
    
    structural = sv_df.rename(columns={
        "Site1_Hugo_Symbol": "SITE1_GENE",
        "Site2_Hugo_Symbol": "SITE2_GENE",
        "SV_Status": "SV_STATUS",
        "Event_info": "EVENT_INFO",
    })
    structural = structural[["ID", "SITE1_GENE", "SITE2_GENE", "SV_STATUS", "EVENT_INFO"]]

    # Merge SV info into Cytogenetics if missing (same logic as OHSU)
    sv_events = (
        structural.groupby("ID")["EVENT_INFO"]
        .apply(lambda values: "; ".join(sorted(set(values))))
        .rename("SV_EVENTS")
        .reset_index()
    )
    clinical = clinical.merge(sv_events, on="ID", how="left")
    
    # Fill missing cytogenetics with SV info if available
    clinical["CYTOGENETICS"] = clinical["CYTOGENETICS"].fillna("")
    mask = (clinical["CYTOGENETICS"] == "") & clinical["SV_EVENTS"].notna()
    clinical.loc[mask, "CYTOGENETICS"] = clinical.loc[mask, "SV_EVENTS"]
    clinical = clinical.drop(columns=["SV_EVENTS"])

    # Final Clinical Columns
    clinical_cols = ["ID", "CENTER", "BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "CYTOGENETICS"]
    clinical = clinical[clinical_cols]

    # --- 4. Mutations ---
    mut_df = pd.read_csv(TCGA_PATHS["mutations"], sep="\t")
    
    # Map Tumor_Sample_Barcode to Patient ID
    mut_df["ID"] = mut_df["Tumor_Sample_Barcode"].str.rsplit("-", n=1).str[0]

    # VAF Calculation
    t_ref = pd.to_numeric(mut_df["t_ref_count"], errors="coerce")
    t_alt = pd.to_numeric(mut_df["t_alt_count"], errors="coerce")
    depth = t_ref.add(t_alt, fill_value=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vaf = np.divide(t_alt, depth)
    vaf = np.where(np.isfinite(vaf), vaf, np.nan)

    # Alt Allele
    alt = mut_df["Tumor_Seq_Allele2"].fillna("")
    mask = alt.eq("") | alt.eq("-")
    alt = alt.mask(mask, mut_df["Tumor_Seq_Allele1"].fillna(""))

    # Protein
    protein = (
        mut_df["HGVSp_Short"].fillna("")
        .where(lambda s: s.ne(""), mut_df["HGVSp"].fillna(""))
    )

    molecular = pd.DataFrame({
        "ID": mut_df["ID"],
        "CHR": mut_df["Chromosome"].astype(str),
        "START": pd.to_numeric(mut_df["Start_Position"], errors="coerce"),
        "END": pd.to_numeric(mut_df["End_Position"], errors="coerce"),
        "REF": mut_df["Reference_Allele"],
        "ALT": alt,
        "GENE": mut_df["Hugo_Symbol"],
        "PROTEIN_CHANGE": protein,
        "EFFECT": mut_df["Consequence"],
        "VAF": vaf,
        "DEPTH": depth,
    })
    molecular = molecular.dropna(subset=["ID", "CHR", "START", "END", "REF", "ALT", "GENE"])
    molecular["VAF"] = molecular["VAF"].round(4)
    molecular = molecular.drop_duplicates()

    # --- 5. Target ---
    # From patient_df
    target = patient_df[["PATIENT_ID", "OS_STATUS", "OS_MONTHS"]].copy()
    target = target.rename(columns={"PATIENT_ID": "ID"})
    target = target.dropna(subset=["OS_STATUS", "OS_MONTHS"])
    
    # Parse OS_STATUS: "1:DECEASED" -> 1, "0:LIVING" -> 0
    target["OS_STATUS"] = target["OS_STATUS"].astype(str).str.split(":").str[0].astype(int)
    target["OS_YEARS"] = pd.to_numeric(target["OS_MONTHS"], errors="coerce") / 12.0
    
    target = target[["ID", "OS_YEARS", "OS_STATUS"]]

    # --- Save ---
    save_dataset(clinical, OUTPUT_DIR / "clinical_tcga.csv", "TCGA Clinical")
    save_dataset(molecular, OUTPUT_DIR / "molecular_tcga.csv", "TCGA Molecular")
    save_dataset(structural, OUTPUT_DIR / "structural_tcga.csv", "TCGA Structural")
    save_dataset(target, OUTPUT_DIR / "target_tcga.csv", "TCGA Target")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process OHSU
    try:
        process_ohsu_2022()
    except Exception as e:
        print(f"[ERROR] Failed to process OHSU 2022: {e}")
        import traceback
        traceback.print_exc()

    # Process TCGA
    try:
        process_tcga_laml()
    except Exception as e:
        print(f"[ERROR] Failed to process TCGA LAML: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
