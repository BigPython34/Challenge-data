from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "datas" / "external" / "aml_ohsu_2022"
OUTPUT_DIR = ROOT_DIR / "datasets_processed"

SAMPLE_PATH = DATA_DIR / "data_clinical_sample.txt"
BLOOD_PATH = DATA_DIR / "data_blood_cell_percentages.txt"
MUTATIONS_PATH = DATA_DIR / "data_mutations.txt"
STRUCTURAL_PATH = DATA_DIR / "data_sv.txt"
PATIENT_PATH = DATA_DIR / "data_clinical_patient.txt"


def load_sample_metadata() -> pd.DataFrame:
    sample_df = pd.read_csv(
        SAMPLE_PATH,
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
    keep_cols = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "CENTER",
        "KARYOTYPE",
    ]
    return sample_df[keep_cols]


def load_blood_metrics() -> pd.DataFrame:
    blood_df = pd.read_csv(BLOOD_PATH, sep="\t")
    blood_df = blood_df.set_index("ENTITY_STABLE_ID")
    blood_df = blood_df.drop(columns=["NAME", "DESCRIPTION"], errors="ignore")

    metric_map: Dict[str, str] = {
        "BM_BLAST": "PERC_BLASTS_IN_BM",
        "WBC": "WBC_COUNT",
        "PERC_NEUT": "PERC_NEUTROPHILS_IN_PB",
        "PERC_MONO": "PERC_MONOCYTES_IN_PB",
        "HB": "HEMOGLOBIN",
        "PLT": "PLATELET_COUNT",
    }

    missing_metrics: List[str] = [v for v in metric_map.values() if v not in blood_df.index]
    if missing_metrics:
        raise ValueError(f"Missing metrics in blood table: {missing_metrics}")

    labs = blood_df.loc[list(metric_map.values())].T
    labs.index.name = "SAMPLE_ID"
    labs = labs.rename(columns={v: k for k, v in metric_map.items()})
    labs = labs.reset_index()

    for col in labs.columns:
        if col == "SAMPLE_ID":
            continue
        labs[col] = pd.to_numeric(labs[col], errors="coerce")

    labs["ANC"] = labs["WBC"] * labs["PERC_NEUT"] / 100.0
    labs["MONOCYTES"] = labs["WBC"] * labs["PERC_MONO"] / 100.0
    labs = labs.drop(columns=["PERC_NEUT", "PERC_MONO"])
    return labs


def build_clinical_dataset(sample_df: pd.DataFrame, labs_df: pd.DataFrame) -> pd.DataFrame:
    clinical = sample_df.merge(labs_df, on="SAMPLE_ID", how="left")
    clinical["CYTOGENETICS"] = clinical["KARYOTYPE"].fillna("")
    clinical = clinical.rename(columns={
        "PATIENT_ID": "ID",
        "BM_BLAST": "BM_BLAST",
        "HB": "HB",
        "PLT": "PLT",
        "WBC": "WBC",
    })
    ordered_cols = [
        "ID",
        "CENTER",
        "BM_BLAST",
        "WBC",
        "ANC",
        "MONOCYTES",
        "HB",
        "PLT",
        "CYTOGENETICS",
    ]
    return clinical[ordered_cols]


def load_mutations(sample_df: pd.DataFrame) -> pd.DataFrame:
    mut_df = pd.read_csv(MUTATIONS_PATH, sep="\t")
    mut_df = mut_df.merge(
        sample_df[["SAMPLE_ID", "PATIENT_ID"]],
        left_on="Tumor_Sample_Barcode",
        right_on="SAMPLE_ID",
        how="left",
    )
    mut_df = mut_df.dropna(subset=["PATIENT_ID"])

    t_ref = pd.to_numeric(mut_df["t_ref_count"], errors="coerce")
    t_alt = pd.to_numeric(mut_df["t_alt_count"], errors="coerce")
    depth = t_ref.add(t_alt, fill_value=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vaf = np.divide(t_alt, depth)
    vaf = np.where(np.isfinite(vaf), vaf, np.nan)

    alt = mut_df["Tumor_Seq_Allele2"].fillna("")
    mask = alt.eq("") | alt.eq("-")
    alt = alt.mask(mask, mut_df["Tumor_Seq_Allele1"].fillna(""))

    protein = (
        mut_df["HGVSp_Short"].fillna("")
        .where(lambda s: s.ne(""), mut_df["HGVSp"].fillna(""))
    )

    molecular = pd.DataFrame(
        {
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
        }
    )

    molecular = molecular.dropna(subset=["ID", "CHR", "START", "END", "REF", "ALT", "GENE"])
    molecular = molecular.sort_values("DEPTH", ascending=False)
    molecular = molecular.drop_duplicates(
        subset=["ID", "CHR", "START", "END", "REF", "ALT", "GENE", "PROTEIN_CHANGE", "EFFECT"],
        keep="first",
    )
    molecular["VAF"] = molecular["VAF"].round(4)
    return molecular


def load_structural_variants(sample_df: pd.DataFrame) -> pd.DataFrame:
    sv_df = pd.read_csv(STRUCTURAL_PATH, sep="\t", comment="#")
    matcher = sv_df["Sample_Id"].str.extract(r"_(BA\d+)$")
    sv_df["SAMPLE_ID"] = matcher[0]

    sample_lookup = sample_df.set_index("SAMPLE_ID")["PATIENT_ID"]
    sv_df["PATIENT_ID"] = sv_df["SAMPLE_ID"].map(sample_lookup)
    fallback = sv_df["Sample_Id"].str.extract(r"^(.*)_BA")[0]
    sv_df["PATIENT_ID"] = sv_df["PATIENT_ID"].fillna(fallback)
    structural = sv_df.dropna(subset=["PATIENT_ID"]).copy()

    structural = structural.rename(
        columns={
            "PATIENT_ID": "ID",
            "Site1_Hugo_Symbol": "SITE1_GENE",
            "Site1_Description": "SITE1_DESC",
            "Site2_Hugo_Symbol": "SITE2_GENE",
            "SV_Status": "SV_STATUS",
            "Event_Info": "EVENT_INFO",
        }
    )

    keep_cols = [
        "ID",
        "SITE1_GENE",
        "SITE1_DESC",
        "SITE2_GENE",
        "SV_STATUS",
        "EVENT_INFO",
    ]
    return structural[keep_cols]


def build_target_dataset() -> pd.DataFrame:
    patient_df = pd.read_csv(PATIENT_PATH, sep="\t", comment="#")
    target = patient_df[["PATIENT_ID", "OS_STATUS", "OS_MONTHS"]].copy()
    target = target.dropna(subset=["OS_STATUS", "OS_MONTHS"])

    target["OS_STATUS"] = (
        target["OS_STATUS"].astype(str).str.split(":").str[0].astype(int)
    )
    target["OS_YEARS"] = target["OS_MONTHS"] / 12.0
    target = target.rename(columns={"PATIENT_ID": "ID"})
    ordered_cols = ["ID", "OS_YEARS", "OS_STATUS"]
    return target[ordered_cols]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_df = load_sample_metadata()
    labs_df = load_blood_metrics()
    clinical = build_clinical_dataset(sample_df, labs_df)
    structural = load_structural_variants(sample_df)

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
    clinical_path = OUTPUT_DIR / "clinical_beat_aml.csv"
    clinical.to_csv(clinical_path, index=False)

    molecular = load_mutations(sample_df)
    molecular_path = OUTPUT_DIR / "molecular_beat_aml.csv"
    molecular.to_csv(molecular_path, index=False)

    structural_path = OUTPUT_DIR / "structural_beat_aml.csv"
    structural.to_csv(structural_path, index=False)

    target = build_target_dataset()
    target_path = OUTPUT_DIR / "target_beat_aml.csv"
    target.to_csv(target_path, index=False)

    print(f"Clinical rows: {len(clinical)} -> {clinical_path}")
    print(f"Molecular rows: {len(molecular)} -> {molecular_path}")
    print(f"Structural rows: {len(structural)} -> {structural_path}")
    print(f"Target rows: {len(target)} -> {target_path}")


if __name__ == "__main__":
    main()
