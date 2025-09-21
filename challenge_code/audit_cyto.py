# audit_cytogenetics.py (Version Améliorée)

import pandas as pd
import numpy as np
import random
from typing import List

# ==============================================================================
# 1. CONFIGURATION DE L'AUDIT
# ==============================================================================

# --- MODIFIEZ CES CHEMINS ---
RAW_DATA_FILE_PATH = "datas/X_test/clinical_test.csv"
FEATURED_DATA_FILE_PATH = "datasets_processed/X_test_processed.csv"
# -----------------------------

CYTO_COLUMN_NAME = "CYTOGENETICS"
PATIENT_ID_COLUMN = "ID"
N_PATIENTS_TO_AUDIT = 20

# Organisation des features par thème pour l'affichage
FEATURE_GROUPS = {
    "1. Scores de Risque Finaux": [
        "eln_cyto_favorable",
        "eln_cyto_intermediate",
        "eln_cyto_adverse",
    ],
    "2. Indicateurs de Complexité Globale": [
        "num_cyto_abnormalities",
        "complex_karyotype",
        "monosomal_karyotype",
        "normal_karyotype",
    ],
    "3. Flags d'Anomalies Spécifiques": [
        "monosomy_7_or_del7q",
        "del_5q_or_mono5",
        "del_17p_or_i17q",
        "rearr_3q26",
        "trisomy_8",
        "minus_Y",
        "plus_21",
        "t_9_11",
        "del_12p",
    ],
    "4. Analyse des Clones": [
        "clone_count",
        "main_clone_abnormality_count",
        "has_idem",
        "total_cell_count",
        "proportion_main_clone",
        "main_clone_is_complex",
    ],
    "5. Comptes Détaillés (non-nuls)": [
        "n_t",
        "n_del",
        "n_inv",
        "n_add",
        "n_der",
        "n_ins",
        "n_i",
        "n_dic",
        "n_plus",
        "n_minus",
        "n_mar",
        "n_ring",
        "n_dmin",
    ],
}

# ==============================================================================
# 2. MOTEUR DE L'AUDIT
# ==============================================================================


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def audit_patient(patient_row: pd.Series):
    patient_id = patient_row.get(PATIENT_ID_COLUMN, "N/A")
    cyto_string = patient_row.get(CYTO_COLUMN_NAME, "CHAÎNE MANQUANTE")

    print(
        f"\n{Colors.MAGENTA}================== AUDIT PATIENT: {patient_id} =================={Colors.RESET}"
    )
    print(f"{Colors.CYAN}Chaîne Originale ->{Colors.RESET} {cyto_string}")

    for group_name, columns in FEATURE_GROUPS.items():
        print(f"\n--- {Colors.BLUE}{group_name}{Colors.RESET} ---")

        has_output = False
        for col in columns:
            if col not in patient_row:
                continue

            value = patient_row[col]
            if (
                pd.isna(value)
                or value == 0
                and group_name == "5. Comptes Détaillés (non-nuls)"
            ):
                continue  # N'affiche pas les compteurs à zéro pour alléger

            has_output = True
            value_str = ""
            if pd.isna(value):
                value_str = f"{Colors.YELLOW}NaN{Colors.RESET}"
            else:
                numeric_value = float(value)
                if numeric_value == 1:
                    if (
                        "adverse" in col
                        or "mono" in col
                        or "del" in col
                        or "complex" in col
                    ):
                        value_str = f"{Colors.RED}{int(numeric_value)}{Colors.RESET}"
                    elif "favorable" in col or "normal" in col:
                        value_str = f"{Colors.GREEN}{int(numeric_value)}{Colors.RESET}"
                    else:
                        value_str = f"{Colors.YELLOW}{int(numeric_value)}{Colors.RESET}"
                elif numeric_value > 0:
                    value_str = f"{Colors.YELLOW}{numeric_value}{Colors.RESET}"
                else:
                    value_str = str(int(numeric_value))

            print(f"  - {col:<30}: {value_str}")

        if not has_output:
            print(f"  ({Colors.GREEN}Aucun trouvé{Colors.RESET})")


def check_special_cases_single(row: pd.Series) -> List[str]:
    """Vérifie les cas spéciaux pour un seul patient et retourne la liste des issues."""
    issues = []
    patient_id = row.get(PATIENT_ID_COLUMN, "N/A")

    # Vérifier que si complex_karyotype == 1, alors eln_cyto_adverse == 1
    if row.get("complex_karyotype", 0) == 1 and row.get("eln_cyto_adverse", 0) != 1:
        issues.append(f"Patient {patient_id}: complex_karyotype=1 mais eln_cyto_adverse != 1")

    # Vérifier que si monosomal_karyotype == 1, alors eln_cyto_adverse == 1
    if row.get("monosomal_karyotype", 0) == 1 and row.get("eln_cyto_adverse", 0) != 1:
        issues.append(f"Patient {patient_id}: monosomal_karyotype=1 mais eln_cyto_adverse != 1")

    # Vérifier les anomalies adverses spécifiques
    adverse_anomalies = ["monosomy_7_or_del7q", "del_5q_or_mono5", "del_17p_or_i17q", "rearr_3q26"]
    for anomaly in adverse_anomalies:
        if row.get(anomaly, 0) == 1 and row.get("eln_cyto_adverse", 0) != 1:
            issues.append(f"Patient {patient_id}: {anomaly}=1 mais eln_cyto_adverse != 1")

    # Vérifier que favorable et adverse ne sont pas tous les deux à 1
    if row.get("eln_cyto_favorable", 0) == 1 and row.get("eln_cyto_adverse", 0) == 1:
        issues.append(f"Patient {patient_id}: eln_cyto_favorable et eln_cyto_adverse tous les deux à 1")

    # Vérifier que normal_karyotype implique intermediate
    if row.get("normal_karyotype", 0) == 1:
        if row.get("eln_cyto_intermediate", 0) != 1:
            issues.append(f"Patient {patient_id}: normal_karyotype=1 mais eln_cyto_intermediate != 1")
        if row.get("eln_cyto_favorable", 0) != 0:
            issues.append(f"Patient {patient_id}: normal_karyotype=1 mais eln_cyto_favorable != 0")
        if row.get("eln_cyto_adverse", 0) != 0:
            issues.append(f"Patient {patient_id}: normal_karyotype=1 mais eln_cyto_adverse != 0")

    return issues
    """Vérifie les cas spéciaux dans un groupe de patients."""
    print(f"\n{Colors.CYAN}--- VÉRIFICATIONS DES CAS SPÉCIAUX ---{Colors.RESET}")

    issues = []

    for idx, row in group.iterrows():
        patient_id = row.get(PATIENT_ID_COLUMN, "N/A")

        # Vérifier que si complex_karyotype == 1, alors eln_cyto_adverse == 1
        if row.get("complex_karyotype", 0) == 1 and row.get("eln_cyto_adverse", 0) != 1:
            issues.append(f"Patient {patient_id}: complex_karyotype=1 mais eln_cyto_adverse != 1")

        # Vérifier que si monosomal_karyotype == 1, alors eln_cyto_adverse == 1
        if row.get("monosomal_karyotype", 0) == 1 and row.get("eln_cyto_adverse", 0) != 1:
            issues.append(f"Patient {patient_id}: monosomal_karyotype=1 mais eln_cyto_adverse != 1")

        # Vérifier les anomalies adverses spécifiques
        adverse_anomalies = ["monosomy_7_or_del7q", "del_5q_or_mono5", "del_17p_or_i17q", "rearr_3q26"]
        for anomaly in adverse_anomalies:
            if row.get(anomaly, 0) == 1 and row.get("eln_cyto_adverse", 0) != 1:
                issues.append(f"Patient {patient_id}: {anomaly}=1 mais eln_cyto_adverse != 1")

        # Vérifier que favorable et adverse ne sont pas tous les deux à 1
        if row.get("eln_cyto_favorable", 0) == 1 and row.get("eln_cyto_adverse", 0) == 1:
            issues.append(f"Patient {patient_id}: eln_cyto_favorable et eln_cyto_adverse tous les deux à 1")

        # Vérifier que normal_karyotype implique intermediate (pas favorable)
        if row.get("normal_karyotype", 0) == 1:
            if row.get("eln_cyto_intermediate", 0) != 1:
                issues.append(f"Patient {patient_id}: normal_karyotype=1 mais eln_cyto_intermediate != 1")
            if row.get("eln_cyto_favorable", 0) != 0:
                issues.append(f"Patient {patient_id}: normal_karyotype=1 mais eln_cyto_favorable != 0")
            if row.get("eln_cyto_adverse", 0) != 0:
                issues.append(f"Patient {patient_id}: normal_karyotype=1 mais eln_cyto_adverse != 0")

        # # Vérifier la cohérence des comptes (désactivé car pondéré par clones)
        # num_abnormalities = row.get("num_cyto_abnormalities", 0)
        # structural_count = sum(row.get(col, 0) for col in ["n_t", "n_del", "n_inv", "n_add", "n_der", "n_ins", "n_i", "n_dic", "n_ring"])
        # numerical_count = sum(row.get(col, 0) for col in ["n_plus", "n_minus"])
        # total_count = structural_count + numerical_count
        # if abs(num_abnormalities - total_count) > 0.1:  # tolérance pour float
        #     issues.append(f"Patient {patient_id}: num_cyto_abnormalities ({num_abnormalities}) != somme des comptes ({total_count})")

    if issues:
        print(f"{Colors.RED}PROBLÈMES DÉTECTÉS:{Colors.RESET}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"{Colors.GREEN}Aucun problème détecté dans ce groupe.{Colors.RESET}")


def main():
    print("Lancement de l'audit...")
    try:
        df_raw = pd.read_csv(
            RAW_DATA_FILE_PATH, usecols=[PATIENT_ID_COLUMN, CYTO_COLUMN_NAME]
        )
        df_featured = pd.read_csv(FEATURED_DATA_FILE_PATH)
    except Exception as e:
        print(f"{Colors.RED}ERREUR lors du chargement des fichiers: {e}{Colors.RESET}")
        return

    print(f"Fusion des données sur la clé '{PATIENT_ID_COLUMN}'...")
    df_audit = pd.merge(df_featured, df_raw, on=PATIENT_ID_COLUMN, how="inner")

    if df_audit.empty:
        print(
            f"{Colors.YELLOW}AVERTISSEMENT: La fusion n'a produit aucun résultat.{Colors.RESET}"
        )
        return

    non_null_df = df_audit[df_audit[CYTO_COLUMN_NAME].notna()].copy()
    if non_null_df.empty:
        print("Aucun patient avec des données cytogénétiques non-nulles à auditer.")
        return

    # Effectuer plusieurs runs avec des seeds différentes pour validation
    for run_num in range(1, 4):  # 3 runs
        print(f"\n{'='*80}")
        print(f"RUN {run_num} (seed = {2120 + run_num})")
        print(f"{'='*80}")
        
        sample_size = min(N_PATIENTS_TO_AUDIT, len(non_null_df))
        audit_sample = non_null_df.sample(n=sample_size, random_state=2120 + run_num)
        
        # Analyser seulement le premier groupe de chaque run pour vérification rapide
        group = audit_sample.iloc[:20]
        print(f"Affichage d'un échantillon de 20 patients pour validation...")
        
        issues_summary = []
        for _, row in group.iterrows():
            audit_patient(row)
            # Collecter les issues pour ce run
            patient_issues = check_special_cases_single(row)
            if patient_issues:
                issues_summary.extend(patient_issues)
        
        print(f"\n--- RÉSUMÉ RUN {run_num} ---")
        if issues_summary:
            print(f"{Colors.RED}PROBLÈMES DÉTECTÉS:{Colors.RESET}")
            for issue in issues_summary:
                print(f"  - {issue}")
        else:
            print(f"{Colors.GREEN}Aucun problème détecté.{Colors.RESET}")
        
        print(f"{'='*80}")

    print(
        f"\n{Colors.GREEN}================== Audit Multi-Run Terminé =================={Colors.RESET}"
    )


if __name__ == "__main__":
    main()
