import os
import sys
import re
import json
from collections import Counter
from typing import Optional
import pandas as pd

# Optional config import for comparison only; exploration itself is config-agnostic
HAVE_CONFIG = False
try:
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    )
    from src.config import (
        ALL_IMPORTANT_GENES,
        CYTOGENETIC_ADVERSE,
        CYTOGENETIC_FAVORABLE,
        CYTOGENETIC_INTERMEDIATE,
    )

    HAVE_CONFIG = True
except Exception:
    HAVE_CONFIG = False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _detect_gene_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["GENE", "Hugo_Symbol", "Gene", "Gene_Symbol"]:
        if cand in df.columns:
            return cand
    return None


def run_comprehensive_data_discovery(
    clinical_path: str, molecular_path: str, out_dir: str = "reports/data_explore"
):
    """
    Exploration 100% data-driven (agnostique à la config) + comparaison optionnelle si la config est disponible.
    Génère des rapports CSV/JSON pour guider la mise à jour de la config.
    """
    _ensure_dir(out_dir)
    print("=" * 60)
    print("  EXPLORATION AGNOSTIQUE DES DONNÉES (config facultative)")
    print("=" * 60)


    print("\n--- PARTIE 1: MOLÉCULAIRE ---")
    try:
        mol_df = pd.read_csv(molecular_path, low_memory=False)
        gene_col = _detect_gene_column(mol_df)
        if not gene_col:
            print(
                "[MOL] Colonne gène introuvable (GENE/Hugo_Symbol/…) – section sautée."
            )
        else:
            mol_df[gene_col] = mol_df[gene_col].astype(str)
            gene_counts = mol_df[gene_col].value_counts()
            gene_counts.to_csv(
                os.path.join(out_dir, "molecular_gene_counts.csv"), header=True
            )
            print(
                f"[MOL] {len(gene_counts)} gènes trouvés. Export: molecular_gene_counts.csv"
            )

            if HAVE_CONFIG:
                genes_from_data = set(gene_counts.index)
                genes_from_cfg = set(ALL_IMPORTANT_GENES)
                missing_from_cfg = sorted(
                    g for g in genes_from_data if g not in genes_from_cfg
                )
                extra_in_cfg = sorted(
                    g for g in genes_from_cfg if g not in genes_from_data
                )
                pd.Series(missing_from_cfg, name="gene").to_csv(
                    os.path.join(out_dir, "molecular_genes_missing_from_config.csv"),
                    index=False,
                )
                pd.Series(extra_in_cfg, name="gene").to_csv(
                    os.path.join(out_dir, "molecular_genes_unused_in_data.csv"),
                    index=False,
                )
                print("[MOL] Comparaison config exportée (missing/unused).")
    except FileNotFoundError:
        print(f"[MOL] Fichier introuvable: {molecular_path}")


    print("\n--- PARTIE 2: CYTOGÉNÉTIQUE ---")
    try:
        clin_df = pd.read_csv(clinical_path, low_memory=False)
        if "CYTOGENETICS" not in clin_df.columns:
            print("[CYTO] Colonne CYTOGENETICS absente – section sautée.")
        else:
            cyto = clin_df["CYTOGENETICS"].dropna().astype(str)
            cyto_lower = cyto.str.lower()

            # Tokenisation large, on garde + et -
            stripped = re.sub(r"[\(\)\[\]\{\};:]+", " ", " ".join(cyto_lower.tolist()))
            stripped = stripped.replace(",", " ").replace("/", " ")
            tokens = [tok for tok in stripped.split() if tok]
            token_counts = Counter(tokens)
            pd.Series(token_counts).sort_values(ascending=False).to_csv(
                os.path.join(out_dir, "cyto_token_counts.csv"), header=["count"]
            )
            print("[CYTO] Export: cyto_token_counts.csv (tokens bruts)")


            patterns = {
                "t_translocation": r"t\(\d+;\d+\)",
                "del_any": r"del\(.*?\)",
                "inv_any": r"inv\(.*?\)",
                "add_any": r"add\(.*?\)",
                "der_any": r"der\(.*?\)",
                "ins_any": r"ins\(.*?\)",
                "i_isochrom": r"\bi\(.*?\)",
                "dic_any": r"\bdic\(.*?\)",
                "plus_mar": r"\+mar\b",
                "minus_Y": r"-y\b",
                "+21": r"\+21\b",
                "idem": r"\bidem\b",
                "normal_46xx_xy": r"\b46[, ]?(?:xx|xy)\b",
            }
            patt_counts = {}
            for name, patt in patterns.items():
                patt_counts[name] = cyto_lower.str.contains(patt, regex=True).sum()
            pd.Series(patt_counts).to_csv(
                os.path.join(out_dir, "cyto_pattern_counts.csv"), header=["count"]
            )
            print("[CYTO] Export: cyto_pattern_counts.csv (patterns structurés)")


            monos = Counter()
            tris = Counter()
            for s in cyto_lower:
                for m in re.findall(r"-(\d+)\b", s):
                    monos[m] += 1
                for p in re.findall(r"\+(\d+)\b", s):
                    tris[p] += 1
            pd.DataFrame(
                {"monosomy": list(monos.keys()), "count": list(monos.values())}
            ).sort_values(by="count", ascending=False).to_csv(
                os.path.join(out_dir, "cyto_monosomies_counts.csv"), index=False
            )
            pd.DataFrame(
                {"trisomy": list(tris.keys()), "count": list(tris.values())}
            ).sort_values(by="count", ascending=False).to_csv(
                os.path.join(out_dir, "cyto_trisomies_counts.csv"), index=False
            )
            print(
                "[CYTO] Export: cyto_monosomies_counts.csv / cyto_trisomies_counts.csv"
            )


            interest_mask = cyto_lower.str.contains(r"\bdic\(|\bidem\b", regex=True)
            cyto[interest_mask].head(50).to_csv(
                os.path.join(out_dir, "cyto_samples_dic_idem_head50.txt"),
                index=False,
                header=False,
            )


            summary = {
                "n_rows_with_idem": int(patt_counts.get("idem", 0)),
                "n_rows_with_dic": int(patt_counts.get("dic_any", 0)),
                "n_rows": int(len(cyto_lower)),
                "present_special_tokens": {
                    "dic": bool(patt_counts.get("dic_any", 0) > 0),
                    "idem": bool(patt_counts.get("idem", 0) > 0),
                    "+mar": bool(patt_counts.get("plus_mar", 0) > 0),
                    "der": bool(patt_counts.get("der_any", 0) > 0),
                    "ins": bool(patt_counts.get("ins_any", 0) > 0),
                    "i()": bool(patt_counts.get("i_isochrom", 0) > 0),
                },
                "top_monosomies": {
                    k: int(v)
                    for k, v in pd.Series(monos)
                    .sort_values(ascending=False)
                    .head(10)
                    .to_dict()
                    .items()
                },
                "top_trisomies": {
                    k: int(v)
                    for k, v in pd.Series(tris)
                    .sort_values(ascending=False)
                    .head(10)
                    .to_dict()
                    .items()
                },
            }

            suggestions = {
                "cyto_adverse_candidates": [
                    f"-{k}" for k, _ in sorted(monos.items(), key=lambda x: -x[1])
                ],
                "cyto_intermediate_candidates": [
                    f"+{k}" for k, _ in sorted(tris.items(), key=lambda x: -x[1])
                ],
                "cyto_special_tokens": [
                    k for k, v in summary["present_special_tokens"].items() if v
                ],
            }

            # If config present, add comparison for patterns
            if HAVE_CONFIG:
                summary["config_present"] = True
                summary["config_patterns_seen_counts"] = {
                    "favorable_any": int(
                        sum(
                            cyto_lower.str.contains(p, regex=True).sum()
                            for p in CYTOGENETIC_FAVORABLE
                        )
                    ),
                    "intermediate_any": int(
                        sum(
                            cyto_lower.str.contains(p, regex=True).sum()
                            for p in CYTOGENETIC_INTERMEDIATE
                        )
                    ),
                    "adverse_any": int(
                        sum(
                            cyto_lower.str.contains(p, regex=True).sum()
                            for p in CYTOGENETIC_ADVERSE
                        )
                    ),
                }
            else:
                summary["config_present"] = False

            with open(
                os.path.join(out_dir, "cyto_summary.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            with open(
                os.path.join(out_dir, "suggested_config_update.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(suggestions, f, ensure_ascii=False, indent=2)
            print("[CYTO] Export: cyto_summary.json / suggested_config_update.json")

    except FileNotFoundError:
        print(f"[CYTO] Fichier clinique introuvable: {clinical_path}")
