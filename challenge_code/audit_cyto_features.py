#!/usr/bin/env python3
"""Compares raw cytogenetic reports against engineered features for every patient."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import pandas as pd

from src.config import DATA_PATHS
from src.data.features.clinical_feature_engineering import (
    CytogeneticFeatureExtraction,
)

Split = Literal["train", "test"]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_PATHS: Dict[Split, Path] = {
    "train": SCRIPT_DIR / "datasets_processed" / "X_train_processed.csv",
    "test": SCRIPT_DIR / "datasets_processed" / "X_test_processed.csv",
}
REPORT_ROOT = PROJECT_ROOT / "reports" / "cyto_feature_audit"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


EVENT_RULES = [
    {
        "feature": "monosomy_X",
        "description": "-X / monosomie X",
        "patterns": [
            r"-\s*x[a-z0-9]*(?:\[[^\]]+\])?",
            r"mono(?:somy)?\s*x",
        ],
    },
    {
        "feature": "trisomy_X",
        "description": "+X / trisomie X",
        "patterns": [
            r"\+\s*x[a-z0-9]*(?:\[[^\]]+\])?",
            r"tris(?:omy|omia)?\s*x",
        ],
    },
    {
        "feature": "trisomy_Y",
        "description": "+Y / trisomie Y",
        "patterns": [
            r"\+\s*y[a-z0-9]*(?:\[[^\]]+\])?",
            r"tris(?:omy|omia)?\s*y",
        ],
    },
    {
        "feature": "trisomy_8",
        "description": "+8 / trisomy 8",
        "patterns": [
            r"\+\s*8(?!\d)",
            r"\btri(?:som(?:y|ie|ia)|s)?\s*-?\s*8\b",
            r"\btri8\b",
            r"\badd\s*\(?\s*8",
        ],
    },
    {
        "feature": "plus_21",
        "description": "+21 / trisomy 21",
        "patterns": [
            r"\+\s*21(?!\d)",
            r"\btri(?:som(?:y|ie|ia)|s)?\s*-?\s*21\b",
            r"\btri21\b",
        ],
    },
    {
        "feature": "del_5q_or_mono5",
        "description": "-5 / del(5q)",
        "patterns": [
            r"-\s*5(?!\d)",
            r"mono(?:somy)?\s*-?\s*5",
            r"del\s*5q",
            r"del\s*\(\s*5\s*\)\s*\(q",
            r"5q-",
        ],
    },
    {
        "feature": "del_17p_or_i17q",
        "description": "del(17p) / i(17q)",
        "patterns": [
            r"del\s*17p",
            r"17p-",
            r"del\s*\(\s*17\s*\)\s*\(p",
            r"i\s*\(\s*17q\s*\)",
        ],
    },
    {
        "feature": "minus_Y",
        "description": "-Y",
        "patterns": [
            r"-\s*y(?![A-Za-z])",
            r"minus\s*y",
            r"del\s*y",
        ],
    },
    {
        "feature": "rearr_3q26",
        "description": "3q26 rearrangement",
        "patterns": [
            r"3q26",
            r"q26",
            r"t\s*\(\s*3\s*;",
            r"inv\s*\(\s*3",
        ],
    },
    {
        "feature": "t_9_11",
        "description": "t(9;11)",
        "patterns": [
            r"t\s*\(\s*9\s*;\s*11\s*\)",
            r"t9;11",
        ],
    },
    {
        "feature": "del_12p",
        "description": "del(12p)",
        "patterns": [
            r"del\s*12p",
            r"del\s*\(\s*12\s*\)\s*\(p",
        ],
    },
    {
        "feature": "n_minus",
        "description": "Monosomie (n_minus)",
        "patterns": [r"-(?:[1-9]\d?)\b"],
    },
    {
        "feature": "n_plus",
        "description": "Trisomie / gain (n_plus)",
        "patterns": [r"\+\s*(?:\d+|[xy])(?:\[[^\]]+\])?"],
    },
    {
        "feature": "n_mar",
        "description": "Marqueur / marker (±mar/m)",
        "patterns": [r"[+-]\s*m(?:ar)?[a-z0-9]*(?:\[[^\]]+\])?"],
    },
    {
        "feature": "n_add",
        "description": "Addition (add)",
        "patterns": [r"add\s*\(", r"\badd\b"],
    },
    {
        "feature": "n_t",
        "description": "Translocation (t)",
        "patterns": [r"t\s*\("],
    },
    {
        "feature": "n_der",
        "description": "Chromosome dérivé (der/order)",
        "patterns": [r"der\s*\(", r"order\s*\("],
    },
    {
        "feature": "n_dic",
        "description": "Dicentrique (dic)",
        "patterns": [r"dic\s*\("],
    },
    {
        "feature": "n_inv",
        "description": "Inversion (inv)",
        "patterns": [r"inv\s*\("],
    },
    {
        "feature": "n_del",
        "description": "Suppression (del/ordel)",
        "patterns": [r"del\s*\(", r"ordel\s*\(", r"\bdel\b"],
    },
    {
        "feature": "n_ring",
        "description": "Chromosome annulaire (+r / r())",
        "patterns": [r"r\s*\(", r"\+r[a-z0-9]*(?:\[[^\]]+\])?"],
    },
    {
        "feature": "n_i",
        "description": "Isochromosome (i)",
        "patterns": [r"\bi\s*\("],
    },
    {
        "feature": "n_ins",
        "description": "Insertion (ins)",
        "patterns": [r"\bins\s*\("],
    },
    {
        "feature": "n_dmin",
        "description": "Double minutes (dmin)",
        "patterns": [r"dmin"],
    },
    {
        "feature": "chromosome_range_flag",
        "description": "Plage de caryotype (par ex. 42-45)",
        "patterns": [r"\b\d{2}\s*[-~]\s*\d{2}\b"],
    },
]

TOKEN_SPLIT_REGEX = re.compile(r"[;,/]|\s{2,}")


def _match_rule(text: str, rule: Dict[str, object]) -> List[str]:
    """Return substrings that satisfy the rule patterns."""
    matches: List[str] = []
    if not text:
        return matches
    lowered = text.lower()
    for pattern in rule["patterns"]:
        for found in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            token = text[found.start() : found.end()].strip()
            if token:
                matches.append(token)
    return matches


def _extract_candidate_tokens(text: str) -> List[str]:
    """Rough tokenization to find snippets that look like cytogenetic events."""
    if not text:
        return []
    tokens: List[str] = []
    for raw in TOKEN_SPLIT_REGEX.split(text):
        tok = raw.strip()
        if not tok:
            continue
        if re.search(r"(tri|mono|del|add|\+|-|t\(|inv)", tok, flags=re.IGNORECASE):
            tokens.append(tok)
    return tokens


def _detect_token_events(text: str) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    """Identify heuristic events, unmatched tokens, and all candidate tokens in the raw cytogenetic string."""
    if not isinstance(text, str):
        if text is None or (hasattr(pd, "isna") and pd.isna(text)):
            text = ""
        else:
            text = str(text)
    normalized_text = CytogeneticFeatureExtraction._normalize_cyto_text(text)
    matched_events: List[Dict[str, str]] = []
    matched_tokens: List[str] = []
    candidate_tokens = _extract_candidate_tokens(text)
    for rule in EVENT_RULES:
        hits = _match_rule(normalized_text, rule)
        for token in hits:
            matched_events.append(
                {
                    "feature": rule["feature"],
                    "token": token,
                    "description": rule["description"],
                }
            )
            matched_tokens.append(token)

    unmatched_tokens: List[str] = []
    matched_lower = [tok.lower() for tok in matched_tokens]
    for token in candidate_tokens:
        token_lower = token.lower()
        normalized_token = CytogeneticFeatureExtraction._normalize_cyto_text(token)
        normalized_lower = normalized_token.lower() if isinstance(normalized_token, str) else token_lower
        if any(mtok in token_lower for mtok in matched_lower) or any(mtok in normalized_lower for mtok in matched_lower):
            continue
        unmatched_tokens.append(token)

    return matched_events, unmatched_tokens, candidate_tokens


def _load_raw_dataframe(split: Split, limit: int | None = None) -> pd.DataFrame:
    """Load the raw clinical dataframe containing CYTOGENETICS text."""
    key = "input_clinical_train" if split == "train" else "input_clinical_test"
    df = pd.read_csv(DATA_PATHS[key])
    if limit is not None:
        df = df.head(limit)
    missing_cols = {"ID", "CYTOGENETICS"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Raw clinical file for {split} must contain columns {missing_cols}, got {df.columns.tolist()}"
        )
    return df[["ID", "CYTOGENETICS"]].copy()


def _load_processed_dataframe(split: Split, limit: int | None = None) -> pd.DataFrame:
    """Load the processed feature matrix."""
    path = PROCESSED_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset missing for {split}: {path}. Run 1_prepare_data.py first."
        )
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit)
    if "ID" not in df.columns:
        raise ValueError(
            f"Processed dataframe {path} must contain an ID column for joins but columns are {df.columns[:10]}..."
        )
    return df


def _detect_expected_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Re-run the cytogenetic feature extraction on raw CYTO strings."""
    expected = CytogeneticFeatureExtraction.extract_cytogenetic_risk_features(raw_df)
    # Keep raw CYTO string for context during reporting.
    expected = expected.merge(raw_df, on="ID", how="left", suffixes=("", "_raw"))
    return expected


def _iter_splits(arg_splits: Iterable[str]) -> List[Split]:
    result: List[Split] = []
    normalized = list(arg_splits) if arg_splits else ["train", "test"]
    if "both" in normalized:
        return ["train", "test"]
    for item in normalized:
        if item not in ("train", "test"):
            raise ValueError(f"Unsupported split '{item}'. Use train, test or both.")
        result.append(item)
    return result


def _compare_split(split: Split, limit: int | None = None) -> Dict[str, object]:
    raw_df = _load_raw_dataframe(split, limit)
    processed_df = _load_processed_dataframe(split, limit)
    expected_df = _detect_expected_features(raw_df)

    # Ensure we only compare available patients (processed_df already aligned to overall dataset)
    comparable_ids = processed_df["ID"].unique()
    expected_df = expected_df[expected_df["ID"].isin(comparable_ids)]

    expected_cols = [c for c in expected_df.columns if c not in {"ID", "CYTOGENETICS"}]
    missing_columns = sorted([c for c in expected_cols if c not in processed_df.columns])
    expected_cols_set = set(expected_cols)

    shared_cols = [c for c in expected_cols if c in processed_df.columns]
    merged = processed_df[["ID", *shared_cols]].merge(
        expected_df[["ID", *shared_cols, "CYTOGENETICS"]], on="ID", suffixes=("_processed", "_expected")
    )

    mismatches: List[Dict[str, object]] = []
    for col in shared_cols:
        proc_col = f"{col}_processed"
        exp_col = f"{col}_expected"
        proc_vals = merged[proc_col].fillna(0)
        exp_vals = merged[exp_col].fillna(0)
        diff_mask = proc_vals.ne(exp_vals)
        if not diff_mask.any():
            continue
        diff_rows = merged.loc[diff_mask, ["ID", "CYTOGENETICS", proc_col, exp_col]]
        for row in diff_rows.itertuples(index=False):
            mismatches.append(
                {
                    "ID": row.ID,
                    "feature": col,
                    "processed_value": getattr(row, proc_col),
                    "expected_value": getattr(row, exp_col),
                    "raw_cytogenetics": row.CYTOGENETICS,
                }
            )

    def _is_active(value: object) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    heur_missed: List[Dict[str, object]] = []
    heur_unknown: List[Dict[str, object]] = []
    per_patient_heuristics: Dict[str, Dict[str, object]] = {}
    expected_tuple_iter = expected_df.itertuples(index=False)
    for row in expected_tuple_iter:
        raw_text = getattr(row, "CYTOGENETICS", "") or ""
        events, unknown_tokens, candidate_tokens = _detect_token_events(raw_text)
        per_patient_heuristics[row.ID] = {
            "raw_text": raw_text,
            "events": events,
            "unknown_tokens": unknown_tokens,
            "candidate_tokens": candidate_tokens,
        }
        for event in events:
            feature = event["feature"]
            if feature not in expected_cols_set:
                continue
            value = getattr(row, feature, 0)
            is_active = _is_active(value)
            if not is_active:
                heur_missed.append(
                    {
                        "ID": row.ID,
                        "feature": feature,
                        "token": event["token"],
                        "description": event["description"],
                        "raw_cytogenetics": raw_text,
                    }
                )
        for token in unknown_tokens:
            heur_unknown.append(
                {
                    "ID": row.ID,
                    "token": token,
                    "raw_cytogenetics": raw_text,
                }
            )

    per_patient_rows: List[Dict[str, object]] = []
    manual_flag_count = 0
    for row in merged.itertuples(index=False):
        heur = per_patient_heuristics.get(
            row.ID,
            {
                "raw_text": getattr(row, "CYTOGENETICS", "") or "",
                "events": [],
                "unknown_tokens": [],
                "candidate_tokens": [],
            },
        )
        expected_active = [col for col in shared_cols if _is_active(getattr(row, f"{col}_expected", 0))]
        processed_active = [col for col in shared_cols if _is_active(getattr(row, f"{col}_processed", 0))]
        missing_features = sorted(set(expected_active) - set(processed_active))
        extra_features = sorted(set(processed_active) - set(expected_active))
        unknown_tokens = heur["unknown_tokens"]
        candidate_tokens = heur["candidate_tokens"]
        heuristic_events = heur["events"]
        event_pairs = [f"{event['token']} -> {event['feature']}" for event in heuristic_events]
        needs_manual_review = bool(missing_features or extra_features or unknown_tokens)
        if needs_manual_review:
            manual_flag_count += 1
        reasons: List[str] = []
        if missing_features:
            reasons.append(
                f"Colonnes attendues absentes: {', '.join(missing_features[:10])}"
                + ("..." if len(missing_features) > 10 else "")
            )
        if extra_features:
            reasons.append(
                f"Colonnes présentes non attendues: {', '.join(extra_features[:10])}"
                + ("..." if len(extra_features) > 10 else "")
            )
        if unknown_tokens:
            reasons.append(
                f"Tokens manuscrits non reconnus: {', '.join(unknown_tokens[:10])}"
                + ("..." if len(unknown_tokens) > 10 else "")
            )

        per_patient_rows.append(
            {
                "ID": row.ID,
                "raw_cytogenetics": getattr(row, "CYTOGENETICS", ""),
                "candidate_tokens": json.dumps(candidate_tokens, ensure_ascii=False),
                "heuristic_events": json.dumps(heuristic_events, ensure_ascii=False),
                "heuristic_events_human": "; ".join(event_pairs),
                "heuristic_unknown_tokens": json.dumps(unknown_tokens, ensure_ascii=False),
                "expected_active_features": json.dumps(expected_active, ensure_ascii=False),
                "processed_active_features": json.dumps(processed_active, ensure_ascii=False),
                "missing_features": json.dumps(missing_features, ensure_ascii=False),
                "extra_features": json.dumps(extra_features, ensure_ascii=False),
                "needs_manual_review": needs_manual_review,
                "manual_review_reasons": "; ".join(reasons),
            }
        )

    report_dir = REPORT_ROOT / split
    report_dir.mkdir(parents=True, exist_ok=True)

    if missing_columns:
        with open(report_dir / "missing_columns.json", "w", encoding="utf-8") as fh:
            json.dump(missing_columns, fh, ensure_ascii=False, indent=2)
    else:
        missing_file = report_dir / "missing_columns.json"
        if missing_file.exists():
            missing_file.unlink()

    def _write_frame(
        path: Path, rows: List[Dict[str, object]], sort_keys: List[str] | None = None, ascending: bool | List[bool] = True
    ) -> None:
        df = pd.DataFrame(rows)
        if df.empty:
            if path.exists():
                path.unlink()
            return
        if sort_keys:
            df = df.sort_values(sort_keys, ascending=ascending)
        df.to_csv(path, index=False)

    _write_frame(report_dir / "value_mismatches.csv", mismatches, ["feature", "ID"])
    _write_frame(report_dir / "heuristic_missed_events.csv", heur_missed, ["feature", "ID"])
    _write_frame(report_dir / "heuristic_unknown_tokens.csv", heur_unknown, ["ID", "token"])
    _write_frame(
        report_dir / "patient_manual_review.csv",
        per_patient_rows,
        ["needs_manual_review", "ID"],
        ascending=[False, True],
    )

    return {
        "split": split,
        "num_patients": len(merged),
        "missing_columns": missing_columns,
        "num_mismatched_features": len(mismatches),
        "num_heuristic_missed": len(heur_missed),
        "num_unknown_tokens": len(heur_unknown),
        "num_manual_flags": manual_flag_count,
        "report_dir": str(report_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit cytogenetic feature coverage by comparing raw strings to engineered columns."
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=["train", "test", "both"],
        default=["train", "test"],
        help="Which splits to audit. Defaults to both train and test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for faster debugging.",
    )
    args = parser.parse_args()

    splits = _iter_splits(args.splits)
    summary = [_compare_split(split, args.limit) for split in splits]

    print("\n=== Cytogenetic Feature Audit Summary ===")
    for item in summary:
        print(
            f"[{item['split']}] patients={item['num_patients']} | missing_columns={len(item['missing_columns'])} | mismatched_rows={item['num_mismatched_features']} | heur_missed={item['num_heuristic_missed']} | unknown_tokens={item['num_unknown_tokens']} | manual_flags={item['num_manual_flags']}"
        )
        if item["missing_columns"]:
            sample = ", ".join(item["missing_columns"][:5])
            print(f"    -> Missing columns (first 5): {sample}")
        print(f"    Reports saved under: {item['report_dir']}")


if __name__ == "__main__":
    main()
