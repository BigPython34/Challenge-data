import json
import re
import pandas as pd
from typing import Dict


class MyVariantCleaner:
    """
    Nettoie et standardise les données brutes extraites de l'API MyVariant.
    - Standardise les noms d'effets en se basant sur une hiérarchie de gravité.
    - Abrège les notations de changement de protéines (HGVS_p) au format 1 lettre.
    """

    def __init__(self):
        """Initialise les dictionnaires de mapping."""
        self.aa3to1_map = self._get_aa_map()

    def _get_aa_map(self) -> Dict[str, str]:
        """Retourne le dictionnaire de conversion des acides aminés 3 lettres -> 1 lettre."""
        return {
            "Ala": "A",
            "Arg": "R",
            "Asn": "N",
            "Asp": "D",
            "Cys": "C",
            "Gln": "Q",
            "Glu": "E",
            "Gly": "G",
            "His": "H",
            "Ile": "I",
            "Leu": "L",
            "Lys": "K",
            "Met": "M",
            "Phe": "F",
            "Pro": "P",
            "Ser": "S",
            "Thr": "T",
            "Trp": "W",
            "Tyr": "Y",
            "Val": "V",
        }

    def standardize_effect(self, effect_str: str) -> str:
        """
        Applique une logique de standardisation et de priorisation des effets.
        Retourne l'effet le plus grave trouvé dans la chaîne de caractères.
        """
        if pd.isna(effect_str):
            return "unknown"

        effect_str = str(effect_str).lower()

        # Priorité 1: Effets à fort impact (tronquants)
        if "frameshift" in effect_str or "stop_gained" in effect_str:
            return "frameshift_or_stop_gained"
        if "stop_lost" in effect_str:
            return "stop_lost"
        if "start_lost" in effect_str or "initiator_codon" in effect_str:
            return "start_lost"

        # Priorité 2: Effets sur l'épissage (splicing)
        if "splice" in effect_str:
            return "splice_site_variant"

        # Priorité 3: Changements dans le cadre de lecture
        if "inframe" in effect_str:
            return "inframe_variant"

        # Priorité 4: Changement d'acide aminé (le plus courant)
        if "missense" in effect_str or "non_synonymous" in effect_str:
            return "non_synonymous_codon"

        # Priorité 5: Effets silencieux
        if "synonymous" in effect_str:
            return "synonymous_codon"

        # Priorité 6: Régions régulatrices ou non codantes
        if "utr_variant" in effect_str:
            return "utr_variant"
        if "upstream" in effect_str or "downstream" in effect_str:
            return "regulatory_variant"

        # Priorité 7: Effets moins informatifs
        if "intron" in effect_str:
            return "intron_variant"
        if "intergenic" in effect_str:
            return "intergenic_variant"

        # Si aucun des cas ci-dessus, retourner 'other'
        return "other"

    def abbreviate_hgvs_p(self, hgvs_p: str) -> str:
        """Convertit une notation HGVS_p de 3 lettres en 1 lettre."""
        if not isinstance(hgvs_p, str) or not hgvs_p.startswith("p."):
            return hgvs_p

        hgvs_p = re.sub(r"[Xx]$", "*", hgvs_p)
        # Gère les cas comme p.Ala123Gly et p.Ala123*
        match = re.match(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*)", hgvs_p)

        if match:
            aa1, pos, aa2 = match.groups()
            aa1_short = self.aa3to1_map.get(aa1, "?")
            aa2_short = "*" if aa2 == "*" else self.aa3to1_map.get(aa2, "?")
            return f"p.{aa1_short}{pos}{aa2_short}"

        return hgvs_p

    def clean_record(self, record: Dict) -> Dict:
        """Nettoie un seul enregistrement de données MyVariant."""
        if not record:
            return None

        snpeff_data = record.get("snpeff", {}) or {}
        ann_data = snpeff_data.get("ann")

        if not ann_data:
            return record

        annotations = [ann_data] if isinstance(ann_data, dict) else ann_data

        cleaned_annotations = []
        for ann in annotations:
            if isinstance(ann, dict):
                if "effect" in ann:
                    ann["effect"] = self.standardize_effect(ann["effect"])

                if "hgvs_p" in ann:
                    ann["hgvs_p"] = self.abbreviate_hgvs_p(ann["hgvs_p"])

                cleaned_annotations.append(ann)

        if isinstance(ann_data, dict):
            record["snpeff"]["ann"] = (
                cleaned_annotations[0] if cleaned_annotations else {}
            )
        else:
            record["snpeff"]["ann"] = cleaned_annotations

        return record

    def process_raw_jsonl(self, input_path: str, output_path: str):
        """Lit un fichier .jsonl brut, le nettoie et le sauvegarde."""
        print(f"Nettoyage du fichier MyVariant brut : {input_path}")
        cleaned_records = 0
        try:
            with open(input_path, "r", encoding="utf-8") as fin, open(
                output_path, "w", encoding="utf-8"
            ) as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    raw_record = json.loads(line)
                    vid, data = list(raw_record.items())[0]
                    cleaned_data = self.clean_record(data)
                    fout.write(
                        json.dumps({vid: cleaned_data}, ensure_ascii=False) + "\n"
                    )
                    cleaned_records += 1
            print(f"✓ Nettoyage terminé. {cleaned_records} enregistrements traités.")
            print(f"✓ Fichier nettoyé sauvegardé à : {output_path}")
        except FileNotFoundError:
            print(f"❌ ERREUR: Fichier d'entrée non trouvé à '{input_path}'")
        except Exception as e:
            print(f"❌ ERREUR inattendue lors du traitement : {e}")
