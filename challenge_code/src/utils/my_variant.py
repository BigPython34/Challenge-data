import json
import re


def abbreviate_hgvs_p(hgvs_p):
    import re

    if not isinstance(hgvs_p, str) or not hgvs_p.startswith("p."):
        return hgvs_p
    # Remplace X ou x final par *
    hgvs_p = re.sub(r"X$", "*", hgvs_p)
    hgvs_p = re.sub(r"x$", "*", hgvs_p)
    aa3to1 = {
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
    # p.AAA123BBB, p.AAA123B, p.A123BBB, p.A123B, p.AAA123*, p.A123*, p.AAA123X, p.A123X
    match = re.match(r"p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3}|\*|X)", hgvs_p)
    if match:
        aa1, pos, aa2 = match.groups()
        aa1_short = aa3to1.get(aa1, aa1[0]) if len(aa1) > 1 else aa1
        if aa2 in ("*", "X"):
            aa2_short = "*"
        elif len(aa2) > 1:
            aa2_short = aa3to1.get(aa2, aa2[0])
        else:
            aa2_short = aa2
        return f"p.{aa1_short}{pos}{aa2_short}"
    return hgvs_p


# Fichier de correspondance abrégé/long
with open("datas/variant_data.jsonl", "r", encoding="utf-8") as fin, open(
    "datas/variant_data_bis.jsonl", "w", encoding="utf-8"
) as fout, open("datas/protein_abbreviation_map.txt", "w", encoding="utf-8") as fmap:
    fmap.write("ID\tabrégé\tlong\n")
    # Mapping des remplacements demandés
    variant_type_map = {
        "upstream_gene_variant": "2KB_upstream_variant",
        "missense_variant": "non_synonymous_codon",
        "synonymous_variant": "synonymous_codon",
        "initiator_codon_variant": "initiator_codon_change",
        "splice_acceptor_variant&intron_variant": "splice_site_variant",
    }
    for line in fin:
        if not line.strip():
            continue
        record = json.loads(line)
        vid, data = list(record.items())[0]
        if data is None:
            fout.write(json.dumps({vid: data}, ensure_ascii=False) + "\n")
            continue
        ann = data.get("snpeff", {}).get("ann", [])
        # Peut être une liste ou un dict
        anns = ann if isinstance(ann, list) else [ann] if isinstance(ann, dict) else []
        for ann0 in anns:
            # Remplacement des types de variants
            if "variant_type" in ann0:
                vt = ann0["variant_type"]
                if vt in variant_type_map:
                    ann0["variant_type"] = variant_type_map[vt]
            if "effect" in ann0:
                eff = ann0["effect"]
                if eff in variant_type_map:
                    ann0["effect"] = variant_type_map[eff]
            if "hgvs_p" in ann0:
                long_form = ann0["hgvs_p"]
                short_form = abbreviate_hgvs_p(long_form)
                # Ecrit la correspondance si différente
                if long_form != short_form:
                    fmap.write(
                        f"{data.get('id','') or data.get('ID','')}	{short_form}	{long_form}\n"
                    )
                ann0["hgvs_p"] = short_form
        # Replace ann in data
        if isinstance(ann, list) and ann:
            data["snpeff"]["ann"] = anns
        elif isinstance(ann, dict):
            data["snpeff"]["ann"] = anns[0] if anns else ann
        fout.write(json.dumps({vid: data}, ensure_ascii=False) + "\n")
