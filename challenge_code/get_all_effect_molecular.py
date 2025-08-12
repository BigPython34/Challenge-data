import pandas as pd


# Chemins des fichiers
train_path = "datas/X_train/molecular_train.csv"
train_filled_path = "datas/X_train/molecular_train_filled.csv"
test_path = "datas/X_test/molecular_test.csv"
test_filled_path = "datas/X_test/molecular_test_filled.csv"

# Lecture des fichiers
df_train = pd.read_csv(train_path)
df_train_filled = pd.read_csv(train_filled_path)
df_test = pd.read_csv(test_path)
df_test_filled = pd.read_csv(test_filled_path)

# Affichage du nombre de valeurs vides par colonne pour chaque fichier
print("Valeurs vides par colonne (molecular_train.csv) :")
print(df_train.isnull().sum())
print("\nValeurs vides par colonne (molecular_train_filled.csv) :")
print(df_train_filled.isnull().sum())
print("\nValeurs vides par colonne (molecular_test.csv) :")
print(df_test.isnull().sum())
print("\nValeurs vides par colonne (molecular_test_filled.csv) :")
print(df_test_filled.isnull().sum())

# Liste des valeurs possibles pour EFFECT dans chaque fichier

print("\nValeurs uniques de EFFECT (molecular_train.csv) :")
print(df_train["EFFECT"].unique())
print("\nValeurs uniques de EFFECT (molecular_test.csv) :")
print(df_test["EFFECT"].unique())

# --- Extraction des effets du fichier variant_data_bis.jsonl ---
import json
from collections import Counter

variant_path = "datas/variant_data_bis.jsonl"
effects = set()


with open(variant_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        for v in data.values():
            if v is None:
                continue
            snpeff = v.get("snpeff", {})
            ann = snpeff.get("ann", None)
            if ann is None:
                continue
            # ann peut être un dict ou une liste
            if isinstance(ann, dict):
                eff = ann.get("effect")
                if eff:
                    effects.add(eff)
            elif isinstance(ann, list):
                for ann_item in ann:
                    eff = ann_item.get("effect")
                    if eff:
                        effects.add(eff)

print("\nListe des effets ('effect') trouvés dans variant_data_bis.jsonl :")
print(sorted(effects))
