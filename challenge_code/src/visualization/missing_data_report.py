import pandas as pd
import os

# Chemins des fichiers
clinical_train_path = os.path.join("datas", "X_train", "clinical_train.csv")
molecular_train_path = os.path.join("datas", "X_train", "molecular_train.csv")
clinical_test_path = os.path.join("datas", "X_test", "clinical_test.csv")
molecular_test_path = os.path.join("datas", "X_test", "molecular_test.csv")
target_train_path = os.path.join("datas", "target_train.csv")

print("--- Analyse des données manquantes ---\n")


def missing_report(df, name):
    print(f"{name} :")
    missing = df.isnull().sum()
    total = len(df)
    for col in df.columns:
        print(f"  {col}: {missing[col]} / {total} ({missing[col]/total:.2%})")
    print()


# Clinical train
clinical_train = pd.read_csv(clinical_train_path)
missing_report(clinical_train, "clinical_train")

# Clinical test
clinical_test = pd.read_csv(clinical_test_path)
missing_report(clinical_test, "clinical_test")

# Molecular train (format long, group by ID)
molecular_train = pd.read_csv(molecular_train_path)
print("molecular_train :")
for col in molecular_train.columns:
    missing = molecular_train[col].isnull().sum()
    total = len(molecular_train)
    print(f"  {col}: {missing} / {total} ({missing/total:.2%})")
print()

# Molecular test
molecular_test = pd.read_csv(molecular_test_path)
print("molecular_test :")
for col in molecular_test.columns:
    missing = molecular_test[col].isnull().sum()
    total = len(molecular_test)
    print(f"  {col}: {missing} / {total} ({missing/total:.2%})")
print()

# Target train
target_train = pd.read_csv(target_train_path)
print("target_train :")
for col in target_train.columns:
    missing = target_train[col].isnull().sum()
    total = len(target_train)
    print(f"  {col}: {missing} / {total} ({missing/total:.2%})")

# Nombre de données censurées (OS_STATUS == 0)
if "OS_STATUS" in target_train.columns:
    censored = (target_train["OS_STATUS"] == 0).sum()
    total = len(target_train)
    print(
        f"\nNombre de données censurées (OS_STATUS == 0): {censored} / {total} ({censored/total:.2%})"
    )
