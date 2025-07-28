import pandas as pd

# Chemin du fichier CSV
csv_path = "models/gb_optimization_results_20250726_004629.csv"

# Lecture du fichier
df = pd.read_csv(csv_path)

# Calcul de la moyenne des c_index_ipcw sur les 5 folds pour chaque trial
mean_ipcw_per_trial = df.groupby("trial")["c_index_ipcw"].mean()

# Calcul de la moyenne globale sur tous les folds et tous les trials
mean_ipcw_global = df["c_index_ipcw"].mean()

print("Moyenne des IPCW par trial :")
top_10 = mean_ipcw_per_trial.sort_values(ascending=False).head(10)
print("Top 10 des meilleures moyennes IPCW par trial :")
print(top_10)
print(
    f"\nMoyenne globale des IPCW sur tous les folds et trials : {mean_ipcw_global:.4f}"
)

# Top 10 des meilleurs IPCW sur le fold 0
top_10_fold0 = df[df["fold"] == 0].sort_values("c_index_ipcw", ascending=False).head(10)
print("\nTop 10 des meilleures IPCW sur le fold 0 :")
print(top_10_fold0[["trial", "c_index_ipcw"]])
