import pandas as pd
import numpy as np
import plotly.express as px
import os


def analyze_feature_correlations(data_path: str, threshold: float = 0.6):
    """
    Charge un jeu de données, calcule la matrice de corrélation,
    génère une heatmap interactive et liste les paires fortement corrélées.
    """
    print("=" * 80)
    print("  ANALYSE DE LA CORRÉLATION DES FEATURES")
    print("=" * 80)

    # --- 1. CHARGEMENT DES DONNÉES ---
    print(f"\n[STEP 1/3] Chargement du jeu de données depuis : {data_path}")
    try:
        df = pd.read_csv(data_path)
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        print(
            f"   -> Données chargées : {df.shape[0]} échantillons, {df.shape[1]} features."
        )
    except FileNotFoundError:
        print(
            f"   [ERREUR] Fichier non trouvé. Assurez-vous que le chemin est correct."
        )
        return

    # --- 2. CALCUL DE LA MATRICE DE CORRÉLATION ---
    print("\n[STEP 2/3] Calcul de la matrice de corrélation...")
    df = df.drop(columns=["CENTER_GROUP"])
    corr_matrix = df.corr()
    print("   -> Matrice de corrélation calculée.")

    # --- 3. IDENTIFICATION DES PAIRES FORTEMENT CORRÉLÉES ---
    print(
        f"\n[STEP 3/3] Identification des paires de features avec |corrélation| > {threshold}..."
    )

    # Créer un masque pour ne regarder que la partie supérieure de la matrice (éviter les doublons)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Trouver les paires qui dépassent le seuil
    highly_correlated_pairs = upper_triangle.stack().reset_index()
    highly_correlated_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    strong_pairs = highly_correlated_pairs[
        highly_correlated_pairs["Correlation"].abs() > threshold
    ]

    if not strong_pairs.empty:
        print(f"   -> {len(strong_pairs)} paires fortement corrélées trouvées :")
        # Trier par la valeur absolue de la corrélation pour voir les plus fortes en premier
        strong_pairs_sorted = strong_pairs.sort_values(
            by="Correlation", ascending=False, key=abs
        )

        with pd.option_context("display.max_rows", None):
            print(strong_pairs_sorted)

        # Sauvegarder les paires fortement corrélées dans un CSV
        os.makedirs("reports", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        csv_path = os.path.join(
            "reports",
            f"highly_correlated_pairs_{base_name}_thr{str(threshold).replace('.', '_')}.csv",
        )
        strong_pairs_sorted.to_csv(csv_path, index=False)
        print(f"   ✓ Paires fortement corrélées sauvegardées dans : {csv_path}")
    else:
        print(
            "   -> Aucune paire de features fortement corrélée trouvée (c'est une bonne nouvelle !)."
        )
        # Créer un fichier CSV vide (avec en-têtes) pour garder une trace
        os.makedirs("reports", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        csv_path = os.path.join(
            "reports",
            f"highly_correlated_pairs_{base_name}_thr{str(threshold).replace('.', '_')}.csv",
        )
        pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"]).to_csv(
            csv_path, index=False
        )
        print(
            f"   ✓ Fichier CSV vide créé (aucune paire au-dessus du seuil) : {csv_path}"
        )

    # --- (Optionnel) Générer une heatmap des 30 features les plus importantes ---
    # Pour cela, il nous faudrait un modèle entraîné. On peut se baser sur la variance pour une approximation.
    print(
        "\n[BONUS] Génération d'une heatmap interactive des features les plus variables..."
    )
    try:
        # Sélectionner les 30 features avec la plus grande variance (souvent les plus informatives)
        top_n = 30
        top_features = df.var().sort_values(ascending=False).head(top_n).index
        corr_subset = df[top_features].corr()

        fig = px.imshow(
            corr_subset,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",  # Red-Blue reversed
            zmin=-1,
            zmax=1,
            title=f"Heatmap de Corrélation des {top_n} Features les plus Variables",
        )

        # Sauvegarder en fichier HTML
        os.makedirs("reports", exist_ok=True)
        report_path = "reports/correlation_heatmap.html"
        fig.write_html(report_path)
        print(f"   ✓ Heatmap interactive sauvegardée dans : {report_path}")
        print("     Ouvrez ce fichier dans votre navigateur pour explorer.")

    except Exception as e:
        print(f"   [AVERTISSEISSEMENT] Échec de la génération de la heatmap : {e}")


if __name__ == "__main__":
    # Spécifiez ici le chemin vers le fichier que vous voulez analyser
    TRAIN_DATA_PATH = "datasets_processed/X_test_processed.csv"
    analyze_feature_correlations(data_path=TRAIN_DATA_PATH)
    TEST_DATA_PATH = "datasets_processed/X_test_processed.csv"
    analyze_feature_correlations(data_path=TEST_DATA_PATH)
