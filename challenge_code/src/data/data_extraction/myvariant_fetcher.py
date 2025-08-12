import asyncio
import aiohttp
import json
import pandas as pd
from typing import List, Dict, Set
from tqdm.asyncio import tqdm_asyncio  # Pour une belle barre de progression asynchrone


class MyVariantFetcher:
    """
    Récupère de manière asynchrone et efficace les annotations de variants
    depuis l'API MyVariant.info.
    """

    def __init__(self, concurrency_limit: int = 50, retries: int = 3):
        self.concurrency_limit = concurrency_limit
        self.retries = retries
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    def _generate_variant_ids_from_files(self, file_paths: List[str]) -> Set[str]:
        """Charge un ou plusieurs fichiers moléculaires et extrait les IDs de variants uniques."""
        print("Génération des IDs de variants uniques à partir des fichiers...")
        all_dfs = []
        for path in file_paths:
            try:
                all_dfs.append(pd.read_csv(path))
            except FileNotFoundError:
                print(f"  [AVERTISSEMENT] Fichier non trouvé, ignoré : {path}")
                continue

        if not all_dfs:
            print("  [ERREUR] Aucun fichier de données valide n'a été trouvé.")
            return set()

        combined_df = pd.concat(all_dfs).dropna(subset=["CHR", "START", "REF", "ALT"])

        def make_id(row):
            try:
                # Gérer le cas où CHR est 'X', 'Y', 'M' etc.
                chr_val = str(row["CHR"]).replace("chr", "")
                return f"chr{chr_val}:g.{int(row['START'])}{row['REF']}>{row['ALT']}"
            except (ValueError, TypeError):
                return None

        variant_ids = combined_df.apply(make_id, axis=1).dropna().unique()
        print(f"✓ {len(variant_ids)} IDs de variants uniques trouvés.")
        return set(variant_ids)

    async def _fetch_one(self, session: aiohttp.ClientSession, vid: str) -> Dict:
        """Récupère l'annotation pour un seul variant avec gestion des erreurs et des réessais."""
        url = f"https://myvariant.info/v1/variant/{vid}?fields=snpeff,vcf"
        async with self.semaphore:
            for attempt in range(self.retries):
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()  # Lève une exception pour les erreurs 4xx/5xx
                        data = await response.json()
                        return {vid: data}
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    json.JSONDecodeError,
                ) as e:
                    print(
                        f"  [AVERTISSEMENT] Échec pour {vid} (essai {attempt+1}/{self.retries}): {e}"
                    )
                    if attempt < self.retries - 1:
                        await asyncio.sleep(1 + attempt)  # Délai exponentiel simple
                    else:
                        print(
                            f"  [ERREUR] Échec final pour {vid} après {self.retries} essais."
                        )
                        return {vid: None}  # Retourner None en cas d'échec final

    async def _fetch_all_and_save(self, variant_ids: Set[str], output_path: str):
        """Orchestre la récupération de tous les variants et sauvegarde en .jsonl."""
        print(f"Récupération des annotations pour {len(variant_ids)} variants...")
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_one(session, vid) for vid in variant_ids]

            with open(output_path, "w", encoding="utf-8") as f:
                # Utiliser tqdm_asyncio pour la barre de progression
                for future in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                    result = await future
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def run(self, input_files: List[str], output_path: str):
        """
        Méthode principale pour exécuter l'ensemble du processus de récupération.

        Args:
            input_files (List[str]): Liste des chemins vers les fichiers moléculaires bruts.
            output_path (str): Chemin vers le fichier .jsonl de sortie.
        """
        variant_ids = self._generate_variant_ids_from_files(input_files)
        if not variant_ids:
            return

        # Lancer la boucle d'événements asynchrone
        asyncio.run(self._fetch_all_and_save(variant_ids, output_path))
        print(
            f"\n✓ Récupération terminée. Données brutes sauvegardées dans : {output_path}"
        )
