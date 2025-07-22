#!/usr/bin/env python3
"""
Script pour nettoyer l'encodage des fichiers Python
Supprime les emojis, caractères spéciaux et remplace les accents
"""

import os
import re
import unicodedata
from pathlib import Path


def remove_accents(text):
    """Supprime les accents d'une chaîne de caractères"""
    # Normalise le texte en NFD (décomposition canonique)
    nfd = unicodedata.normalize("NFD", text)
    # Filtre les caractères de combinaison (accents)
    without_accents = "".join(
        char for char in nfd if unicodedata.category(char) != "Mn"
    )
    return without_accents


def clean_text(text):
    """Nettoie le texte en supprimant les emojis et caractères spéciaux"""

    # Table de remplacement pour les caractères spéciaux courants
    replacements = {
        # Guillemets
        '"': '"',
        '"': '"',
        """: "'",
        """: "'",
        "«": '"',
        "»": '"',
        # Tirets
        "–": "-",
        "—": "-",
        # Espaces
        "\u00a0": " ",  # Espace insécable
        "\u2009": " ",  # Espace fine
        "\u2002": " ",  # Espace demi-cadratin
        # Points de suspension
        "…": "...",
        # Autres symboles
        "©": "(c)",
        "®": "(R)",
        "™": "(TM)",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "°": " degres",
        "±": "+/-",
        "²": "2",
        "³": "3",
        "¹": "1",
    }

    # Appliquer les remplacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Supprimer les accents
    text = remove_accents(text)

    # Supprimer tous les emojis et caractères Unicode spéciaux
    # Regex pour les emojis
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symboles & pictogrammes
        "\U0001f680-\U0001f6ff"  # transport & symboles de carte
        "\U0001f1e0-\U0001f1ff"  # drapeaux (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f900-\U0001f9ff"  # symboles supplémentaires
        "\U00002600-\U000026ff"  # symboles divers
        "\U0001f170-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    text = emoji_pattern.sub("", text)

    # Supprimer d'autres caractères spéciaux courants
    special_chars = [
        "✅",
        "❌",
        "⚠️",
        "📊",
        "💾",
        "🧹",
        "🎯",
        "📈",
        "🧠",
        "🎉",
        "🏁",
        "👥",
        "⭐",
        "💡",
        "🔍",
        "🎨",
        "⚡",
        "🌟",
        "💰",
        "🚀",
        "🛠️",
        "📋",
        "📌",
        "💼",
        "🎲",
        "🔥",
        "⭕",
        "🔴",
        "🟠",
        "🟡",
        "🟢",
        "🔵",
        "🟣",
        "⚪",
        "⚫",
        "🟤",
        "📁",
        "📂",
        "🤖",
        "📤",
        "🐍",
    ]

    for char in special_chars:
        text = text.replace(char, "")

    # Nettoyer les espaces multiples
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s+\n", "\n\n", text)

    # Garder seulement les caractères ASCII imprimables et quelques caractères étendus nécessaires
    # Permettre les caractères de base + quelques accents si nécessaires pour les commentaires
    allowed_chars = set(range(32, 127))  # ASCII imprimables
    allowed_chars.update([10, 13])  # \n et \r

    cleaned_text = ""
    for char in text:
        if ord(char) in allowed_chars or char.isascii():
            cleaned_text += char
        else:
            # Remplacer par un espace si c'est un caractère de séparation
            if unicodedata.category(char).startswith("Z"):
                cleaned_text += " "
            # Sinon ignorer le caractère

    return cleaned_text


def clean_python_file(file_path):
    """Nettoie un fichier Python"""
    print(f"Nettoyage de {file_path}...")

    try:
        # Lire le fichier avec différents encodages possibles
        content = None
        for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"  Erreur: Impossible de lire {file_path}")
            return False

        # Nettoyer le contenu
        original_content = content
        cleaned_content = clean_text(content)

        # Sauvegarder seulement si changé
        if cleaned_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            print("  -> Fichier nettoye et sauvegarde")
            return True
        else:
            print("  -> Aucun changement necessaire")
            return False

    except Exception as e:
        print(f"  Erreur lors du nettoyage de {file_path}: {e}")
        return False


def main():
    """Nettoie tous les fichiers Python essentiels"""
    print("=== NETTOYAGE DE L'ENCODAGE DES FICHIERS PYTHON ===")
    print("Suppression des emojis, caracteres speciaux et accents...")
    print("=" * 60)

    # Fichiers essentiels à nettoyer
    essential_files = [
        "1_prepare_data.py",
        "2_train_models.py",
        "3_predict.py",
        "src/config.py",
        "src/__init__.py",
        "src/utils/helpers.py",
        "src/data/__init__.py",
        "src/data/load.py",
        "src/data/prepare.py",
        "src/data/features.py",
        "src/modeling/__init__.py",
        "src/modeling/train.py",
        "src/modeling/evaluate.py",
        "src/modeling/predict.py",
        "src/visualization/__init__.py",
        "src/visualization/plots.py",
    ]

    cleaned_count = 0
    total_count = 0

    for file_path in essential_files:
        if os.path.exists(file_path):
            total_count += 1
            if clean_python_file(file_path):
                cleaned_count += 1
        else:
            print(f"Fichier non trouve: {file_path}")

    print("\n" + "=" * 60)
    print("NETTOYAGE TERMINE:")
    print(f"  Fichiers verifies: {total_count}")
    print(f"  Fichiers modifies: {cleaned_count}")
    print(f"  Fichiers inchanges: {total_count - cleaned_count}")

    if cleaned_count > 0:
        print("\nLes fichiers ont ete nettoyes et sont maintenant en UTF-8 pur.")
        print("Vous pouvez maintenant executer le pipeline sans problemes d'encodage.")
    else:
        print("\nTous les fichiers etaient deja propres.")


if __name__ == "__main__":
    main()
