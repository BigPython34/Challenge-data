#!/usr/bin/env python3
"""
Script d'installation automatique des dépendances
Vérifie et installe les packages Python nécessaires
"""
import subprocess
import sys
import pkg_resources
from pathlib import Path


def check_package(package_name):
    """Vérifie si un package est installé"""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


def install_package(package_spec):
    """Installe un package via pip"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_spec
        ], capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def read_requirements():
    """Lit le fichier requirements.txt"""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ Fichier requirements.txt non trouvé")
        return []
    
    requirements = []
    with open(req_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extraire le nom du package (avant >= ou ==)
                package_name = line.split('>=')[0].split('==')[0].split('[')[0]
                requirements.append((package_name, line))
    
    return requirements


def main():
    """Fonction principale"""
    print("📦 INSTALLATION DES DÉPENDANCES")
    print("=" * 40)
    
    requirements = read_requirements()
    if not requirements:
        print("❌ Aucune dépendance trouvée dans requirements.txt")
        return 1
    
    print(f"📋 {len(requirements)} dépendances trouvées")
    print()
    
    missing_packages = []
    installed_packages = []
    
    # Vérifier les packages installés
    for package_name, package_spec in requirements:
        if check_package(package_name):
            print(f"✅ {package_name} - déjà installé")
            installed_packages.append(package_name)
        else:
            print(f"❌ {package_name} - manquant")
            missing_packages.append((package_name, package_spec))
    
    if not missing_packages:
        print(f"\n🎉 Toutes les dépendances sont déjà installées!")
        return 0
    
    print(f"\n📥 Installation de {len(missing_packages)} packages manquants...")
    
    success_count = 0
    failed_packages = []
    
    for package_name, package_spec in missing_packages:
        print(f"📦 Installation de {package_name}...")
        if install_package(package_spec):
            print(f"  ✅ {package_name} installé avec succès")
            success_count += 1
        else:
            print(f"  ❌ Échec de l'installation de {package_name}")
            failed_packages.append(package_name)
    
    print("\n" + "=" * 40)
    print("📊 RÉSUMÉ DE L'INSTALLATION")
    print("=" * 40)
    print(f"✅ Packages déjà installés: {len(installed_packages)}")
    print(f"✅ Packages installés: {success_count}")
    print(f"❌ Échecs d'installation: {len(failed_packages)}")
    
    if failed_packages:
        print(f"\n⚠️  Packages non installés: {', '.join(failed_packages)}")
        print("💡 Essayez de les installer manuellement:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return 1
    else:
        print(f"\n🎉 INSTALLATION TERMINÉE AVEC SUCCÈS!")
        print("✨ Vous pouvez maintenant exécuter: python main.py")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
