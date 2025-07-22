# 🧬 Challenge Data : Leukemia Risk Prediction

## 🎯 Objectif

Prédire le risque de leucémie aiguë myéloïde (LAM) à partir de données cliniques, cytogénétiques et moléculaires en utilisant des techniques avancées d'analyse de survie et des modèles de deep learning.

**Métrique principale** : IPCW-C-Index (τ=7 ans) pour l'évaluation du risque de récidive.

## 🚀 Nouveautés & Améliorations

- ✨ **Pipeline unifié** : Scripts modulaires (1_, 2_, 3_) + architecture MVC
- 🤖 **Modèles avancés** : Intégration PyCox DeepSurv + modèles classiques
- 📊 **Feature Engineering** : 100+ features enrichies (scores composites, ratios)
- 🔧 **Optimisation** : Hyperparamètres optimisés avec Optuna
- 📈 **Visualisations** : Graphiques interactifs et analyses SHAP
- 🛡️ **Robustesse** : Gestion avancée des données manquantes
- 📦 **Modularité** : Code réutilisable et maintenable

## 📁 Structure du projet

```
challenge_code/
├── 1_prepare_data.py         # 🔧 Préparation des données
├── 2_train_models.py         # 🎯 Entraînement des modèles
├── 3_predict.py              # 📊 Génération des prédictions
├── main.py                   # 🚀 Orchestrateur principal
├── datas/                    # 📂 Données sources
│   ├── X_train/             # Données d'entraînement
│   ├── X_test/              # Données de test
│   └── target_train.csv     # Variables cibles
├── datasets/                 # 📈 Données enrichies
│   ├── enriched_train.csv   # Dataset enrichi d'entraînement
│   ├── enriched_test.csv    # Dataset enrichi de test
│   └── training_dataset.pkl # Dataset final pour ML
├── models/                   # 🤖 Modèles individuels (auto-save)
├── submissions/              # 📤 Fichiers de soumission
├── src/                      # 💻 Code source modulaire
│   ├── config.py            # ⚙️ Configuration centralisée
│   ├── data/                # 📊 Modules de gestion des données
│   ├── modeling/            # 🧠 Modules de modélisation
│   ├── utils/               # 🔧 Utilitaires
│   └── visualization/       # 📈 Visualisations avancées
├── notebooks/               # 📓 Analyses exploratoires
├── requirements.txt         # 📋 Dépendances
└── README.md               # 📖 Documentation

```

## 🚀 Utilisation

### Démarrage rapide (Nouveau !)

```bash
# 🚀 Script de démarrage interactif (RECOMMANDÉ)
cd challenge_code
python start.py
```

Le script `start.py` vous guide à travers toutes les étapes :
- ✅ Vérifications préliminaires
- 📦 Installation automatique des dépendances  
- 🧪 Tests de structure
- 🚀 Exécution du pipeline
- 🧹 Nettoyage des sorties

### Pipeline complet

```bash
# Installation automatique des dépendances
python install_deps.py

# Exécution complète du pipeline
python main.py
```

**Le pipeline exécute automatiquement :**
1. **Préparation** → Chargement, nettoyage, feature engineering (100+ features)
2. **Entraînement** → 7 modèles de survie (Cox, RSF, DeepSurv, etc.)
3. **Prédiction** → Génération fichier de soumission + analyses

### Scripts individuels

```bash
# Étape par étape (optionnel)
python 1_prepare_data.py    # Préparation des données
python 2_train_models.py    # Entraînement des modèles  
python 3_predict.py         # Génération des prédictions
```

### Outils utilitaires

```bash
# 🧪 Vérifier la structure et les dépendances
python test_structure.py

# 📊 Rapport d'état du projet (nouveau !)
python project_report.py

# 🧹 Nettoyer les sorties
python clean_outputs.py
```

### Modules Python (Usage avancé)

```python
from src.data.load import load_all_data
from src.modeling.train import train_and_save_all_models
from src.modeling.predict import generate_final_submission

# Workflow personnalisé
data = load_all_data()
models = train_and_save_all_models(X_train, y_train)
submission = generate_final_submission(models, X_test)
```

## 🧬 Features Engineering Avancé

### 📊 Données Cliniques (45+ features)
- **Ratios cellulaires** : ANC/WBC, MONO/WBC, BASO/WBC
- **Scores composites** : Cytopénie, prolifération, inflammation
- **Transformations** : log, sqrt, interactions
- **Seuils cliniques** : Anémie, thrombocytopénie, leucopénie

### 🧬 Données Cytogénétiques (25+ features)
- **Classification ELN 2017** : Favorable/Intermédiaire/Défavorable
- **Anomalies spécifiques** : Monosomie 7, del(5q), t(8;21)
- **Scores de complexité** : Nombre d'anomalies, caryotype complexe
- **Features binaires** : Présence d'anomalies majeures

### 🔬 Données Moléculaires (35+ features)
- **Mutations dans 12 gènes** : TP53, FLT3, NPM1, DNMT3A, etc.
- **Scores VAF pondérés** : Impact clinique des mutations
- **Co-mutations** : Interactions génétiques significatives
- **Voies biologiques** : Méthylation, signalisation, épigénétique

## 🤖 Modèles Implémentés

### Modèles Classiques
1. **Cox Proportional Hazards** - Baseline linéaire
2. **CoxNet** - Régularisation L1/L2 (Elastic Net)
3. **Random Survival Forest** - Ensemble d'arbres
4. **Extra Survival Trees** - Arbres extrêmement randomisés
5. **Gradient Boosting Survival** - Boosting optimisé
6. **Component-wise Gradient Boosting** - Sélection de features

### Modèles Deep Learning
7. **PyCox DeepSurv** - Réseau de neurones pour la survie

### Optimisation & Évaluation
- **Hyperparamètres** : Optimisation automatique avec Optuna
- **Validation** : Split stratifié train/test
- **Métriques** : IPCW-C-Index, Concordance Index
- **Sauvegarde** : Tous les modèles sont automatiquement sauvés

## 📈 Analyses & Visualisations

- 🔍 **Exploration** : Distribution des variables, corrélations
- 📊 **Performance** : Comparaison des modèles, courbes ROC
- 🎯 **Importance** : Features les plus prédictives
- 🧠 **Explainabilité** : Analyses SHAP (optionnel)
- 📈 **Prédictions** : Distribution des scores de risque

## ⚙️ Configuration & Paramètres

Le fichier `src/config.py` centralise :
- **Chemins** : Données, modèles, résultats
- **Hyperparamètres** : Tous les modèles
- **Gènes d'intérêt** : 12 gènes clés pour l'AML
- **Reproductibilité** : Seed fixe pour tous les algorithmes

## 🔧 Outils & Scripts Utilitaires

```bash
# 🚀 Script de démarrage interactif (menu guidé)
python start.py

# 📦 Installation automatique des dépendances
python install_deps.py

# 🧪 Vérifier la structure du projet et les imports
python test_structure.py

# 📊 Rapport d'état complet du projet
python project_report.py

# 🧹 Nettoyer tous les fichiers générés
python clean_outputs.py
```

## 📊 Sorties Générées

```
challenge_code/
├── models/                   # 🤖 Modèles individuels (.pkl, .joblib)
│   ├── model_package.pkl    # Package complet des 7 modèles
│   ├── best_model.joblib    # Meilleur modèle individuel
│   └── model_info.txt       # Performances et métadonnées
├── submissions/             # 📤 Fichiers de soumission (.csv)
│   ├── latest_submission.csv
│   └── submission_*_timestamp.csv
├── datasets/                # 📊 Données enrichies
│   ├── enriched_train.csv   # Dataset d'entraînement (103 features)
│   ├── enriched_test.csv    # Dataset de test (71 features)
│   └── training_dataset.pkl # Dataset final pour ML
└── notebooks/               # 📓 Analyses exploratoires
```

## �️ Technologies Utilisées

### Core ML
- **scikit-survival** : Modèles de survie classiques
- **PyCox** : Deep learning pour la survie
- **Optuna** : Optimisation d'hyperparamètres
- **pandas/numpy** : Manipulation de données

### Visualisation & Analyse
- **matplotlib/seaborn** : Graphiques
- **SHAP** : Explainabilité (optionnel)
- **Jupyter** : Notebooks interactifs

## 🎯 Workflow Complet

1. **Chargement** → `src.data.load` → Données brutes
2. **Préparation** → `src.data.prepare` → Nettoyage + imputation
3. **Features** → `src.data.features` → Engineering avancé
4. **Entraînement** → `src.modeling.train` → 7 modèles de survie
5. **Évaluation** → `src.modeling.evaluate` → Métriques + comparaison
6. **Prédiction** → `src.modeling.predict` → Soumission finale
7. **Visualisation** → `src.visualization.plots` → Analyses graphiques

## � Notes Importantes

- **Données volumineuses** : Les modèles (*.pkl) sont ignorés par Git
- **Reproductibilité** : Seed fixe pour tous les algorithmes aléatoires
- **Performance** : Pipeline optimisé pour CPU (GPU optionnel pour DeepSurv)
- **Mémoire** : ~2-4 GB RAM recommandés pour l'entraînement complet

## 📝 Changelog

### Version 2.0 (Janvier 2025)
- ✨ **Nouveau** : Pipeline unifié avec scripts numérotés
- 🤖 **Ajout** : Modèles PyCox DeepSurv
- 📊 **Amélioration** : 100+ features enrichies
- 🔧 **Optimisation** : Hyperparamètres avec Optuna
- 📦 **Refactoring** : Architecture modulaire propre
- 🛡️ **Robustesse** : Meilleure gestion des erreurs

## 👥 Auteur

**Challenge Data** - Gustave Roussy x Qube RT  
Prédiction du risque de leucémie aiguë myéloïde (AML)
