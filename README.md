# Challenge Data : Leukemia Risk Prediction

Ce projet vise à prédire le risque de leucémie à partir de données cliniques, cytogénétiques et moléculaires en utilisant des techniques d'analyse de survie.

## 🎯 Objectif

Prédire le risque de maladie pour les patients atteints de cancer du sang dans le contexte de sous-types spécifiques de leucémies myéloïdes adultes, en utilisant la métrique **IPCW-C-Index**.

## 📁 Structure du projet

```
challenge_code/
├── datas/                    # Données brutes et intermédiaires
│   ├── X_train/             # Données d'entraînement
│   ├── X_test/              # Données de test
│   └── target_train.csv     # Variables cibles
├── models/                   # Modèles entraînés (sauvegardés automatiquement)
├── results/                  # Résultats, figures, soumissions
├── src/                      # Code source principal
│   ├── config.py            # Configuration globale
│   ├── data/                # Modules de données
│   │   ├── load.py          # Chargement des données
│   │   ├── prepare.py       # Préparation et nettoyage
│   │   └── features.py      # Feature engineering
│   ├── modeling/            # Modules de modélisation
│   │   ├── train.py         # Entraînement des modèles
│   │   ├── evaluate.py      # Évaluation et métriques
│   │   └── predict.py       # Prédictions et soumissions
│   ├── utils/               # Fonctions utilitaires
│   │   └── helpers.py       # Outils divers
│   └── visualization/       # Visualisations
│       └── plots.py         # Graphiques et analyses
├── notebooks/               # Notebooks exploratoires
│   ├── original_exploration.ipynb  # Notebook original
│   └── modular_workflow.ipynb      # Workflow modulaire
├── main.py                  # Script principal
├── requirements.txt         # Dépendances
└── README.md               # Ce fichier
```

## 🚀 Installation et utilisation

### 1. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. Utilisation

#### Pipeline complet
```bash
# Lancer tout le pipeline de ML
python main.py
```

Le pipeline exécute automatiquement :
1. Chargement des données
2. Préparation et ingénierie des features
3. Entraînement de plusieurs modèles (Cox, Random Survival Forest, Gradient Boosting)
4. Évaluation et comparaison des modèles
5. Génération des prédictions et fichiers de soumission
6. Création du rapport de visualisation

#### Tester la structure
```bash
# Vérifier que tous les modules sont correctement importables
python test_structure.py
```

#### Nettoyer les sorties
```bash
# Supprimer tous les fichiers générés (modèles, résultats)
python clean_outputs.py
```

#### Via notebook interactif
Ouvrir `notebooks/modular_workflow.ipynb` dans Jupyter.

#### Utilisation modulaire
```python
from src.data.load import load_all_data
from src.data.prepare import prepare_enriched_dataset
from src.modeling.train import train_and_save_all_models

# Charger les données
data = load_all_data()

# Préparer le dataset
df_enriched, imputer = prepare_enriched_dataset(
    data["clinical_train"], 
    data["molecular_train"], 
    data["target_train"]
)

# Entraîner les modèles
models = train_and_save_all_models(X_train, y_train)
```

## 🧬 Features Engineering

### Données cliniques
- Ratios cellulaires (ANC/WBC, MONO/WBC, etc.)
- Scores composites (cytopénie, prolifération)
- Transformations logarithmiques
- Seuils cliniques (anémie, thrombocytopénie)

### Données cytogénétiques
- Classification des anomalies (favorable/défavorable/intermédiaire)
- Détection d'anomalies spécifiques (monosomie 7, del(5q), etc.)
- Score de risque cytogénétique basé sur ELN 2017

### Données moléculaires
- Présence/absence de mutations dans 12 gènes clés
- Scores d'impact pondérés par VAF
- Co-mutations cliniquement significatives
- Disruption des voies biologiques

## 🤖 Modèles implémentés

1. **Cox Proportional Hazards** - Modèle linéaire de référence
2. **Random Survival Forest** - Modèle d'ensemble basé sur les arbres
3. **Gradient Boosting Survival** - Modèle de boosting optimisé

### Sauvegarde automatique
Tous les modèles sont automatiquement sauvegardés avec leurs paramètres dans le dossier `models/`.

## 📊 Évaluation

- **Métrique principale** : IPCW-C-Index (τ=7 ans)
- **Validation** : Split train/test (80/20)
- **Comparaison automatique** des performances

## 📈 Visualisations

- Matrice de corrélation des features
- Importance des variables
- Comparaison des modèles
- Analyse SHAP pour l'explainabilité
- Distribution des prédictions

## ⚙️ Configuration

Le fichier `src/config.py` centralise :
- Chemins des données
- Paramètres des modèles
- Gènes d'intérêt
- Seed pour la reproductibilité

## 📝 Workflow type

1. **Chargement** des données via `data.load`
2. **Préparation** et enrichissement via `data.prepare` et `data.features`
3. **Entraînement** des modèles via `modeling.train`
4. **Évaluation** via `modeling.evaluate`
5. **Prédictions** et soumissions via `modeling.predict`
6. **Visualisations** via `visualization.plots`

## 🔧 Fonctionnalités avancées

- **Optimisation hyperparamètres** avec Optuna
- **Pipeline modulaire** réutilisable
- **Sauvegarde automatique** des modèles et résultats
- **Gestion des données manquantes**
- **Validation croisée**

## 👥 Auteur

Projet développé dans le cadre du Challenge Data Gustave Roussy x Qube RT.
