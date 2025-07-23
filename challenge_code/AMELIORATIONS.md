# Améliorations du Pipeline de ML

## Nouvelles fonctionnalités

### 1. Préparation des données de test

Le script `1_prepare_data.py` prépare maintenant aussi les données de test et les sauvegarde automatiquement :

- **Fichier créé** : `datasets/enriched_test.csv`
- **Contenu** : Données de test préparées avec les mêmes transformations que l'entraînement
- **Avantage** : Cohérence dans le preprocessing entre train et test

### 2. Sélection de modèle pour les prédictions

Le script `3_predict.py` accepte maintenant un paramètre `--model` pour choisir quel modèle utiliser :

```bash
# Utiliser le meilleur modèle (par défaut)
python 3_predict.py

# Utiliser un modèle spécifique
python 3_predict.py --model cox_alpha1.0_20250723_120432

# Voir tous les modèles disponibles
python list_models.py
```

### 3. Pipeline complet amélioré

Le script `main.py` supporte maintenant la sélection de modèle :

```bash
# Pipeline complet avec meilleur modèle
python main.py

# Pipeline complet avec modèle spécifique
python main.py --model gradient_boosting_n_estimators850_learning_rate0.03

# Seulement les prédictions avec modèle spécifique
python main.py --step-3 --model rsf_n_estimators100_min_samples_split10
```

## Modèles disponibles

Basé sur les fichiers trouvés dans le répertoire `models/` :

1. `cox_alpha1.0_20250723_120432`
2. `coxnet_l1_ratio0.9_alphasNone_n_alphas100_normalizeTrue_max_iter100000_20250723_120531`
3. `rsf_n_estimators100_min_samples_split10_min_samples_leaf5_max_depth5_20250723_120437`
4. `gradient_boosting_n_estimators850_learning_rate0.03160736770883459_max_depth3_subsample0.8118022194771892_min_samples_leaf4_min_samples_split4_20250723_120531`
5. `extra_trees_n_estimators100_min_samples_split6_min_samples_leaf3_max_depth10_max_featuressqrt_20250723_120533`
6. `componentwise_gb_n_estimators100_learning_rate0.1_subsample1.0_20250723_120542`

## Exemples d'utilisation

### Tester un modèle Cox simple
```bash
python 3_predict.py --model cox_alpha1.0_20250723_120432
```

### Tester un modèle Random Survival Forest
```bash
python 3_predict.py --model rsf_n_estimators100_min_samples_split10_min_samples_leaf5_max_depth5_20250723_120437
```

### Tester un modèle Gradient Boosting
```bash
python 3_predict.py --model gradient_boosting_n_estimators850_learning_rate0.03160736770883459_max_depth3_subsample0.8118022194771892_min_samples_leaf4_min_samples_split4_20250723_120531
```

## Fichiers générés

### Lors de la préparation (étape 1)
- `datasets/enriched_train.csv` - Données d'entraînement préparées
- `datasets/enriched_test.csv` - Données de test préparées ✨ **NOUVEAU**

### Lors des prédictions (étape 3)
- `submissions/submission_<model>_<timestamp>.csv` - Prédictions avec timestamp
- `submissions/latest_submission.csv` - Dernières prédictions (lien)
- `submissions/summary_<timestamp>.txt` - Résumé des prédictions

## Dépendances

Toutes les dépendances existantes sont respectées :
- Les imputers d'entraînement sont utilisés pour les données de test
- La structure des features est maintenue cohérente
- Les transformations sont appliquées dans le même ordre

## Pipeline recommandé

1. **Préparation complète** (données train + test)
   ```bash
   python 1_prepare_data.py
   ```

2. **Entraînement des modèles**
   ```bash
   python 2_train_models.py
   ```

3. **Voir les modèles disponibles**
   ```bash
   python list_models.py
   ```

4. **Tester différents modèles**
   ```bash
   python 3_predict.py --model cox_alpha1.0_20250723_120432
   python 3_predict.py --model rsf_n_estimators100_min_samples_split10_min_samples_leaf5_max_depth5_20250723_120437
   # etc.
   ```

5. **Comparer les résultats**
   - Vérifier les fichiers dans `submissions/`
   - Comparer les scores de risque
   - Analyser les résumés générés
