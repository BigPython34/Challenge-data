import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import featuretools as ft
import re
import json
import math
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.models import Model
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def process_outliers(df, threshold, multiplier=1.5):
    """
    Traite les outliers d'un DataFrame en les limitant (winsorizing) selon la méthode IQR.
    
    Pour chaque colonne numérique, on calcule le premier (Q1) et le troisième (Q3) quartile,
    puis on détermine l'intervalle acceptable [Q1 - multiplier * IQR, Q3 + multiplier * IQR].
    Les valeurs en dehors de cet intervalle sont remplacées par la valeur limite correspondante.
    
    Parameters:
        df (pd.DataFrame): Le DataFrame à traiter.
        multiplier (float): Le facteur multiplicateur pour l'IQR (par défaut 1.5).
        
    Returns:
        pd.DataFrame: Le DataFrame avec les outliers traités.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df_out[col].quantile(threshold)
        Q3 = df_out[col].quantile(1-threshold)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        # Limiter (clip) les valeurs en dehors de l'intervalle
        df_out[col] = df_out[col].clip(lower_bound, upper_bound)
        print("Outliers removed for column:", col)
    return df_out

def create_entity(params):

    clinical_train = pd.read_csv("data/X_train/clinical_train.csv")
    molecular_train = pd.read_csv("data/X_train/molecular_train.csv")

    clinical_test = pd.read_csv("data/X_test/clinical_test.csv")
    molecular_test = pd.read_csv("data/X_test/molecular_test.csv")

    clinical = pd.concat([clinical_train, clinical_test]).reset_index(drop=True)
    molecular = pd.concat([molecular_train, molecular_test]).reset_index(drop=True)

    additional_list = params.get('additional', [])
    molecular = add_myvariant_data(molecular, additional_list)


    es = ft.EntitySet(id="data")

    es = es.add_dataframe(
        dataframe_name='clinical',
        dataframe=clinical,
        index='ID' 
    )

    es = es.add_dataframe(
        dataframe_name='molecular',
        dataframe=molecular,
        index='index'   
    )

    es.add_relationship(
        parent_dataframe_name='clinical',
        parent_column_name='ID',
        child_dataframe_name='molecular',
        child_column_name='ID'
    )

    return es

def main_preprocess(data, params):

    clinical_process = params.get('clinical', [])
    molecular_process = params.get('molecular', [])
    merge_process = params.get('merge', [])
    # data = create_entity()
    # clinical_process = ["CYTOGENETICS"]
    # molecular_process = ["GENE"]
    # merge_process = "featureto"

    found = False

    if "CYTOGENETICS" in clinical_process:
        data['clinical'] = parse_cytogenetics_column(data['clinical'].reset_index(drop=True), column_name='CYTOGENETICS')
    
    elif "CYTOGENETICSv2" in clinical_process:
        data['clinical'] = parse_cytogenetics_column_v2(data['clinical'].reset_index(drop=True), column_name='CYTOGENETICS')

    elif "CYTOGENETICSv3" in clinical_process:
        data['clinical'] = parse_cytogenetics_column_v3(data['clinical'].reset_index(drop=True), column_name='CYTOGENETICS')

    elif "CYTOGENETICSv4" in clinical_process:
        data['clinical'] = parse_cytogenetics_column_v4(data['clinical'].reset_index(drop=True), column_name='CYTOGENETICS')

    else:
        data['clinical'] = data['clinical'].drop(columns=['CYTOGENETICS'])

    if "CENTER" in clinical_process:
        data['clinical'] = create_one_hot(data['clinical'], id_col='ID', ref_col='CENTER', min_count=0, rare_label='gene_other')




    if "featuretools" in merge_process:
        X, features_defs = ft.dfs(entityset=data, target_dataframe_name="clinical")
        found = True
    else:
        X = None
    
    if "gpt" in merge_process:
        gpt = preprocess_mutation_data(data['molecular'])
        if X is None:
            X = data['clinical'].set_index('ID')

        X = gpt.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')
        found = True
    
    if not found:
        X = data['clinical'].set_index('ID')

    if "GENE" in molecular_process:
        #gene_emb = create_embedding_features(data['molecular'], id_col='ID', ref_col='GENE', embedding_dim=16, min_count=50, rare_label='gene_other')
        gene = create_one_hot(data['molecular'], id_col='ID', ref_col='GENE', min_count=50, rare_label='gene_other')
        #X = gene_emb.merge(X, left_index=True, right_index=True, how='right')
        X = gene.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    if "EFFECT" in molecular_process:
        effect = create_one_hot(data['molecular'], id_col='ID', ref_col='EFFECT', min_count=20, rare_label='effect_other')
        X = effect.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    if "CHR" in molecular_process:
        chr = create_one_hot(data['molecular'], id_col='ID', ref_col='CHR', min_count=0, rare_label='chr_other')
        X = chr.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    if "REF" in molecular_process:
        ref = count_bases_per_id(data['molecular'], id_col='ID', ref_col='REF')
        X = ref.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    if "ALT" in molecular_process:
        alt = count_bases_per_id(data['molecular'], id_col='ID', ref_col='ALT')
        X = alt.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    for prcs in clinical_process:
        if '/' in prcs:
            num_col, den_col = prcs.split('/')
            X = add_ratio_column(X, num_col, den_col)

        if '-' in prcs:
            col_to_soust, col_soust = prcs.split('-')
            X = add_soustrac_column(X, col_soust, col_to_soust)

        if 'log' in prcs:
            col = prcs.split('log')[1]
            X[prcs] = np.log(X[col]+1)
            X.drop(col, axis=1, inplace=True)

        if '<' in prcs:
            print(prcs)
            col1, col2 = prcs.split('<')
            X[prcs] = (X[col1] < X[col2]).astype(int)

        if '+' in prcs:
            col1, col2 = prcs.split('+')
            X[prcs] = X[col1] + X[col2]

        if '*' in prcs:
            col1, col2 = prcs.split('*')
            X[prcs] = X[col1] * X[col2]


    for prcs in molecular_process:
        if '/' in prcs:
            num_col, den_col = prcs.split('/')
            X = add_ratio_column(X, num_col, den_col)

        if '-' in prcs:
            col_to_soust, col_soust = prcs.split('-')
            soust = add_soustrac_column(data['molecular'], col_soust, col_to_soust, molecular=True)
            X = soust.merge(X, left_on='ID', right_index=True, how='right').set_index('ID')

    X = process_categories(X, method="del")

    if 'outliers' in params:
        X = process_outliers(X, **params['outliers'])
    return X

def process_categories(df, method="del"):
    categorical_columns = df.select_dtypes(include=['category']).columns

    if method == "del":
        return df.drop(categorical_columns, axis=1)
    
    if method == "dummies":
        return pd.get_dummies(df, columns=categorical_columns)

def to_df(df_list, columns):
    return (pd.DataFrame(df_list[i], columns=columns) for i in range(len(df_list)))

def impute_sex(X_train, X_test, X_eval, columns):

    X_train, X_test, X_eval = to_df([X_train, X_test, X_eval], columns)
    """
    Impute les valeurs inconnues (0.5) de la colonne 'sex' dans X_train, X_test et X_eval
    en entraînant un modèle de bagging sur les observations dont le sexe est connu (0 ou 1).
    
    Parameters:
        X_train, X_test, X_eval (pd.DataFrame): DataFrames déjà imputés pour les autres variables.
    
    Returns:
        tuple: (X_train, X_test, X_eval) mis à jour avec la colonne 'sex' imputée.
    """
    # On agrège temporairement les trois datasets pour disposer d'un ensemble complet
    temp = pd.concat([X_train, X_test, X_eval], axis=0)
    
    # Sélectionner les observations où le sexe est connu (0 ou 1)
    known = temp[temp["sex"] != 0.5]
    
    # Si aucune donnée connue n'est présente, on retourne simplement les datasets originaux
    if known.empty:
        return X_train, X_test, X_eval

    # Préparer les données d'entraînement pour prédire 'sex'
    X_model = known.drop(columns=["sex"])
    y_model = known["sex"]
    
    # Création du modèle de bagging avec 500 estimateurs et l'estimateur de base: DecisionTreeClassifier
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=500,
        oob_score=True,
        random_state=42
    )
    model.fit(X_model, y_model)
    print("OOB Score :", model.oob_score_)
    
    # Pour chaque dataset, prédire et mettre à jour les valeurs de 'sex' là où elles sont 0.5
    for dataset in [X_train, X_test, X_eval]:
        mask = dataset["sex"] == 0.5
        if mask.any():
            X_features = dataset.loc[mask].drop(columns=["sex"])
            preds = model.predict(X_features)
            dataset.loc[mask, "sex"] = preds

    return X_train.to_numpy(), X_test.to_numpy(), X_eval.to_numpy()

def process_missing_values(X_train, X_test, X_eval, columns, strategy="median", sex=False):

    

    imputer = SimpleImputer(strategy=strategy)
    X_train, X_test, X_eval = imputer.fit_transform(X_train), imputer.transform(X_test), imputer.transform(X_eval)

    if sex:
        return impute_sex(X_train, X_test, X_eval, columns)
    else:
        return X_train, X_test, X_eval

def parse_cytogenetics(cyto_str):
    """
    Prend en entrée une chaîne décrivant la cytogénétique (ex : "46,xy,del(3)(q26q27)[15]/46,xy[5]")
    et retourne un dictionnaire de features utiles.
    """
    
    if pd.isna(cyto_str):
        # Si la valeur est manquante, on peut retourner un dictionnaire vide
        return {
            'num_subclones': 0, 'sex': None, 'avg_chromosomes': None,
            'total_mitoses': 0, 'num_translocations': 0, 'num_deletions': 0,
            'num_inversions': 0, 'num_duplications': 0, 'num_additions': 0,
            'num_monosomies': 0, 'num_trisomies': 0, 'complexity_score': 0
        }

    # Séparer la description en sous-clones (split sur '/')
    subclones = cyto_str.split('/')
    
    # On initialise des compteurs globaux
    total_mitoses = 0
    total_translocations = 0
    total_deletions = 0
    total_inversions = 0
    total_duplications = 0
    total_additions = 0
    total_monosomies = 0
    total_trisomies = 0
    
    # Liste pour stocker le nombre de chromosomes détectés dans chaque sous-clone
    clone_chromosome_numbers = []
    
    # Pour détecter le sexe
    sex = None
    
    for clone in subclones:
        # Exemple de clone : "46,xy,del(3)(q26q27)[15]" ou "46,xx[10]"
        
        # Extraire le nombre de mitoses dans [x], s'il existe
        mitoses = 0
        mitoses_match = re.search(r'\[(\d+)\]', clone)
        if mitoses_match:
            mitoses = int(mitoses_match.group(1))
        total_mitoses += mitoses
        
        # Retirer la partie [x]
        clone_clean = re.sub(r'\[\d+\]', '', clone)
        
        # Split par virgule
        parts = clone_clean.split(',')
        
        for p in parts:
            p = p.strip().lower()
            
            # Nombre de chromosomes ?
            if re.match(r'^\d+$', p):
                # ex : "46"
                clone_chromosome_numbers.append(int(p))
            
            # Détecter le sexe
            if "xy" in p:
                if sex is None:
                    sex = 1.0
            elif "xx" in p:
                if sex is None:
                    sex = 0.0

            
            # Rechercher anomalies
            if re.search(r't\(\d+;\d+\)', p):
                total_translocations += 1
            if "del(" in p:
                total_deletions += 1
            if "inv(" in p:
                total_inversions += 1
            if "dup(" in p:
                total_duplications += 1
            if "add(" in p:
                total_additions += 1
            
            # Trisomies / monosomies notées +7, -5, etc.
            plus_match = re.search(r'\+(\d+)', p)
            minus_match = re.search(r'\-(\d+)', p)
            if plus_match:
                total_trisomies += 1
            if minus_match:
                total_monosomies += 1
    
    # Nombre de sous-clones
    num_subclones = len(subclones)
    
    # Moyenne (non pondérée) du nombre de chromosomes
    if len(clone_chromosome_numbers) > 0:
        avg_chromosomes = sum(clone_chromosome_numbers) / len(clone_chromosome_numbers)
    else:
        avg_chromosomes = None
    
    complexity_score = (total_translocations + total_deletions + total_inversions +
                        total_duplications + total_additions)

    return {
        'num_subclones': num_subclones,
        'sex': sex if sex is not None else 0.5,
        'avg_chromosomes': avg_chromosomes,
        'total_mitoses': total_mitoses,
        'num_translocations': total_translocations,
        'num_deletions': total_deletions,
        'num_inversions': total_inversions,
        'num_duplications': total_duplications,
        'num_additions': total_additions,
        'num_monosomies': total_monosomies,
        'num_trisomies': total_trisomies,
        'complexity_score': complexity_score
    }

def parse_cytogenetics_column(df, column_name='CYTOGENETICS'):
    """
    Prend un DataFrame `df` et le nom de la colonne cytogénétique `column_name`.
    Retourne un nouveau DataFrame comprenant les features extraites.
    """
    # Appliquer la fonction parse_cytogenetics à chaque ligne
    parsed_series = df[column_name].apply(parse_cytogenetics)
    
    # Convertir la série de dictionnaires en DataFrame
    parsed_df = pd.json_normalize(parsed_series)
    
    # Concaténer avec le DataFrame d'origine (sans dupliquer la colonne de base si vous voulez la garder)
    # Si vous préférez garder la colonne CYTOGENETICS, ne la supprimez pas.
    final_df = pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)

    return final_df

def parse_cytogenetics_v2(cyto_str):
    """
    Parse une chaîne cytogénétique au format ISCN standard et en extrait des features.
    
    La méthode fonctionne de la façon suivante :
      1. Si la chaîne est manquante ou vide, on retourne des valeurs par défaut.
      2. On vérifie si la chaîne contient le mot "complex" (indicateur d'un caryotype complexe).
      3. On divise la chaîne en sous-clones en utilisant le séparateur '/'.
      4. Pour chaque sous-clone, on extrait le nombre de cellules (la valeur entre crochets)
         et on traite chaque partie séparée par des virgules.
      5. On récupère :
         - Le nombre de chromosomes indiqué en première position (si présent).
         - Le sexe (si "xy" ou "xx" est présent) en pondérant selon le nombre de cellules.
         - Les anomalies (translocations, délétions, inversions, duplications, additions)
           via des expressions régulières.
         - Les anomalies numériques (trisomies et monosomies) en recherchant des patterns
           du type "+num" ou "-num".
      6. On calcule ensuite :
         - La moyenne pondérée du nombre de chromosomes.
         - Un score de complexité égal à la somme des anomalies structurelles détectées.
         - Le nombre total de mitoses (somme des nombres entre crochets).
    
    Retourne un dictionnaire contenant toutes ces features.
    """
    
    # Cas particulier : valeur manquante ou chaîne vide
    if pd.isna(cyto_str) or not isinstance(cyto_str, str) or cyto_str.strip() == "":
        return {
            'num_subclones': 0, 'sex': 0.5, 'avg_chromosomes': None,
            'total_mitoses': 0, 'num_translocations': 0, 'num_deletions': 0,
            'num_inversions': 0, 'num_duplications': 0, 'num_additions': 0,
            'num_monosomies': 0, 'num_trisomies': 0, 'complexity_score': 0,
            'complex_flag': False
        }
    
    # Vérification du flag "complex"
    is_complex = 'complex' in cyto_str.lower()

    # Séparation en sous-clones (le séparateur est "/")
    subclones = cyto_str.split('/')
    
    # Initialisation des compteurs (en pondérant par le nombre de cellules de chaque clone)
    total_mitoses      = 0
    total_translocations = 0
    total_deletions      = 0
    total_inversions     = 0
    total_duplications   = 0
    total_additions      = 0
    total_monosomies     = 0
    total_trisomies      = 0

    # Pour la moyenne pondérée du nombre de chromosomes
    sum_chromosomes  = 0
    count_chromosomes = 0

    # Comptage pour déterminer le sexe (on considère "xy" comme masculin et "xx" comme féminin)
    sex_counts = {'xy': 0, 'xx': 0}

    for clone in subclones:
        # Recherche du nombre de cellules dans [x] (poids du clone)
        mitosis_match = re.search(r'\[(\d+)\]', clone)
        clone_weight = int(mitosis_match.group(1)) if mitosis_match else 1
        total_mitoses += clone_weight

        # On retire la partie "[x]" pour faciliter le parsing
        clone_clean = re.sub(r'\[\d+\]', '', clone)
        # Découpage par virgule
        parts = [p.strip() for p in clone_clean.split(',') if p.strip()]

        # La première partie est généralement le nombre de chromosomes
        if parts and re.match(r'^\d+$', parts[0]):
            try:
                chrom_count = int(parts[0])
                sum_chromosomes += chrom_count * clone_weight
                count_chromosomes += clone_weight
            except Exception:
                pass

        # Analyse de chaque partie pour détecter les anomalies
        for part in parts:
            part_lower = part.lower()

            # Détection du sexe
            if 'xy' in part_lower:
                sex_counts['xy'] += clone_weight
            elif 'xx' in part_lower:
                sex_counts['xx'] += clone_weight

            # Comptage des anomalies structurelles avec des expressions régulières :
            # Translocations : ex: t(3;3)
            trans_matches = re.findall(r't\(\d+;\d+\)', part_lower)
            total_translocations += len(trans_matches) * clone_weight

            # Délétions : ex: del(3)(q26q27)
            del_matches = re.findall(r'del\([^)]*\)', part_lower)
            total_deletions += len(del_matches) * clone_weight

            # Inversions : ex: inv(3)(p21q27)
            inv_matches = re.findall(r'inv\([^)]*\)', part_lower)
            total_inversions += len(inv_matches) * clone_weight

            # Duplications : ex: dup(3)(q21q26)
            dup_matches = re.findall(r'dup\([^)]*\)', part_lower)
            total_duplications += len(dup_matches) * clone_weight

            # Additions : ex: add(5)(q31)
            add_matches = re.findall(r'add\([^)]*\)', part_lower)
            total_additions += len(add_matches) * clone_weight

            # Comptage des anomalies numériques (monosomies et trisomies)
            monosomy_matches = re.findall(r'(?<![a-zA-Z\(])\-\d+', part_lower)
            total_monosomies += len(monosomy_matches) * clone_weight

            trisomy_matches = re.findall(r'\+\d+', part_lower)
            total_trisomies += len(trisomy_matches) * clone_weight

    # Calcul de la moyenne pondérée du nombre de chromosomes
    avg_chromosomes = sum_chromosomes / count_chromosomes if count_chromosomes > 0 else None

    # Calcul du "complexity_score" en sommant les anomalies structurelles
    complexity_score = total_translocations + total_deletions + total_inversions + total_duplications + total_additions

    # Détermination du sexe majoritaire (1.0 pour masculin, 0.0 pour féminin, 0.5 si indéterminé ou mixte)
    if sex_counts['xy'] > sex_counts['xx']:
        sex = 1.0
    elif sex_counts['xy'] < sex_counts['xx']:
        sex = 0.0
    else:
        sex = 0.5

    return {
        'num_subclones': len(subclones),
        'sex': sex,
        'avg_chromosomes': avg_chromosomes,
        'total_mitoses': total_mitoses,
        'num_translocations': total_translocations,
        'num_deletions': total_deletions,
        'num_inversions': total_inversions,
        'num_duplications': total_duplications,
        'num_additions': total_additions,
        'num_monosomies': total_monosomies,
        'num_trisomies': total_trisomies,
        'complexity_score': complexity_score,
        'complex_flag': is_complex
    }

def parse_cytogenetics_column_v2(df, column_name='CYTOGENETICS'):
    """
    Applique la fonction parse_cytogenetics à chaque ligne de la colonne spécifiée du DataFrame.
    
    Retourne un nouveau DataFrame qui contient à la fois les données d'origine
    et les features extraites.
    """
    # Applique la fonction à chaque valeur de la colonne
    parsed_series = df[column_name].apply(parse_cytogenetics_v2)
    # Transforme la série de dictionnaires en DataFrame
    parsed_df = pd.json_normalize(parsed_series)
    # Concatène avec le DataFrame d'origine (vous pouvez garder ou supprimer la colonne initiale)
    final_df = pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)
    return final_df

def parse_cytogenetics_v3(cyto_str):
    """
    Transforme une chaîne ISCN (ex: "46,xy,del(3)(q26q27)[15]/46,xy[5]") 
    en un dictionnaire de features exploitables pour la prédiction.
    
    L'output comprend notamment :
      - num_subclones : nombre de sous-clones détectés
      - sex : 1.0 pour masculin, 0.0 pour féminin, 0.5 si ambigu
      - avg_chromosomes : moyenne (non pondérée) du nombre de chromosomes
      - total_mitoses : somme des nombres entre crochets
      - num_translocations, num_deletions, num_inversions, num_duplications, num_additions
      - num_monosomies, num_trisomies
      - complexity_score : somme des anomalies structurelles
      - complex_flag : True si "complex" est présent dans la chaîne
      - mosaicism_index : indice de diversité clonale (0 si un seul clone)
    """
    # Cas où la donnée est manquante ou vide
    if pd.isna(cyto_str) or not isinstance(cyto_str, str) or cyto_str.strip() == "":
        return {
            'num_subclones': 0, 'sex': 0.5, 'avg_chromosomes': None,
            'total_mitoses': 0, 'num_translocations': 0, 'num_deletions': 0,
            'num_inversions': 0, 'num_duplications': 0, 'num_additions': 0,
            'num_monosomies': 0, 'num_trisomies': 0, 'complexity_score': 0,
            'complex_flag': False, 'mosaicism_index': 0
        }
    
    # Vérifier si le terme "complex" est présent dans la chaîne (flag qualitatif)
    complex_flag = 'complex' in cyto_str.lower()
    
    # Séparer la chaîne en sous-clones (le séparateur est "/")
    clones = cyto_str.split('/')
    num_subclones = len(clones)
    
    # Initialisation des compteurs globaux
    total_mitoses = 0
    clone_chromosome_numbers = []  # pour calculer la moyenne des chromosomes
    clone_weights = []             # pour calculer la diversité clonale
    total_translocations = 0
    total_deletions = 0
    total_inversions = 0
    total_duplications = 0
    total_additions = 0
    total_monosomies = 0
    total_trisomies = 0
    sex_counts = {'xy': 0, 'xx': 0}
    
    # Parcours de chaque clone
    for clone in clones:
        clone_str = clone.strip().lower()
        # Extraction du nombre de mitoses (poids du clone) via [x]
        mitosis_match = re.search(r'\[(\d+)\]', clone_str)
        weight = int(mitosis_match.group(1)) if mitosis_match else 1
        clone_weights.append(weight)
        total_mitoses += weight
        
        # Retirer la partie [x] pour faciliter le parsing
        clone_clean = re.sub(r'\[\d+\]', '', clone_str)
        parts = [p.strip() for p in clone_clean.split(',') if p.strip()]
        
        for p in parts:
            # Si la partie est uniquement numérique, elle correspond au nombre de chromosomes
            if re.match(r'^\d+$', p):
                try:
                    clone_chromosome_numbers.append(int(p))
                except:
                    pass
            
            # Détection du sexe
            if 'xy' in p:
                sex_counts['xy'] += weight
            elif 'xx' in p:
                sex_counts['xx'] += weight
                
            # Comptage des anomalies structurelles
            if re.search(r't\(\d+;\d+\)', p):
                total_translocations += 1 * weight
            if 'del(' in p:
                total_deletions += 1 * weight
            if 'inv(' in p:
                total_inversions += 1 * weight
            if 'dup(' in p:
                total_duplications += 1 * weight
            if 'add(' in p:
                total_additions += 1 * weight
                
            # Comptage des anomalies numériques (ex: +7 ou -5)
            if re.search(r'\+\d+', p):
                total_trisomies += 1 * weight
            if re.search(r'\-\d+', p):
                total_monosomies += 1 * weight
    
    # Calcul de la moyenne (non pondérée) du nombre de chromosomes
    avg_chromosomes = (sum(clone_chromosome_numbers) / len(clone_chromosome_numbers)
                       if clone_chromosome_numbers else 46)
    
    # Score de complexité = somme des anomalies structurelles
    complexity_score = (total_translocations + total_deletions + total_inversions +
                        total_duplications + total_additions)
    
    # Détermination du sexe : on privilégie le type majoritaire
    if sex_counts['xy'] > sex_counts['xx']:
        sex = 1.0
    elif sex_counts['xx'] > sex_counts['xy']:
        sex = 0.0
    else:
        sex = 0.5
    
    # Calcul d'un indice de mosaicisme à partir des poids clonaux
    # Ici, on utilise l'entropie de Shannon normalisée pour quantifier la diversité
    if total_mitoses > 0 and len(clone_weights) > 1:
        entropy = -sum((w / total_mitoses) * math.log(w / total_mitoses) for w in clone_weights if w > 0)
        # Normalisation par log(num_subclones) pour obtenir une valeur entre 0 et 1
        mosaicism_index = entropy / math.log(num_subclones) if num_subclones > 1 else 0
    else:
        mosaicism_index = 0
    
    return {
        'num_subclones': num_subclones,
        'sex': sex,
        'avg_chromosomes': avg_chromosomes,
        'total_mitoses': total_mitoses,
        'num_translocations': total_translocations,
        'num_deletions': total_deletions,
        'num_inversions': total_inversions,
        'num_duplications': total_duplications,
        'num_additions': total_additions,
        'num_monosomies': total_monosomies,
        'num_trisomies': total_trisomies,
        'complexity_score': complexity_score,
        'complex_flag': complex_flag,
        'mosaicism_index': mosaicism_index
    }

def parse_cytogenetics_column_v3(df, column_name='CYTOGENETICS'):
    """
    Applique la fonction parse_cytogenetics à chaque ligne de la colonne indiquée
    et retourne un nouveau DataFrame comprenant les features extraites.
    """
    parsed_series = df[column_name].apply(parse_cytogenetics_v3)
    parsed_df = pd.json_normalize(parsed_series)
    final_df = pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)
    return final_df

def parse_cytogenetics_v4(cyto_str):
    """
    Transforme une chaîne ISCN (ex: "46,xy,del(3)(q26q27)[15]/46,xy[5]") 
    en un dictionnaire de features exploitables pour la prédiction.
    
    L'output comprend notamment :
      - num_subclones : nombre de sous-clones détectés
      - sex : 1.0 pour masculin, 0.0 pour féminin, 0.5 si ambigu
      - avg_chromosomes : moyenne (non pondérée) du nombre de chromosomes
      - total_mitoses : somme des nombres entre crochets
      - num_translocations, num_deletions, num_inversions, num_duplications, num_additions
      - num_monosomies, num_trisomies
      - complexity_score : somme des anomalies structurelles
      - complex_flag : True si "complex" est présent dans la chaîne
      - mosaicism_index : indice de diversité clonale (0 si un seul clone)
    """
    # Cas où la donnée est manquante ou vide
    if pd.isna(cyto_str) or not isinstance(cyto_str, str) or cyto_str.strip() == "":
        return {
            'num_subclones': 0, 'sex': 0.5, 'avg_chromosomes': None,
            'total_mitoses': 0, 'num_translocations': 0, 'num_deletions': 0,
            'num_inversions': 0, 'num_duplications': 0, 'num_additions': 0,
            'num_monosomies': 0, 'num_trisomies': 0, 'complexity_score': 0,
            'complex_flag': False, 'mosaicism_index': 0
        }
    
    # Vérifier si le terme "complex" est présent dans la chaîne (flag qualitatif)
    complex_flag = 'complex' in cyto_str.lower()
    
    # Séparer la chaîne en sous-clones (le séparateur est "/")
    clones = cyto_str.split('/')
    num_subclones = len(clones)
    
    # Initialisation des compteurs globaux
    total_mitoses = 0
    clone_chromosome_numbers = []  # pour calculer la moyenne des chromosomes
    clone_weights = []             # pour calculer la diversité clonale
    total_translocations = 0
    total_deletions = 0
    total_inversions = 0
    total_duplications = 0
    total_additions = 0
    total_monosomies = 0
    total_trisomies = 0
    sex_counts = {'xy': 0, 'xx': 0}
    
    # Parcours de chaque clone
    for clone in clones:
        clone_str = clone.strip().lower()
        # Extraction du nombre de mitoses (poids du clone) via [x]
        mitosis_match = re.search(r'\[(\d+)\]', clone_str)
        weight = int(mitosis_match.group(1)) if mitosis_match else 1
        clone_weights.append(weight)
        total_mitoses += weight
        
        # Retirer la partie [x] pour faciliter le parsing
        clone_clean = re.sub(r'\[\d+\]', '', clone_str)
        parts = [p.strip() for p in clone_clean.split(',') if p.strip()]
        
        for p in parts:
            # Si la partie est uniquement numérique, elle correspond au nombre de chromosomes
            if re.match(r'^\d+$', p):
                try:
                    clone_chromosome_numbers.append(int(p))
                except:
                    pass

            weight = 1
            
            # Détection du sexe
            if 'xy' in p:
                sex_counts['xy'] += weight
            elif 'xx' in p:
                sex_counts['xx'] += weight
                
            # Comptage des anomalies structurelles
            if re.search(r't\(\d+;\d+\)', p):
                total_translocations += 1 * weight
            if 'del(' in p:
                total_deletions += 1 * weight
            if 'inv(' in p:
                total_inversions += 1 * weight
            if 'dup(' in p:
                total_duplications += 1 * weight
            if 'add(' in p:
                total_additions += 1 * weight
                
            # Comptage des anomalies numériques (ex: +7 ou -5)
            if re.search(r'\+\d+', p):
                total_trisomies += 1 * weight
            if re.search(r'\-\d+', p):
                total_monosomies += 1 * weight
    
    # Calcul de la moyenne (non pondérée) du nombre de chromosomes
    avg_chromosomes = (sum(clone_chromosome_numbers) / len(clone_chromosome_numbers)
                       if clone_chromosome_numbers else 46)
    
    # Score de complexité = somme des anomalies structurelles
    complexity_score = (total_translocations + total_deletions + total_inversions +
                        total_duplications + total_additions)
    
    # Détermination du sexe : on privilégie le type majoritaire
    if sex_counts['xy'] > sex_counts['xx']:
        sex = 1.0
    elif sex_counts['xx'] > sex_counts['xy']:
        sex = 0.0
    else:
        sex = 0.5
    
    # Calcul d'un indice de mosaicisme à partir des poids clonaux
    # Ici, on utilise l'entropie de Shannon normalisée pour quantifier la diversité
    if total_mitoses > 0 and len(clone_weights) > 1:
        entropy = -sum((w / total_mitoses) * math.log(w / total_mitoses) for w in clone_weights if w > 0)
        # Normalisation par log(num_subclones) pour obtenir une valeur entre 0 et 1
        mosaicism_index = entropy / math.log(num_subclones) if num_subclones > 1 else 0
    else:
        mosaicism_index = 0
    
    return {
        'num_subclones': num_subclones,
        'sex': sex,
        'avg_chromosomes': avg_chromosomes,
        'total_mitoses': total_mitoses,
        'num_translocations': total_translocations,
        'num_deletions': total_deletions,
        'num_inversions': total_inversions,
        'num_duplications': total_duplications,
        'num_additions': total_additions,
        'num_monosomies': total_monosomies,
        'num_trisomies': total_trisomies,
        'complexity_score': complexity_score,
        'complex_flag': complex_flag,
        'mosaicism_index': mosaicism_index
    }

def parse_cytogenetics_column_v4(df, column_name='CYTOGENETICS'):
    """
    Applique la fonction parse_cytogenetics à chaque ligne de la colonne indiquée
    et retourne un nouveau DataFrame comprenant les features extraites.
    """
    parsed_series = df[column_name].apply(parse_cytogenetics_v4)
    parsed_df = pd.json_normalize(parsed_series)
    final_df = pd.concat([df.drop(columns=[column_name]), parsed_df], axis=1)
    return final_df

def create_one_hot(
    df, 
    id_col='ID', 
    ref_col='GENE', 
    min_count=10,
    rare_label='gene_other'
):
    """
    Pour chaque 'ID' unique, crée un One-Hot Encoding des gènes.
    Les gènes qui apparaissent moins de min_count fois dans df
    sont regroupés sous la même catégorie (rare_label).
    
    Paramètres
    ----------
    df : DataFrame
        Contient au moins les colonnes [id_col, gene_col].
    id_col : str
        Nom de la colonne identifiant unique du patient (ID).
    gene_col : str
        Nom de la colonne contenant le gène.
    min_count : int
        Seuil de fréquence minimum en dessous duquel un gène est classé dans rare_label.
    rare_label : str
        Nom de la catégorie pour regrouper les gènes rares.
    
    Retour
    ------
    pivoted_df : DataFrame
        Tableau pivoté de dimension [nb_ID x nb_gènes+1], 
        où chaque gène (ou GENE_AUTRE) est codé 0/1.
        L'index est remis dans une colonne 'ID'.
    """

    # 1) Compter la fréquence de chaque gène
    freq = df[ref_col].value_counts()

    # 2) Identifier les gènes rares
    rare_genes = freq[freq < min_count].index  # index -> liste des gènes
    # => Ce sont tous les gènes qui apparaissent < min_count fois

    # 3) Créer une colonne "gene_aggreg" 
    #    qui remplace les gènes rares par "GENE_AUTRE"
    df[f'{ref_col.lower()}_aggreg'] = df[ref_col].apply(
        lambda g: rare_label if g in rare_genes else g
    )

    # 4) Réaliser le pivot / crosstab
    #    Pour chaque ID, on indique 1 si le gène (agrégé) est présent, 0 sinon.
    pivoted = pd.crosstab(df[id_col], df[f'{ref_col.lower()}_aggreg'])

    # Facultatif : renommer les colonnes pour y faire apparaître "gene_"
    pivoted.columns = [f"{ref_col.lower()}_{str(col)}" for col in pivoted.columns]

    # Remettre l'index (ID) comme colonne
    pivoted.reset_index(inplace=True)

    return pivoted

def count_bases_per_id(df, id_col='ID', ref_col='REF'):
    """
    Agrège les chaînes de caractères de la colonne `ref_col` par `id_col`,
    puis compte le nombre de A, G, C, T. Retourne un DataFrame 
    avec les colonnes [id_col, ref_A, ref_G, ref_C, ref_T].
    
    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant au moins `id_col` et `ref_col`.
    id_col : str
        Nom de la colonne identifiant (ex. 'ID').
    ref_col : str
        Nom de la colonne contenant des lettres (ex. f'{col.lower()}').

    Retour
    ------
    pd.DataFrame
        Un DataFrame de taille [nombre_unique_de_ID x 5], 
        avec les colonnes: [id_col, f'{col.lower()}_A', f'{col.lower()}_G', f'{col.lower()}_C', f'{col.lower()}_T'].
    """
    
    # 1) Copie de sécurité pour éviter de modifier df directement
    df_copy = df.copy()
    
    # 2) Remplacer les NaN par des chaînes vides dans la colonne `ref_col`
    df_copy[ref_col] = df_copy[ref_col].astype(str).fillna('')
    
    # 3) Grouper par `id_col`, puis concaténer toutes les chaînes en une seule
    grouped_strings = df_copy.groupby(id_col)[ref_col].apply(lambda x: ''.join(x))
    
    # 4) Définir une fonction pour compter les occurrences de A, G, C, T
    def count_bases(sequence):
        return pd.Series({
            f'{ref_col.lower()}_A': sequence.count('A'),
            f'{ref_col.lower()}_G': sequence.count('G'),
            f'{ref_col.lower()}_C': sequence.count('C'),
            f'{ref_col.lower()}_T': sequence.count('T')
        })
    
    # 5) Appliquer cette fonction de comptage sur chaque séquence agrégée
    base_counts = grouped_strings.apply(count_bases).reset_index()
    
    # 6) Vous pouvez simplement retourner base_counts,
    #    ou si vous voulez être certain de conserver tous les ID originaux
    #    (y compris ceux n'ayant aucune entrée dans `ref_col`),
    #    on merge avec la liste des ID distincts :
    distinct_ids = df_copy[[id_col]].drop_duplicates()
    result = distinct_ids.merge(base_counts, on=id_col, how='left')
    
    # 7) Optionnel: remplir les NaN (pour les ID sans aucune base) par 0
    result[[f'{ref_col.lower()}_A',f'{ref_col.lower()}_G',f'{ref_col.lower()}_C',f'{ref_col.lower()}_T']] = result[[f'{ref_col.lower()}_A',f'{ref_col.lower()}_G',f'{ref_col.lower()}_C',f'{ref_col.lower()}_T']].fillna(0)
    
    return result

def add_ratio_column(df, col_num, col_den, new_col_name=None):
    """
    Crée dans `df` une nouvelle colonne représentant le ratio col_num / col_den.
    Gère les cas où col_den est 0 ou NaN en attribuant NaN au ratio.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à enrichir.
    col_num : str
        Nom de la colonne faisant office de numérateur.
    col_den : str
        Nom de la colonne faisant office de dénominateur.
    new_col_name : str, optionnel
        Nom de la nouvelle colonne. S'il n'est pas fourni,
        le nom sera 'ratio_{col_num}_{col_den}'.

    Retour
    ------
    pd.DataFrame
        Le même DataFrame (modifié en place) avec la colonne ratio ajoutée.
    """
    if new_col_name is None:
        new_col_name = f"ratio_{col_num}_{col_den}"

    # Copie de sécurité optionnelle (décommentez si vous ne voulez pas modifier le df original)
    # df = df.copy()

    # Pour gérer les cas où le dénominateur est NaN ou 0,
    # on calcule le ratio dans une série, puis on l'assigne.
    ratio_series = pd.Series(np.nan, index=df.index)

    # On peut définir une condition qui identifie où col_den est non-nul, non-NaN
    valid_den = (df[col_den].notna()) & (df[col_den] != 0)

    # Pour les lignes valides, on calcule le ratio
    ratio_series[valid_den] = df.loc[valid_den, col_num] / df.loc[valid_den, col_den]

    # On ajoute la série au DataFrame
    df[new_col_name] = ratio_series

    return df

def add_soustrac_column(df, col_soust, col_to_soust, new_col_name=None, molecular=False):

    if new_col_name is None:
        new_col_name = f"sous_{col_to_soust}_{col_soust}"

    # Copie de sécurité optionnelle (décommentez si vous ne voulez pas modifier le df original)
    # df = df.copy()

    if molecular:
        # Pour les lignes valides, on calcule la soustraction
        df[new_col_name] = (df[col_to_soust] - df[col_soust] +1).fillna(0).astype(int)
        # On fait un groupby sur "ID" et on calcule la moyenne pour chaque ID
        df = df.groupby('ID')[new_col_name].sum().reset_index()
    else:
        # Pour les lignes valides, on calcule la soustraction
        df[new_col_name] = df[col_to_soust] - df[col_soust]

    return df

def log_add_one(df, col):
    """
    Ajoute 1 à toutes les valeurs de la colonne `col` du DataFrame `df`,
    puis applique le logarithme naturel à chaque valeur.
    
    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame contenant la colonne à transformer.
    col : str
        Nom de la colonne à transformer.
    
    Retour
    ------
    pd.Series
        La colonne transformée.
    """
    return np.log(df[col]+1)

def preprocess_mutation_data(df_mutations: pd.DataFrame) -> pd.DataFrame:
    """
    df_mutations : DataFrame contenant les colonnes
        [ID, CHR, START, END, REF, ALT, GENE, PROTEIN_CHANGE, EFFECT, VAF, DEPTH]
    Retourne : Un DataFrame agrégé au niveau du patient, avec des features pour le modèle de survie.
    """
    # -- Étape 1 : création de variables binaires/catégorielles à partir d'EFFECT --
    df_mutations = encode_effect(df_mutations)

    # -- Étape 2 : sélection ou transformation d'autres colonnes (CHR, PROTEIN_CHANGE, etc.) --
    # ex. encoder le chromosome, calculer une taille d'indel si besoin, etc.
    df_mutations = transform_other_cols(df_mutations)

    # -- Étape 3 : agrégation par ID patient --
    df_agg = aggregate_by_patient(df_mutations)

    # -- Étape 4 : (optionnel) normalisation ou filtrage si nécessaire --
    #df_agg = normalize_features(df_agg)

    return df_agg


def encode_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des colonnes binaires ou catégorielles à partir de la colonne EFFECT :
    - stop_gained, frameshift_variant, splice_site_variant, non_synonymous_codon, etc.
    """
    # Exemple : variable binaire "is_truncating" pour frameshift + stop_gained
    truncating_effects = ["stop_gained", "frameshift_variant"]
    df["is_truncating"] = df["EFFECT"].isin(truncating_effects).astype(int)

    # Autres exemples : non_synonymous, splice_site, ...
    df["is_non_synonymous"] = (df["EFFECT"] == "non_synonymous_codon").astype(int)
    df["is_splice_site"] = (df["EFFECT"] == "splice_site_variant").astype(int)

    # Vous pouvez multiplier ce genre de mapping en fonction de votre usage
    return df


def transform_other_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traite d'autres colonnes comme CHR, PROTEIN_CHANGE, etc.
    """
    # (1) Encoder le chromosome de manière catégorielle (CHR = 1, 2, 3, X, etc.)
    #     => Soit on convertit en "chr1, chr2, ..." ou on mappe X, Y en 23, 24...
    df["CHR"] = df["CHR"].astype(str)  # s'assure que c'est du string
    # Exemple de map "X" -> 23, "Y" -> 24 si vous préférez en numérique
    chr_map = {"X": "23", "Y": "24"}  # ou faire l'inverse : "23"->"X" si besoin
    df["CHR"] = df["CHR"].replace(chr_map)
    # Optionnel : on peut laisser CHR en string et faire un one-hot encoding plus tard

    # (2) Calculer la "taille" de la mutation (END - START + 1) si c'est un indel
    df["mut_length"] = (df["END"] - df["START"] + 1).fillna(0).astype(int)

    # (3) Extraire la position d'aa impactée depuis PROTEIN_CHANGE si format "p.E545K"
    #     => p.E545K => on veut la position 545
    #     On peut faire une regex simple : p.([A-Z])(\d+)([A-Z*])
    import re
    def extract_aa_position(prot_change):
        # Prot_change type "p.E545K" -> renvoie 545 (int)
        if not isinstance(prot_change, str):
            return None
        match = re.match(r"p\.[A-Z](\d+).*", prot_change)
        if match:
            return int(match.group(1))
        return None

    df["AA_position"] = df["PROTEIN_CHANGE"].apply(extract_aa_position)

    return df


def aggregate_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les mutations par ID patient pour obtenir un set de features complet.
    Exemple de features :
        - total_mutations
        - mean_vaf, max_vaf
        - nb_truncating
        - nb_non_synonymous
        - nb_unique_genes
        ...
    """

    agg_dict = {
        "is_truncating": "sum",         # nombre de variants truncants
        "is_non_synonymous": "sum",     # nombre de variants non-synonymes
        "is_splice_site": "sum",        # nombre de variants au site de splice
        #"GENE": "nunique",              # nombre de gènes uniques
        #"CHR": ["sum", "mean", "min", "max"],               # nombre de chromosomes touchés 
        #"VAF" : ["sum", "mean", "min", "max", "std", "skew"],   # VAF moyen, max, etc.
        #"DEPTH": ["sum", "mean", "min", "max", "std", "skew"],  # profondeur moyenne, etc.
        #"END": ["sum", "mean", "min", "max"],  # position moyenne de fin
        "mut_length": ["sum", "mean", "max"],  # taille moyenne d'indel
        "AA_position": ["sum", "mean", "min", "max"] # position moyenne d'aa impacté
        # etc.
    }

    # On veut aussi compter le nombre total de mutations, 
    # donc on peut ajouter une colonne dummy pour compter
    df["count_mut"] = 1

    # On agrège
    df_agg = df.groupby("ID").agg(agg_dict)

    # Flatten les multi-index de colonnes
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]

    # Nombre total de mutations
    df_agg["total_mutations"] = df.groupby("ID")["count_mut"].sum()

    # Nombre de gènes distincts
    df_agg["unique_genes"] = df.groupby("ID")["GENE"].nunique()

    # Autres features, ex. fraction de mutations avec VAF > 0.3
    df_agg["frac_vaf_gt_0_3"] = df.groupby("ID").apply(
        lambda group: (group["VAF"] > 0.3).mean()
    )

    df_agg = df_agg.reset_index()

    return df_agg


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionnel : normaliser ou standardiser certaines features continues.
    """
    from sklearn.preprocessing import StandardScaler

    # Sélection des colonnes à normaliser (exclure ID, etc.)
    cols_to_scale = [
        c for c in df.columns 
        if c not in ["ID"]  # On garde ID intact
    ]

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df


def make_variant_id(row):
    return f"chr{row['CHR']}:g.{int(row['START'])}{row['REF']}>{row['ALT']}"

def extract_field(variant, field_path, default=None):
    """
    Parcourt le dictionnaire variant en suivant le chemin indiqué par field_path (liste de clés).
    Retourne la valeur trouvée ou default si une clé n'est pas présente.
    """
    try:
        val = variant
        for key in field_path:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return default
    
def process_exon_field(exon_str):
    """
    Traite le champ exon au format "x/y" et retourne (exon_number, total_exons, exon_ratio).
    Si le format est incorrect, retourne (None, None, None).
    """
    try:
        x, y = exon_str.split('/')
        exon_number = int(x)
        total_exons = int(y)
        exon_ratio = exon_number / total_exons if total_exons != 0 else None
        return exon_number, total_exons, exon_ratio
    except Exception:
        return None, None, None

def add_field(variant_data, df, field_list, id_col='variant_id'):
    """
    Ajoute une colonne au DataFrame df en extrayant les données de variant_data
    selon le chemin field_list. Le champ id_col est utilisé pour faire la jointure.
    """
    col_name = '_'.join(field_list)
    df[col_name] = df[id_col].apply(lambda vid: extract_field(variant_data.get(vid, {}), field_list, None))

    if "exon" in col_name:
        exon_processed = df[col_name].apply(lambda x: process_exon_field(x) if x is not None else (None, None, None))
        df[[f'{col_name}_number', f'{col_name}_total', f'{col_name}_ratio']] = pd.DataFrame(exon_processed.tolist(), index=df.index)

    return df

def add_myvariant_data(df: pd.DataFrame, fields_list: list) -> pd.DataFrame:
    """
    Enrichit le DataFrame `df` avec les données de variantes contenues dans le fichier JSON `variant_data.json`.
    Les champs à ajouter sont spécifiés dans `fields_list`.
    """
    if fields_list == []:
        return df

    required_cols = ['CHR', 'START', 'REF', 'ALT']
    
    # Séparer les lignes valides (sans NaN dans les colonnes importantes) et les autres
    valid_df = df.dropna(subset=required_cols).copy()
    invalid_df = df[df[required_cols].isna().any(axis=1)].copy()
    
    # Calculer variant_id pour les lignes valides
    valid_df['variant_id'] = valid_df.apply(make_variant_id, axis=1)
    
    # Charger les annotations MyVariant depuis le fichier JSON
    with open('data/variant_data.json') as json_file:
        variant_data = json.load(json_file)
    
    for field in fields_list:
        valid_df = add_field(variant_data, valid_df, field)
    
    # Fusionner les données enrichies (valid_df) avec les lignes non traitées (invalid_df)
    # On utilise pd.concat pour ne pas perdre les lignes où les colonnes critiques étaient manquantes
    enriched_df = pd.concat([valid_df, invalid_df], sort=False)
    
    # Optionnel : vous pouvez trier le DataFrame final selon un index ou une colonne d'identifiant
    enriched_df.reset_index(drop=True, inplace=True)
    
    return enriched_df


# Fonction pour construire le modèle d'embedding
def build_embedding_model(num_categories, embedding_dim):
    input_cat = Input(shape=(1,), name='input_cat')
    emb = Embedding(input_dim=num_categories, output_dim=embedding_dim, name='embedding_layer')(input_cat)
    flat = Flatten(name='flatten')(emb)
    model = Model(inputs=input_cat, outputs=flat)
    return model

# Fonction pour préparer les embeddings à partir d'une colonne catégorielle
def create_embedding_features(df, id_col, ref_col, embedding_dim, min_count=0, rare_label=None):
    df = df.copy()
    # Comptage des occurrences
    counts = df[ref_col].value_counts()
    # Pour filtrer les catégories rares
    if min_count > 0:
        valid_categories = counts[counts >= min_count].index
        if rare_label is not None:
            df[ref_col] = df[ref_col].apply(lambda x: x if x in valid_categories else rare_label)
            # Recalculer après remplacement
            counts = df[ref_col].value_counts()
            valid_categories = counts.index
    # Créer le mapping catégorie -> indice
    unique_categories = df[ref_col].unique()
    cat_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    df['cat_idx'] = df[ref_col].map(cat_to_idx)
    num_categories = len(unique_categories)
    
    # Construire le modèle d'embedding
    emb_model = build_embedding_model(num_categories, embedding_dim)
    # Ici, on n'entraîne pas le modèle (les poids restent aléatoires) – dans un vrai pipeline, vous pouvez l'entraîner
    # Obtenir l'embedding pour chaque ligne
    embeddings = emb_model.predict(df['cat_idx'].values.reshape(-1, 1))
    
    # Construire un DataFrame à partir des embeddings
    emb_cols = [f"{ref_col}_emb_{i}" for i in range(embedding_dim)]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df[id_col])
    
    return df_emb