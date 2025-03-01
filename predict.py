import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
import warnings
warnings.filterwarnings('ignore')

def encode_with_saved_categories(df, column, categories):
    """Encode une colonne en utilisant les catégories sauvegardées"""
    df[column] = df[column].apply(lambda x: x if x in categories else 'AUTRE')
    dummies = pd.get_dummies(df[column], prefix=column)
    # Ajout de la colonne 'AUTRE' si elle n'existe pas
    autre_col = f'{column}_AUTRE'
    if autre_col not in dummies.columns:
        dummies[autre_col] = 0
    return dummies

def align_features(df, expected_columns):
    """Aligne les colonnes du DataFrame avec celles attendues"""
    # Filtrer les colonnes à exclure (mais garder coteprob)
    columns_to_exclude = ['id', 'jour', 'cheval', 'coteprob', 'cl']
    filtered_columns = [col for col in expected_columns if col not in columns_to_exclude]
    
    # Créer un nouveau DataFrame avec uniquement les colonnes attendues
    aligned_df = pd.DataFrame(index=df.index)
    
    # Ajouter chaque colonne attendue
    for col in filtered_columns:
        if col in df.columns:
            aligned_df[col] = df[col]
        else:
            aligned_df[col] = 0
    
    # Vérifier que toutes les colonnes sont présentes
    missing_cols = set(filtered_columns) - set(aligned_df.columns)
    if missing_cols:
        print(f"ATTENTION: {len(missing_cols)} colonnes manquantes: {list(missing_cols)[:5]}...")
    
    # Vérifier qu'il n'y a pas de colonnes en trop
    extra_cols = set(aligned_df.columns) - set(filtered_columns)
    if extra_cols:
        print(f"ATTENTION: {len(extra_cols)} colonnes en trop: {list(extra_cols)[:5]}...")
        aligned_df = aligned_df.drop(columns=list(extra_cols))
    
    return aligned_df

def find_feature_config(model_path):
    """Cherche le feature_config.joblib dans le dossier du modèle ou dans le dossier models"""
    # Chercher d'abord dans le dossier du modèle
    model_dir = os.path.dirname(model_path)
    feature_config_path = os.path.join(model_dir, 'feature_config.joblib')
    
    if os.path.exists(feature_config_path):
        return feature_config_path
    
    # Sinon, chercher dans le dossier models
    feature_config_path = 'models/feature_config.joblib'
    if os.path.exists(feature_config_path):
        return feature_config_path
    
    # Si toujours pas trouvé, chercher dans le dossier parent de models
    parent_dir = os.path.dirname(os.path.dirname(model_path))
    feature_config_path = os.path.join(parent_dir, 'feature_config.joblib')
    if os.path.exists(feature_config_path):
        return feature_config_path
    
    return None

def evaluate_model(sample_size=1000, model_path=None, imputer_path=None, feature_config_path=None, save_results=True, output_dir=None):
    """Évalue le modèle sur un échantillon de courses historiques"""
    # Déterminer les chemins des fichiers
    if model_path is None:
        model_path = 'models/horse_race_model.joblib'
    
    if imputer_path is None:
        imputer_path = 'models/imputer.joblib'
    
    if feature_config_path is None:
        feature_config_path = find_feature_config(model_path)
        if feature_config_path is None:
            print("ERREUR: Impossible de trouver le fichier feature_config.joblib")
            return
    
    # Déterminer le dossier de sortie
    if output_dir is None:
        # Si le modèle est dans un sous-dossier de models, utiliser ce dossier
        model_dir = os.path.dirname(model_path)
        if os.path.basename(os.path.dirname(model_dir)) == 'models':
            output_dir = model_dir
        else:
            output_dir = 'data'
    
    # Extraire le nom du modèle à partir du chemin
    model_name = os.path.basename(os.path.dirname(model_path)) if os.path.dirname(model_path) != 'models' else 'model'
    
    print(f"Utilisation du modèle: {model_path}")
    print(f"Utilisation de l'imputer: {imputer_path}")
    print(f"Utilisation du feature_config: {feature_config_path}")
    print(f"Dossier de sortie: {output_dir}")
    
    # Connexion à la base de données
    engine = create_engine('mysql://root:root@127.0.0.1:3306/aspiturf')
    
    # Requête SQL pour obtenir un échantillon aléatoire de courses
    query = f"""
        SELECT 
            c.id,
            c.jour,
            c.hippo,
            c.dist,
            c.typec,
            c.partant,
            c.cheval,
            c.age,
            c.sexe,
            c.jockey,
            c.entraineur,
            c.coteprob,
            c.cotedirect,
            c.recence,
            c.eloCheval,
            c.eloJockey,
            c.eloEntraineur,
            c.pourcVictCheval,
            c.pourcVictJock,
            c.pourcVictEnt,
            c.coursescheval,
            c.victoirescheval,
            c.coursesjockey,
            c.victoiresjockey,
            c.coursesentraineur,
            c.victoiresentraineur,
            ca.meteo,
            ca.temperature,
            ca.forceVent,
            ca.directionVent,
            h.corde,
            h.pays,
            c.cl
        FROM cachedate c
        LEFT JOIN caractrap ca ON c.comp = ca.comp AND c.jour = ca.jour
        LEFT JOIN hippo h ON c.hippo = h.hippo
        ORDER BY RAND()
        LIMIT {sample_size}
    """
    
    print(f'Chargement d\'un échantillon de {sample_size} courses...')
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print('Aucune course trouvée')
        return
    
    # Préparation des données
    print('Préparation des données...')
    
    # Chargement de la configuration des features
    feature_config = joblib.load(feature_config_path)
    categories = feature_config['categories']
    expected_columns = feature_config['columns']
    
    print(f"Colonnes attendues: {len(expected_columns)}")
    
    # Création d'un DataFrame pour stocker les features encodées
    df_encoded = df.copy()
    
    # Variables avec peu de catégories - encodage direct
    # Au lieu d'utiliser pd.get_dummies directement, nous allons créer manuellement les colonnes
    # pour s'assurer qu'elles correspondent exactement aux colonnes attendues
    simple_categorical = ['typec', 'sexe', 'meteo', 'directionVent']
    
    # Récupérer les préfixes des colonnes attendues pour chaque variable catégorielle
    categorical_prefixes = {}
    for col in simple_categorical:
        prefix = f"{col}_"
        categorical_prefixes[col] = [c for c in expected_columns if c.startswith(prefix)]
    
    # Encoder manuellement chaque variable catégorielle
    for col in simple_categorical:
        if col in df.columns:
            # Supprimer la colonne originale
            values = df[col].copy()
            df_encoded = df_encoded.drop(columns=[col])
            
            # Créer les colonnes encodées
            for prefix_col in categorical_prefixes[col]:
                # Extraire la valeur de la catégorie du nom de la colonne
                category = prefix_col[len(col)+1:]
                # Créer la colonne encodée (1 si la valeur correspond, 0 sinon)
                df_encoded[prefix_col] = (values == category).astype(int)
    
    print(f"Après encodage simple: {df_encoded.shape[1]} colonnes")
    
    # Variables avec beaucoup de catégories
    for col, cats in categories.items():
        if col in df.columns:
            # Supprimer la colonne originale
            values = df[col].copy()
            if col in df_encoded.columns:
                df_encoded = df_encoded.drop(columns=[col])
            
            # Créer les colonnes encodées
            for cat in cats:
                col_name = f"{col}_{cat}"
                df_encoded[col_name] = (values == cat).astype(int)
            
            # Ajouter la colonne AUTRE
            autre_col = f'{col}_AUTRE'
            df_encoded[autre_col] = (~values.isin(cats)).astype(int)
    
    print(f"Après encodage complexe: {df_encoded.shape[1]} colonnes")
    
    # Suppression des colonnes catégorielles originales
    columns_to_drop = ['hippo', 'jockey', 'entraineur', 'id', 'jour', 'cheval']
    columns_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]
    if columns_to_drop:
        df_encoded = df_encoded.drop(columns=columns_to_drop)
        print(f"Après suppression des colonnes: {df_encoded.shape[1]} colonnes")
    
    # Supprimer coteprob des features car c'est notre variable cible
    if 'coteprob' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['coteprob'])
        print("Suppression de coteprob des features")
    
    # Conserver la colonne cl pour l'évaluation mais ne pas l'inclure dans les features
    cl_values = None
    if 'cl' in df_encoded.columns:
        cl_values = df_encoded['cl'].copy()
        df_encoded = df_encoded.drop(columns=['cl'])
        print("Conservation de cl pour l'évaluation, mais suppression des features")
    
    # Chargement du modèle et de l'imputer
    print(f'Chargement du modèle depuis {model_path}...')
    model = joblib.load(model_path)
    
    print(f'Chargement de l\'imputer depuis {imputer_path}...')
    imputer = joblib.load(imputer_path)
    
    # Vérifier que les colonnes correspondent avant d'appliquer l'imputer
    imputer_feature_names = imputer.feature_names_in_
    print(f"Nombre de colonnes dans l'imputer: {len(imputer_feature_names)}")
    
    # Alignement des colonnes avec celles attendues
    print(f"Colonnes avant alignement: {sorted(df_encoded.columns)[:5]}...")
    features = align_features(df_encoded, imputer_feature_names)
    print(f"Après alignement: {features.shape[1]} colonnes")
    print(f"Colonnes après alignement: {sorted(features.columns)[:5]}...")
    
    # Création de la variable cible (top 3 basé sur le classement réel)
    if cl_values is not None:
        y_true = cl_values.apply(lambda x: 1 if str(x) in ['1', '2', '3'] else 0)
    else:
        # Fallback si cl n'est pas disponible
        print("ATTENTION: Colonne 'cl' non disponible, utilisation de coteprob comme approximation")
        y_true = (df['coteprob'] <= 3).astype(int)
    
    # Application de l'imputer
    try:
        features_imputed = pd.DataFrame(
            imputer.transform(features),
            columns=features.columns
        )
        print("Imputation réussie!")
    except Exception as e:
        print(f"ERREUR lors de l'imputation: {str(e)}")
        # Afficher plus de détails sur l'erreur
        import traceback
        traceback.print_exc()
        return
    
    # Prédiction
    print('Prédiction des résultats...')
    predictions = model.predict_proba(features_imputed)
    y_pred = (predictions[:, 1] >= 0.5).astype(int)
    
    # Évaluation des performances
    print('\nRapport de classification:')
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))
    
    # Calcul de la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Analyse des résultats par seuil de probabilité
    print('\nAnalyse par seuil de probabilité:')
    threshold_analysis = {}
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        y_pred_threshold = (predictions[:, 1] >= threshold).astype(int)
        n_predictions = y_pred_threshold.sum()
        threshold_analysis[str(threshold)] = {
            'n_predictions': int(n_predictions)
        }
        
        if n_predictions > 0:
            precision = (y_pred_threshold & y_true).sum() / n_predictions
            threshold_analysis[str(threshold)]['precision'] = float(precision)
            print(f'Seuil {threshold:.1f}: {n_predictions} prédictions, '
                  f'Précision: {precision:.2%}')
    
    # Analyse par type de course
    print('\nAnalyse par type de course:')
    type_analysis = {}
    for type_course in df['typec'].unique():
        mask = df['typec'] == type_course
        if mask.sum() > 0:
            y_true_type = y_true[mask]
            y_pred_type = y_pred[mask]
            n_courses = int(mask.sum())
            precision = float((y_pred_type & y_true_type).sum() / y_pred_type.sum()) if y_pred_type.sum() > 0 else 0
            
            type_analysis[type_course] = {
                'n_courses': n_courses,
                'precision': precision
            }
            
            print(f'{type_course}: {n_courses} courses, '
                  f'Précision: {precision:.2%}')
    
    # Création d'un dictionnaire avec les résultats d'évaluation
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    evaluation_results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'sample_size': sample_size,
        'accuracy': float(report['accuracy']),
        'precision_class1': float(report['1']['precision']),
        'recall_class1': float(report['1']['recall']),
        'f1_class1': float(report['1']['f1-score']),
        'threshold_analysis': threshold_analysis,
        'type_analysis': type_analysis,
        'confusion_matrix': {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    }
    
    # Sauvegarde des résultats détaillés
    if save_results:
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Générer un nom de fichier avec timestamp
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M')
        results_csv_path = os.path.join(output_dir, f'evaluation_{model_name}_{timestamp_file}.csv')
        results_json_path = os.path.join(output_dir, f'evaluation_{model_name}_{timestamp_file}.json')
        
        # Sauvegarde des résultats détaillés en CSV
        results_df = pd.DataFrame({
            'date': df['jour'],
            'hippo': df['hippo'],
            'course': df['partant'],
            'cheval': df['cheval'],
            'jockey': df['jockey'],
            'cote': df['cotedirect'],
            'probabilite_top3': predictions[:, 1],
            'prediction': y_pred,
            'realite': y_true
        })
        results_df.to_csv(results_csv_path, index=False)
        
        # Sauvegarde des métriques en JSON
        with open(results_json_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        print(f'\nRésultats détaillés sauvegardés dans {results_csv_path}')
        print(f'Métriques d\'évaluation sauvegardées dans {results_json_path}')
    
    return evaluation_results

def predict_races(date=None):
    """Prédit les résultats pour une date donnée"""
    print('ATTENTION: Cette fonction nécessite des données du jour dans la base de données')
    print('Utilisez evaluate_model() pour tester le modèle sur des données historiques')

if __name__ == '__main__':
    import sys
    
    # Vérifier si des arguments sont fournis
    if len(sys.argv) > 1:
        # Si le premier argument est un chemin vers un modèle
        model_path = sys.argv[1]
        if os.path.exists(model_path):
            model_dir = os.path.dirname(model_path)
            imputer_path = os.path.join(model_dir, 'imputer.joblib')
            
            # Vérifier si les fichiers nécessaires existent
            if not os.path.exists(imputer_path):
                print(f"ERREUR: Fichier imputer non trouvé: {imputer_path}")
                sys.exit(1)
            
            # Chercher le feature_config dans le dossier du modèle ou ailleurs
            feature_config_path = find_feature_config(model_path)
            if feature_config_path is None:
                print(f"ERREUR: Fichier feature_config non trouvé")
                sys.exit(1)
            
            # Déterminer la taille de l'échantillon
            sample_size = 1000
            if len(sys.argv) > 2:
                try:
                    sample_size = int(sys.argv[2])
                except ValueError:
                    print("Argument invalide pour la taille de l'échantillon. Utilisation de la valeur par défaut.")
            
            # Évaluer le modèle spécifié
            evaluate_model(
                sample_size=sample_size,
                model_path=model_path,
                imputer_path=imputer_path,
                feature_config_path=feature_config_path,
                output_dir=model_dir
            )
        else:
            print(f"ERREUR: Modèle non trouvé: {model_path}")
    else:
        # Évaluation sur 1000 courses avec le modèle par défaut
        evaluate_model(1000)