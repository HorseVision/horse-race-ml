import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import warnings
warnings.filterwarnings('ignore')

# Connexion à la base de données
engine = create_engine('mysql://root:root@127.0.0.1:3306/aspiturf')

# Requête SQL pour extraire les données pertinentes
query = """
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
    WHERE c.jour >= DATE_SUB(NOW(), INTERVAL 2 YEAR)
    ORDER BY c.jour DESC
"""

# Chargement des données
print('Chargement des données...')
df = pd.read_sql(query, engine)

# Nettoyage des données
print('Nettoyage des données...')

# Conversion des dates
df['jour'] = pd.to_datetime(df['jour'])

# Fonction pour encoder les top catégories
def get_top_categories(df, column, top_n=100):
    value_counts = df[column].value_counts()
    return list(value_counts.head(top_n).index)

def encode_with_categories(df, column, categories):
    df[column] = df[column].apply(lambda x: x if x in categories else 'AUTRE')
    dummies = pd.get_dummies(df[column], prefix=column)
    # Ajout de la colonne 'AUTRE' si elle n'existe pas
    autre_col = f'{column}_AUTRE'
    if autre_col not in dummies.columns:
        dummies[autre_col] = 0
    return dummies

# Obtention des catégories
print('Extraction des catégories...')
categories = {
    'hippo': get_top_categories(df, 'hippo', 50),
    'jockey': get_top_categories(df, 'jockey', 100),
    'entraineur': get_top_categories(df, 'entraineur', 100)
}

# Variables avec peu de catégories - encodage direct
print('Encodage des variables catégorielles...')
simple_categorical = ['typec', 'sexe', 'meteo', 'directionVent']
df_encoded = pd.get_dummies(df, columns=simple_categorical)

# Variables avec beaucoup de catégories
encoded_parts = [df_encoded]
for col, cats in categories.items():
    encoded = encode_with_categories(df, col, cats)
    encoded_parts.append(encoded)

df_encoded = pd.concat(encoded_parts, axis=1)

# Suppression des colonnes originales
df_encoded = df_encoded.drop(['hippo', 'jockey', 'entraineur'], axis=1)

# Sauvegarde des catégories et de la structure des colonnes
print('Sauvegarde des catégories et de la structure...')
feature_config = {
    'categories': categories,
    'columns': list(df_encoded.columns)
}
joblib.dump(feature_config, 'models/feature_config.joblib')

# Sauvegarde des données préparées
print('Sauvegarde des données...')
df_encoded.to_csv('data/prepared_data.csv', index=False)
print('Données préparées et sauvegardées dans data/prepared_data.csv')
print('Configuration des features sauvegardée dans models/feature_config.joblib')

# Création de la variable cible (1 si le cheval finit dans les 3 premiers, 0 sinon)
# Utiliser 'cl' (classement à l'arrivée) au lieu de 'coteprob'
df['target'] = df['cl'].apply(lambda x: 1 if x in ['1', '2', '3'] else 0)

# Séparation des features et de la cible
print('Préparation des features...')
features = df.drop(['id', 'jour', 'cheval', 'target', 'cl'], axis=1, errors='ignore')
target = df['target']

def align_features(df, expected_columns):
    """Aligne les colonnes du DataFrame avec celles attendues"""
    # Filtrer les colonnes à exclure (mais garder coteprob)
    columns_to_exclude = ['id', 'jour', 'cheval']
    filtered_columns = [col for col in expected_columns if col not in columns_to_exclude] 