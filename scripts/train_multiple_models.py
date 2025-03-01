import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
import datetime
import sys
import os

# Ajouter le répertoire parent au chemin pour pouvoir importer save_model_version
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from save_model_version import save_model_version

warnings.filterwarnings('ignore')

def train_and_save_models():
    """Entraîne et sauvegarde plusieurs modèles différents"""
    
    # Chargement des données préparées
    print('Chargement des données...')
    df = pd.read_csv('../data/prepared_data.csv')
    
    # Création de la variable cible
    print('Préparation de la cible...')
    df['target'] = df['cl'].apply(lambda x: 1 if str(x) in ['1', '2', '3'] else 0)
    
    # Séparation des features et de la cible
    print('Préparation des features...')
    features = df.drop(['id', 'jour', 'cheval', 'target', 'cl'], axis=1, errors='ignore')
    target = df['target']
    
    # Identification des colonnes numériques
    print('Identification des types de colonnes...')
    numeric_columns = features.select_dtypes(include=['int64', 'float64']).columns
    features = features[numeric_columns]
    
    # Gestion des valeurs manquantes
    print('Gestion des valeurs manquantes...')
    imputer = SimpleImputer(strategy='mean')
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns
    )
    
    # Split des données en train et test
    print('Séparation des données en train/test...')
    X_train, X_test, y_train, y_test = train_test_split(
        features_imputed, 
        target, 
        test_size=0.2, 
        random_state=42
    )
    
    # Définition des modèles à entraîner
    models = [
        ('random_forest', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ('gradient_boosting', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )),
        ('adaboost', AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )),
        ('logistic_regression', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ]
    
    # Entraînement et sauvegarde de chaque modèle
    for name, model in models:
        print(f'\n--- Entraînement du modèle: {name} ---')
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = model.predict(X_test)
        
        # Nom unique pour ce modèle
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        model_name = f"{name}_{timestamp}"
        
        # Sauvegarde du modèle avec son code et ses performances
        save_model_version(model, imputer, model_name, X_test, y_test, y_pred)

if __name__ == '__main__':
    train_and_save_models() 