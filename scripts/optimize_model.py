import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
import datetime
import sys
import os

# Ajouter le répertoire parent au chemin pour pouvoir importer save_model_version
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from save_model_version import save_model_version

warnings.filterwarnings('ignore')

def optimize_random_forest():
    """Optimise les hyperparamètres d'un RandomForest et sauvegarde le meilleur modèle"""
    
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
    
    # Définition de la grille de paramètres
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Création du modèle de base
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Recherche par grille avec validation croisée
    print('Optimisation des hyperparamètres...')
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Entraînement sur toutes les combinaisons
    grid_search.fit(X_train, y_train)
    
    # Affichage des meilleurs paramètres
    print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score F1: {grid_search.best_score_:.4f}")
    
    # Récupération du meilleur modèle
    best_model = grid_search.best_estimator_
    
    # Évaluation sur les données de test
    y_pred = best_model.predict(X_test)
    
    # Nom unique pour ce modèle
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    model_name = f"random_forest_optimized_{timestamp}"
    
    # Sauvegarde du modèle avec son code et ses performances
    save_model_version(best_model, imputer, model_name, X_test, y_test, y_pred)
    
    return grid_search.best_params_

if __name__ == '__main__':
    optimize_random_forest() 