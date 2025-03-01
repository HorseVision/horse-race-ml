import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import warnings
import datetime
import os
import shutil
from save_model_version import save_model_version

# Chargement des données préparées
print('Chargement des données...')
df = pd.read_csv('data/prepared_data.csv')

# Création de la variable cible (1 si le cheval finit dans les 3 premiers, 0 sinon)
df['target'] = (df['coteprob'] <= 3).astype(int)

# Séparation des features et de la cible
print('Préparation des features...')
features = df.drop(['id', 'jour', 'cheval', 'target', 'coteprob'], axis=1, errors='ignore')
target = df['target']

# Identification des colonnes numériques et non numériques
print('Identification des types de colonnes...')
numeric_columns = features.select_dtypes(include=['int64', 'float64']).columns
non_numeric_columns = features.select_dtypes(exclude=['int64', 'float64']).columns

print(f'Colonnes numériques: {len(numeric_columns)}')
print(f'Colonnes non numériques: {len(non_numeric_columns)}')

if len(non_numeric_columns) > 0:
    print(f'Exemples de colonnes non numériques: {list(non_numeric_columns)[:5]}')
    print('Suppression des colonnes non numériques...')
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

# Entraînement du modèle
print('Entraînement du modèle...')
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Évaluation du modèle
print('\nÉvaluation du modèle:')
y_pred = model.predict(X_test)
print('\nRapport de classification:')
print(classification_report(y_test, y_pred))

# Affichage des features les plus importantes
feature_importance = pd.DataFrame({
    'feature': features.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print('\nFeatures les plus importantes:')
print(feature_importance.head(10))

# Définir un nom unique pour ce modèle
model_name = f"random_forest_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

# Sauvegarder le modèle avec son code et ses performances
model_dir = save_model_version(model, imputer, model_name, X_test, y_test, y_pred)

# Copier le feature_config.joblib dans le dossier du modèle
feature_config_src = 'models/feature_config.joblib'
feature_config_dst = os.path.join(model_dir, 'feature_config.joblib')

if os.path.exists(feature_config_src):
    print(f'Copie du feature_config.joblib dans le dossier du modèle: {model_dir}')
    shutil.copy2(feature_config_src, feature_config_dst)
else:
    print(f'ATTENTION: feature_config.joblib non trouvé à l\'emplacement: {feature_config_src}')

print(f'Modèle sauvegardé dans: {model_dir}') 