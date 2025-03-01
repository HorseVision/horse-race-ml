import os
import json
import shutil
import datetime
import joblib
from sklearn.metrics import classification_report

def save_model_version(model, imputer, model_name, X_test, y_test, y_pred):
    """
    Sauvegarde un modèle avec son code source et ses performances
    
    Args:
        model: Le modèle entraîné
        imputer: L'imputer utilisé
        model_name: Nom du modèle (ex: 'random_forest_20240612_1423')
        X_test: Données de test
        y_test: Cibles réelles
        y_pred: Prédictions du modèle
    """
    # Créer le dossier pour ce modèle
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Sauvegarde du modèle {model_name}...")
    
    # Sauvegarder le modèle et l'imputer
    joblib.dump(model, f"{model_dir}/model.joblib")
    joblib.dump(imputer, f"{model_dir}/imputer.joblib")
    
    # Sauvegarder les fichiers de code source
    for file in ['prepare_data.py', 'train_model.py', 'predict.py']:
        if os.path.exists(file):
            shutil.copy(file, f"{model_dir}/{file}")
            print(f"Code source copié: {file}")
    
    # Calculer et sauvegarder les métriques de performance
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Ajouter des métadonnées
    performance = {
        'model_name': model_name,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'accuracy': report['accuracy'],
        'precision_class1': report['1']['precision'],
        'recall_class1': report['1']['recall'],
        'f1_class1': report['1']['f1-score'],
        'hyperparameters': model.get_params(),
        'feature_importance': None
    }
    
    # Ajouter l'importance des features si disponible
    if hasattr(model, 'feature_importances_'):
        feature_names = X_test.columns
        importance = model.feature_importances_
        feature_importance = {name: float(imp) for name, imp in zip(feature_names, importance)}
        performance['feature_importance'] = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10])
    
    # Sauvegarder les performances
    with open(f"{model_dir}/performance.json", 'w') as f:
        json.dump(performance, f, indent=4)
    
    print(f"Modèle {model_name} sauvegardé avec son code et ses performances dans {model_dir}/")
    return model_dir 