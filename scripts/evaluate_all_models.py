import os
import sys
import glob

# Ajouter le répertoire parent au chemin pour pouvoir importer predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import evaluate_model

def evaluate_all_models(sample_size=1000):
    """Évalue tous les modèles sauvegardés sur le même échantillon de données"""
    # Obtenir le chemin absolu du répertoire racine du projet
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, 'models')
    
    # Vérifier que le dossier models existe
    if not os.path.exists(models_dir):
        print(f"ERREUR: Le dossier models n'existe pas à l'emplacement: {models_dir}")
        return
    
    # Rechercher tous les dossiers de modèles
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_dirs:
        print("Aucun modèle trouvé dans le dossier models/")
        return
    
    print(f"Évaluation de {len(model_dirs)} modèles sur un échantillon de {sample_size} courses...")
    
    # Évaluer chaque modèle
    for model_dir in model_dirs:
        model_path = os.path.join(models_dir, model_dir, 'model.joblib')
        imputer_path = os.path.join(models_dir, model_dir, 'imputer.joblib')
        feature_config_path = os.path.join(models_dir, model_dir, 'feature_config.joblib')
        
        # Si feature_config n'existe pas dans le dossier du modèle, utiliser celui du dossier models
        if not os.path.exists(feature_config_path):
            feature_config_path = os.path.join(models_dir, 'feature_config.joblib')
        
        if os.path.exists(model_path) and os.path.exists(imputer_path):
            print(f"\n--- Évaluation du modèle: {model_dir} ---")
            try:
                evaluate_model(
                    sample_size=sample_size,
                    model_path=model_path,
                    imputer_path=imputer_path,
                    feature_config_path=feature_config_path,
                    save_results=True,
                    output_dir=os.path.join(models_dir, model_dir)
                )
            except Exception as e:
                print(f"ERREUR lors de l'évaluation du modèle {model_dir}: {str(e)}")
                # Afficher plus de détails sur l'erreur
                import traceback
                traceback.print_exc()
        else:
            print(f"Fichiers manquants pour le modèle {model_dir}")
    
    print("\nÉvaluation terminée. Utilisez scripts/compare_evaluations.py pour comparer les résultats.")

if __name__ == '__main__':
    # Utiliser un argument en ligne de commande pour spécifier la taille de l'échantillon
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print("Argument invalide. Utilisation de la valeur par défaut.")
            sample_size = 1000
    else:
        sample_size = 1000
    
    evaluate_all_models(sample_size) 