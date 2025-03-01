import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def compare_evaluations():
    """Compare les résultats d'évaluation de différents modèles"""
    # Obtenir le chemin absolu du répertoire racine du projet
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    models_dir = os.path.join(root_dir, 'models')
    
    # Rechercher tous les fichiers JSON d'évaluation dans le dossier data et dans les sous-dossiers de models
    evaluation_files = []
    
    # Rechercher dans le dossier data
    if os.path.exists(data_dir):
        data_files = glob.glob(os.path.join(data_dir, 'evaluation_*.json'))
        evaluation_files.extend(data_files)
    
    # Rechercher dans les sous-dossiers de models
    if os.path.exists(models_dir):
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            if os.path.isdir(model_path):
                model_files = glob.glob(os.path.join(model_path, 'evaluation_*.json'))
                evaluation_files.extend(model_files)
    
    if not evaluation_files:
        print("Aucun fichier d'évaluation trouvé")
        return
    
    print(f"Fichiers d'évaluation trouvés: {len(evaluation_files)}")
    
    # Charger les données d'évaluation
    evaluations = []
    for file in evaluation_files:
        try:
            with open(file, 'r') as f:
                evaluation = json.load(f)
                evaluations.append(evaluation)
                print(f"Chargé: {file}")
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {str(e)}")
    
    if not evaluations:
        print("Aucune évaluation valide trouvée")
        return
    
    # Créer un DataFrame avec les métriques principales
    df = pd.DataFrame([{
        'model_name': eval.get('model_name', 'inconnu'),
        'timestamp': eval.get('timestamp', ''),
        'sample_size': eval.get('sample_size', 0),
        'accuracy': eval.get('accuracy', 0),
        'precision': eval.get('precision_class1', 0),
        'recall': eval.get('recall_class1', 0),
        'f1_score': eval.get('f1_class1', 0),
        'true_positives': eval.get('confusion_matrix', {}).get('true_positives', 0),
        'false_positives': eval.get('confusion_matrix', {}).get('false_positives', 0),
        'true_negatives': eval.get('confusion_matrix', {}).get('true_negatives', 0),
        'false_negatives': eval.get('confusion_matrix', {}).get('false_negatives', 0)
    } for eval in evaluations])
    
    # Trier par F1-score
    df = df.sort_values('f1_score', ascending=False)
    
    # Afficher le tableau comparatif
    print("\nComparaison des évaluations de modèles:")
    print(df[['model_name', 'timestamp', 'accuracy', 'precision', 'recall', 'f1_score']])
    
    # Créer un graphique de comparaison des métriques
    plt.figure(figsize=(12, 8))
    
    # Préparer les données pour le graphique
    plot_data = df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score']]
    plot_data = plot_data.set_index('model_name')
    
    # Créer le graphique
    ax = plot_data.plot(kind='bar', figsize=(12, 6))
    plt.title('Comparaison des métriques d\'évaluation')
    plt.ylabel('Score')
    plt.xlabel('Modèle')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Métrique')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'evaluation_comparison.png'))
    
    # Créer un graphique de l'analyse par seuil
    plt.figure(figsize=(12, 6))
    
    # Extraire les données d'analyse par seuil pour chaque modèle
    threshold_data = {}
    for eval in evaluations:
        model_name = eval.get('model_name', 'inconnu')
        thresholds = []
        precisions = []
        
        for threshold, data in eval.get('threshold_analysis', {}).items():
            thresholds.append(float(threshold))
            precisions.append(data.get('precision', 0))
        
        if thresholds and precisions:
            threshold_data[model_name] = {
                'thresholds': thresholds,
                'precisions': precisions
            }
    
    # Tracer les courbes de précision par seuil
    for model_name, data in threshold_data.items():
        plt.plot(data['thresholds'], data['precisions'], marker='o', label=model_name)
    
    plt.title('Précision par seuil de probabilité')
    plt.xlabel('Seuil')
    plt.ylabel('Précision')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'threshold_analysis.png'))
    
    # Créer un graphique de l'analyse par type de course
    plt.figure(figsize=(14, 8))
    
    # Extraire les données d'analyse par type de course
    type_data = {}
    all_types = set()
    
    for eval in evaluations:
        model_name = eval.get('model_name', 'inconnu')
        type_data[model_name] = {}
        
        for type_course, data in eval.get('type_analysis', {}).items():
            type_data[model_name][type_course] = data.get('precision', 0)
            all_types.add(type_course)
    
    # Créer un DataFrame pour le graphique
    type_df = pd.DataFrame(index=list(all_types))
    
    for model_name, data in type_data.items():
        type_df[model_name] = pd.Series(data)
    
    # Tracer le graphique
    ax = type_df.plot(kind='bar', figsize=(14, 8))
    plt.title('Précision par type de course')
    plt.xlabel('Type de course')
    plt.ylabel('Précision')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Modèle')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'type_analysis.png'))
    
    print("\nGraphiques sauvegardés dans le dossier data/")
    return df

if __name__ == '__main__':
    compare_evaluations() 