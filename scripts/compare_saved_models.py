import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def compare_saved_models():
    """Compare tous les modèles sauvegardés"""
    models_data = []
    
    # Parcourir tous les dossiers de modèles
    for model_dir in os.listdir('models'):
        perf_file = f"models/{model_dir}/performance.json"
        if os.path.exists(perf_file):
            with open(perf_file, 'r') as f:
                performance = json.load(f)
                models_data.append(performance)
    
    # Créer un DataFrame avec les performances
    df = pd.DataFrame(models_data)
    
    # Trier par F1-score
    df = df.sort_values('f1_class1', ascending=False)
    
    # Afficher le tableau comparatif
    print("Comparaison des modèles:")
    print(df[['model_name', 'accuracy', 'precision_class1', 'recall_class1', 'f1_class1', 'timestamp']])
    
    # Créer un graphique
    plt.figure(figsize=(12, 6))
    plt.barh(df['model_name'], df['f1_class1'], color='skyblue')
    plt.xlabel('F1-Score (classe positive)')
    plt.ylabel('Modèle')
    plt.title('Comparaison des performances des modèles')
    plt.tight_layout()
    plt.savefig('models/comparison.png')
    plt.show()
    
    return df

if __name__ == "__main__":
    compare_saved_models()
