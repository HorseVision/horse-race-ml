# Prédiction de Courses Hippiques

Ce projet utilise le machine learning pour prédire les résultats des courses hippiques en se basant sur les données historiques.

## Structure du Projet

```
.
├── data/               # Dossier contenant les données préparées
├── models/            # Dossier contenant les modèles entraînés
├── prepare_data.py    # Script de préparation des données
├── train_model.py     # Script d'entraînement du modèle
├── predict.py         # Script de prédiction
└── requirements.txt   # Dépendances Python
```

## Installation

1. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Préparer les données :
```bash
python prepare_data.py
```

2. Entraîner le modèle :
```bash
python train_model.py
```

3. Faire des prédictions :
```bash
python predict.py
```

Pour prédire les courses d'une date spécifique, modifier le script `predict.py` :
```python
from datetime import date
predict_races(date(2024, 1, 1))  # Pour le 1er janvier 2024
```

## Features Utilisées

- Données du cheval (âge, sexe, performances passées)
- Données du jockey (performances, statistiques)
- Données de l'entraîneur (performances, statistiques)
- Conditions de course (distance, type, météo)
- Scores ELO (cheval, jockey, entraîneur)
- Statistiques historiques
- Cotes et probabilités

## Modèle

Le modèle utilise un Random Forest Classifier pour prédire si un cheval finira dans les 3 premiers de sa course. Les hyperparamètres ont été choisis pour équilibrer performance et généralisation.
