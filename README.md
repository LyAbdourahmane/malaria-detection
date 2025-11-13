# Détection du Paludisme par Deep Learning

Ce projet utilise un modèle de deep learning (CNN) pour détecter automatiquement la présence du paludisme dans des images de cellules sanguines. Le modèle est entraîné sur un dataset d'images de cellules parasitées et non-parasitées, et atteint une précision d'environ 94-95%.

## Description

Le paludisme est une maladie mortelle causée par des parasites transmis par les moustiques. La détection précoce est cruciale pour un traitement efficace. Ce projet automatise la détection du paludisme en analysant des images de cellules sanguines à l'aide de techniques de deep learning.

Le modèle utilise une architecture CNN (Convolutional Neural Network) pour classifier les cellules sanguines en deux catégories :
- **Parasitized** : Cellules infectées par le parasite du paludisme
- **Uninfected** : Cellules saines

##  Technologies Utilisées

- **Python** 3.x
- **TensorFlow/Keras** : Pour la construction et l'entraînement du modèle CNN
- **NumPy** : Pour les opérations numériques
- **Pandas** : Pour la manipulation des données
- **Matplotlib** : Pour la visualisation
- **Seaborn** : Pour les graphiques statistiques
- **scikit-learn** : Pour les métriques d'évaluation
- **ImageDataGenerator** : Pour l'augmentation de données

## Installation

### Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn requests
```

Ou créez un fichier `requirements.txt` avec le contenu suivant :

```
tensorflow>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
requests>=2.26.0
```

Puis installez les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

1. **Téléchargement des données** :
   Le dataset est automatiquement téléchargé depuis l'URL fournie dans le notebook lors de l'exécution.

2. **Exécution du notebook** :
   Ouvrez le notebook `malaria.ipynb` dans Jupyter Notebook ou Google Colab et exécutez les cellules dans l'ordre.

3. **Structure des données** :
   Les images doivent être organisées dans la structure suivante :
   ```
   cell_images/
   ├── train/
   │   ├── parasitized/
   │   └── uninfected/
   └── test/
       ├── parasitized/
       └── uninfected/
   ```

## Résultats

Le modèle atteint les performances suivantes :
- **Précision (Accuracy)** : ~94-95%
- **Précision (Precision)** : 0.97 (classe 0), 0.93 (classe 1)
- **Rappel (Recall)** : 0.92 (classe 0), 0.97 (classe 1)
- **Score F1** : 0.94 (classe 0), 0.95 (classe 1)

### Architecture du modèle

Le modèle CNN utilise l'architecture suivante :
- 3 couches de convolution (Conv2D) avec MaxPooling
- 1 couche de mise à plat (Flatten)
- 1 couche dense avec 128 neurones
- 1 couche de dropout (0.5)
- 1 couche de sortie avec fonction sigmoid

### Augmentation de données

Le modèle utilise l'augmentation de données pour améliorer la généralisation :
- Rotation (45°)
- Translation horizontale et verticale (10%)
- Zoom (10%)
- Retournement horizontal
- Cisaillement (10%)

## Structure du Projet

```
pallu/
├── malaria.ipynb          # Notebook principal avec le code complet
├── README.md              # Ce fichier
└── requirements.txt       # Dépendances Python
```

## Méthodologie

1. **Exploration des données** : Analyse des dimensions des images et distribution des classes
2. **Préprocessing** : Redimensionnement des images à 130x130 pixels et normalisation
3. **Augmentation de données** : Application de transformations pour augmenter la diversité du dataset
4. **Modélisation** : Construction d'un CNN avec TensorFlow/Keras
5. **Entraînement** : Utilisation d'EarlyStopping et ModelCheckpoint pour optimiser l'entraînement
6. **Évaluation** : Calcul des métriques de performance sur le jeu de test

## Améliorations Possibles

- Utilisation de modèles pré-entraînés (Transfer Learning) comme VGG16, ResNet, ou EfficientNet
- Optimisation des hyperparamètres
- Augmentation de la taille du dataset
- Exploration d'autres architectures de CNN
- Implémentation d'une interface web pour la prédiction

## Notes

- Le dataset contient 24 958 images d'entraînement et 2 600 images de test
- Les images sont redimensionnées à 130x130 pixels pour uniformiser les dimensions
- Le modèle utilise la fonction de perte `binary_crossentropy` et l'optimiseur `adam`
- EarlyStopping est utilisé pour éviter le surapprentissage

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est libre d'utilisation à des fins éducatives et de recherche.

## Remerciements

- Dataset fourni par [moncoachdata.com](https://moncoachdata.com)
- Communauté TensorFlow/Keras pour les ressources et la documentation

---

**Note** : Ce modèle est destiné à des fins éducatives et de recherche. Pour une utilisation en contexte médical réel, une validation clinique approfondie est nécessaire.