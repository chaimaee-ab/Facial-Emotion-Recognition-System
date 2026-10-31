# Rapport Emojify

## 1. Introduction
Ce projet, nommé **Emojify**, a pour objectif la classification des émotions faciales humaines (happy, sad, angry, fear, surprise, neutral) à partir d'images et leur association à des emojis correspondants. Ce domaine de la vision par ordinateur joue un rôle fondamental dans l'analyse comportementale et l'interaction homme-machine.

## 2. Description du Dataset
Nous utilisons un dataset d'images faciales réparties en 6 classes d'émotions.
- **Format** : Images en niveaux de gris
- **Résolution** : 48x48 pixels
- **Problème** : Déséquilibre de classes (ex: plus de "happy" que de "surprise").

*Note : Les données doivent être organisées dans `data/train` et `data/test`.*

## 3. Prétraitement et Extraction de Caractéristiques
1. **Redimensionnement et Normalisation** : Les images ont été converties en une taille fixe de 48x48, puis les valeurs de pixels mises à l'échelle (de 0-255 à 0-1).
2. **Extraction CNN** : Aucune extraction classique n'est nécessaire. On garde une forme spatiale 2D.
3. **Extraction ML (HOG/Flatten)** : Pour alimenter les modèles SVM ou KNN, nous avons extrait les caractéristiques structurelles avec l'algorithme *Histogram of Oriented Gradients* et aplatissé les images.

## 4. Modèles Utilisés
- **Deep Learning (CNN)** : Architecture personnalisée avec des couches successives de `Conv2D`, `MaxPooling2D`, avec `Dropout` pour limiter l'overfitting.
- **Machine Learning (KNN/Random Forest)** : Modèles de classification standards entraînés sur les représentations vectorielles (flatten/HOG).

## 5. Résultats et Comparaison
*(Mettre à jour cette section après l'entraînement sur votre dataset)*
Généralement, le modèle CNN obtient de bien meilleures performances qu'un algorithme KNN traditionnel sur les problématiques d'image, en partie parce qu'il comprend la hiérarchie spatiale, tandis qu'un KNN sera basé sur des distances souvent biaisées dans les hautes dimensions.

## 6. Conclusion
Ce projet permet de couvrir toutes les phases primordiales d'un projet Machine Learning appliqué à l'image. Avec l'ajout de l'application *Streamlit* et du système webcam en temps réel via *OpenCV*, l'expérience devient interactive et immédiatement exploitable pour un projet universitaire ou de portfolio.
