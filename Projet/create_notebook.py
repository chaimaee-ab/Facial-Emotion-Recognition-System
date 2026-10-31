import json
import os

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_md("# Projet Emojify : Détection d'Émotions Faciales\n\nCe notebook couvre les étapes de 1 à 7 et les étapes 9 (Visualisation). L'objectif est de reconnaître les émotions et de les associer à des emojis.")

add_md("## Étape 1 : Prétraitement des données\nOn importe les librairies et on définit les fonctions pour charger et prétraiter les images.\n\nAssurez-vous que vos données sont dans les dossiers `data/train` et `data/test`. Les classes attendues sont `happy`, `sad`, `angry`, `fear`, `surprise`, `neutral`.")

add_code("""import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Config
IMG_SIZE = 48
DATA_DIR = './' # Dossier contenant train/ et test/
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
emotion_to_emoji = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐"
}

def load_and_preprocess_data(directory):
    X = []
    y = []
    if not os.path.exists(directory):
        print(f"Le dossier {directory} n'existe pas. Veuillez télécharger le dataset.")
        return np.array(X), np.array(y)
        
    for emotion_idx, emotion in enumerate(EMOTIONS):
        folder_path = os.path.join(directory, emotion)
        if not os.path.exists(folder_path):
            continue
            
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            # Lecture en niveaux de gris
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Redimensionnement (48x48)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(emotion_idx)
                
    # Normalisation entre 0 et 1
    X = np.array(X) / 255.0
    y = np.array(y)
    return X, y

print("Chargement des données d'entraînement...")
X_train, y_train = load_and_preprocess_data(TRAIN_DIR)
print(f"Images d'entraînement shape: {X_train.shape}")""")

add_md("## Étape 2 : Extraction des caractéristiques\n\nPour le Machine Learning Classique, nous allons utiliser les pixels aplatis (flatten) ou HOG. Pour le CNN, nous garderons les images 2D.")

add_code("""
# Préparation des données pour CNN (ajout de la dimension Channel)
if len(X_train) > 0:
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    y_train_cnn = to_categorical(y_train, num_classes=len(EMOTIONS))
else:
    X_train_cnn, y_train_cnn = np.array([]), np.array([])
    
# Flatten pour les modèles ML
def get_flatten_features(X):
    return X.reshape(X.shape[0], -1)

# HOG: Histogram of Oriented Gradients pour ML
def get_hog_features(X):
    hog_features = []
    for img in X:
        feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
        hog_features.append(feature)
    return np.array(hog_features)

if len(X_train) > 0:
    X_train_flat = get_flatten_features(X_train)
    X_train_hog = get_hog_features(X_train)
""")

add_md("## Étape 3 & 4 : Définition et Entraînement des Modèles\n### 3.1 Deep Learning : CNN")

add_code("""
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(EMOTIONS), activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

cnn_model = create_cnn_model()
cnn_model.summary()

# (Optionnel) Décommentez pour entraîner si vous avez des données
# history = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=15, validation_split=0.2, batch_size=64)
# cnn_model.save('emojify_cnn_model.h5')
""")

add_md("### 3.2 Machine Learning Classique : KNN et Random Forest")

add_code("""
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Si les données existent, on entraîne:
# print("Entraînement KNN...")
# knn_model.fit(X_train_hog, y_train)

# print("Entraînement Random Forest...")
# rf_model.fit(X_train_hog, y_train)
""")

add_md("## Étape 5 & 9 : Évaluation et Visualisation\nVisualisons les courbes d'apprentissage et la matrice de confusion (pour du code factice si pas de dataset, ou réel si entraîné).")

add_code("""
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Loss')
    plt.legend()
    plt.show()

# plot_history(history)

def evaluate_model(y_true, y_pred, title):
    print(f"--- Evaluation de : {title} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=EMOTIONS))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()

# y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
# evaluate_model(y_test, y_pred_cnn, "CNN Model")
""")

add_md("## Étape 6 & 7 : Dictionnaire d'Emojis et Fonction de Prédiction")

add_code("""
def predict_emotion(img_path, model, model_type='cnn'):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image non trouvée")
        return None
        
    # Prétraitement
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    
    if model_type == 'cnn':
        img_prep = np.expand_dims(img_normalized, axis=(0, -1))
        preds = model.predict(img_prep)
        emotion_idx = np.argmax(preds)
    elif model_type == 'ml_hog':
        feature = get_hog_features([img_normalized])
        emotion_idx = model.predict(feature)[0]
    else: # ml_flat
        feature = get_flatten_features(np.array([img_normalized]))
        emotion_idx = model.predict(feature)[0]
        
    emotion_name = EMOTIONS[emotion_idx]
    emoji = emotion_to_emoji.get(emotion_name, "")
    
    # Affichage
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title(f"{emotion_name} {emoji}")
    plt.axis('off')
    plt.show()
    
    return emotion_name, emoji

# predict_emotion('chemin/vers/image.jpg', cnn_model, 'cnn')
""")

output_file = "Emojify.ipynb"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook {output_file} généré avec succès!")
