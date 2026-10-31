# ============================================
# PROJET - ENTRAÎNEMENT SUR TOUTES LES DONNÉES
# ============================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import joblib
import sys

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# ============================================
# CONFIGURATION DES CHEMINS
# ============================================
BASE_DIR = r"D:\S8\Machine Learning\monTP\Projet"
data_dir = os.path.join(BASE_DIR, "data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(models_dir, exist_ok=True)

print(f"Base directory : {BASE_DIR}")
print(f"Train directory : {train_dir}")
print(f"Test directory : {test_dir}")
print(f"Models directory : {models_dir}")

# Verification des dossiers
if not os.path.exists(train_dir):
    print(f"ERREUR: Dossier train introuvable: {train_dir}")
    sys.exit(1)

if not os.path.exists(test_dir):
    print(f"ERREUR: Dossier test introuvable: {test_dir}")
    sys.exit(1)

# ============================================
# FONCTION POUR CHARGER LES IMAGES
# ============================================
def load_images_from_directory(directory):
    """Charge toutes les images d'un dossier"""
    if not os.path.exists(directory):
        return np.array([]), np.array([]), []
    
    classes = sorted(os.listdir(directory))
    if not classes:
        return np.array([]), np.array([]), []
    
    X, y = [], []
    corrupted = 0
    total = 0
    
    for label, cls in enumerate(classes):
        folder = os.path.join(directory, cls)
        if not os.path.exists(folder):
            continue
        
        for img_name in os.listdir(folder):
            total += 1
            img_path = os.path.join(folder, img_name)
            try:
                img = np.array(Image.open(img_path).convert("L"))
                if img.mean() >= 5:
                    img = cv2.resize(img, (48, 48))
                    X.append(img)
                    y.append(label)
                else:
                    corrupted += 1
            except Exception:
                corrupted += 1
                continue
    
    print(f"  Total: {total}, Corrompues: {corrupted}, Gardees: {len(X)}")
    return np.array(X), np.array(y), classes

# ============================================
# CHARGEMENT DES DONNEES
# ============================================
print("\n" + "="*50)
print("CHARGEMENT DES DONNEES")
print("="*50)

print("\nChargement des donnees d'entrainement...")
X_train, y_train, classes_train = load_images_from_directory(train_dir)

print("\nChargement des donnees de test...")
X_test, y_test, classes_test = load_images_from_directory(test_dir)

# Verification des classes
if len(classes_train) > 0:
    classes = classes_train
elif len(classes_test) > 0:
    classes = classes_test
else:
    print("ERREUR: Aucune classe trouvee")
    sys.exit(1)

print(f"\nClasses: {classes}")
print(f"Nombre de classes: {len(classes)}")

# ============================================
# CONCATENATION DES DONNEES
# ============================================
print("\n" + "="*50)
print("CONCATENATION DES DONNEES")
print("="*50)

if len(X_train) > 0 and len(X_test) > 0:
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
elif len(X_train) > 0:
    X_all = X_train
    y_all = y_train
else:
    X_all = X_test
    y_all = y_test

print(f"Donnees totales: {X_all.shape}")
print(f"Labels totaux: {y_all.shape}")

# Distribution des classes
unique, counts = np.unique(y_all, return_counts=True)
print("\nDistribution des classes:")
for cls, count in zip(classes, counts):
    print(f"  {cls:10s}: {count} images")
print(f"  {'Total':10s}: {sum(counts)} images")

# ============================================
# PRETRAITEMENT
# ============================================
print("\n" + "="*50)
print("PRETRAITEMENT")
print("="*50)

IMG_SIZE = 48

# Normalisation
X_cnn = X_all / 255.0
X_cnn = X_cnn.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# One-hot encoding
y_cnn = tf.keras.utils.to_categorical(y_all)

print(f"X_cnn shape: {X_cnn.shape}")
print(f"y_cnn shape: {y_cnn.shape}")

# ============================================
# CONSTRUCTION DU CNN
# ============================================
print("\n" + "="*50)
print("CONSTRUCTION DU CNN")
print("="*50)

cnn_model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# ============================================
# ENTRAINEMENT DU CNN
# ============================================
print("\n" + "="*50)
print("ENTRAINEMENT DU CNN")
print("="*50)

history = cnn_model.fit(X_cnn, y_cnn, epochs=10, batch_size=32, verbose=1)
print(f"CNN entraine avec accuracy: {history.history['accuracy'][-1]:.4f}")

# ============================================
# FEATURE EXTRACTOR
# ============================================
print("\n" + "="*50)
print("CONSTRUCTION DU FEATURE EXTRACTOR")
print("="*50)

cnn_feature_extractor = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu')
])

# Copie des poids
for i in range(len(cnn_feature_extractor.layers)):
    cnn_feature_extractor.layers[i].set_weights(cnn_model.layers[i].get_weights())

print("Feature extractor cree")

# ============================================
# EXTRACTION DES FEATURES
# ============================================
print("\n" + "="*50)
print("EXTRACTION DES FEATURES")
print("="*50)

X_features = cnn_feature_extractor.predict(X_cnn, verbose=1)
print(f"Features extraites: {X_features.shape}")

# ============================================
# ENTRAINEMENT SVM
# ============================================
print("\n" + "="*50)
print("ENTRAINEMENT SVM RBF")
print("="*50)

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
])

pipeline_svm.fit(X_features, y_all)
print("SVM RBF entraine avec succes")

# ============================================
# SAUVEGARDE
# ============================================
print("\n" + "="*50)
print("SAUVEGARDE DES MODELES")
print("="*50)

# SVM
svm_path = os.path.join(models_dir, "svm_model.pkl")
joblib.dump(pipeline_svm, svm_path)
print(f"Modele SVM sauvegarde: {svm_path}")

# CNN complet
cnn_path = os.path.join(models_dir, "cnn_model.h5")
cnn_model.save(cnn_path)
print(f"Modele CNN sauvegarde: {cnn_path}")

# Feature extractor
cnn_feature_path = os.path.join(models_dir, "cnn_feature_extractor.h5")
cnn_feature_extractor.save(cnn_feature_path)
print(f"Feature extractor sauvegarde: {cnn_feature_path}")

# Classes
classes_path = os.path.join(models_dir, "classes.npy")
np.save(classes_path, classes)
print(f"Classes sauvegardees: {classes_path}")

print("\n" + "="*50)
print("ENTRAINEMENT TERMINE AVEC SUCCES")
print("="*50)