import cv2
import numpy as np
import os
import sys

from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

def create_cnn_model(img_size=48, num_classes=6):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

MODEL_PATH = 'emojify_cnn_model.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Modèle pré-entraîné chargé avec succès !")
else:
    model = create_cnn_model()
    print("ATTENTION : Modèle pré-entraîné non trouvé.")
    print("Mise en place d'un réseau de neurones vierge pour démo (Prédictions aléatoires).")
    print("Il faudra l'entraîner dans le notebook Emojify.ipynb pour de vrais résultats.")

emotion_to_emoji = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐"
}
EMOTIONS = list(emotion_to_emoji.keys())
IMG_SIZE = 48

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Caméra en cours de démarrage... (Appuyez sur 'q' sur le clavier pour quitter)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lecteur webcam.")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_normalized = roi_gray / 255.0
        roi_prep = np.expand_dims(roi_normalized, axis=(0, -1))
        
        preds = model.predict(roi_prep, verbose=0)
        idx = np.argmax(preds)
        
        emotion = EMOTIONS[idx]
        emoji = emotion_to_emoji[emotion]
        certitude = preds[0][idx] * 100
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        texte = f"{emotion} ({certitude:.0f}%)"
        cv2.putText(frame, texte, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
    cv2.imshow('Emojify IA Detective (Appuyer sur Q pour quitter)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
