import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# =========================
# 1. Chemins
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "cnn_model.h5")
emoji_path = os.path.join(BASE_DIR, "static", "emojis")

# =========================
# 2. Charger modèle
# =========================
model = load_model(model_path)

emotion_labels = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

# =========================
# 3. Choix avatar
# =========================
avatar_type = input("Choisir avatar (male/female): ")

# =========================
# 4. Détecteur visage
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# 5. Webcam
# =========================
cap = cv2.VideoCapture(0)

print("Appuyez sur 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = "neutral"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Prétraitement
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Prédiction
        preds = model.predict(face)
        emotion = emotion_labels[np.argmax(preds)]

        # Dessiner rectangle + texte
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # =========================
    # 6. Afficher webcam
    # =========================
    cv2.imshow("Camera - Emojify", frame)

    # =========================
    # 7. Afficher avatar
    # =========================
    avatar_file = os.path.join(emoji_path, avatar_type, f"{emotion}.png")

    if os.path.exists(avatar_file):
        avatar_img = cv2.imread(avatar_file)
        avatar_img = cv2.resize(avatar_img, (250, 250))
        cv2.imshow("Avatar", avatar_img)

    # Quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# 8. Nettoyage
# =========================
cap.release()
cv2.destroyAllWindows()