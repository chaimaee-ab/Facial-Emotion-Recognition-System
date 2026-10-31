import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Charger modèle
model = load_model(os.path.join(BASE_DIR, "models/cnn_model.h5"))

emotion_labels = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

avatar_type = input("Choisir avatar (male/female): ")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        preds = model.predict(face)
        emotion = emotion_labels[np.argmax(preds)]

        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # chemin avatar corrigé
        avatar_path = os.path.join(
            BASE_DIR, "static/emojis", avatar_type, f"{emotion}.png"
        )

        avatar_img = cv2.imread(avatar_path)

        if avatar_img is not None:
            avatar_img = cv2.resize(avatar_img, (200, 200))
            cv2.imshow("Avatar", avatar_img)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()