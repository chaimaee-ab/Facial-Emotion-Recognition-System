import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(page_title="Emojify", layout="wide")

st.title("😄 Emojify - Emotion Detection")
st.markdown("### Détection des émotions + Avatar dynamique")

# =========================
# CHEMINS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "cnn_model.h5")
emoji_path = os.path.join(BASE_DIR, "static", "emojis")

# =========================
# CHARGEMENT MODELE
# =========================
@st.cache_resource
def load_my_model():
    return load_model(model_path)

model = load_my_model()

emotion_labels = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

# =========================
# SIDEBAR (OPTIONS)
# =========================
st.sidebar.header("⚙️ Paramètres")

avatar_type = st.sidebar.selectbox("Choisir avatar", ["male", "female"])

run = st.sidebar.checkbox("🎥 Activer Webcam")

# =========================
# ZONES AFFICHAGE
# =========================
col1, col2 = st.columns(2)

frame_window = col1.image([])
avatar_window = col2.image([])

emotion_text = st.empty()

# =========================
# DETECTION VISAGE
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

# =========================
# BOUCLE PRINCIPALE
# =========================
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Erreur webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = "neutral"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        preds = model.predict(face)
        emotion = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Afficher webcam
    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Afficher émotion texte
    emotion_text.markdown(f"## Emotion détectée : **{emotion.upper()}**")

    # Charger avatar
    avatar_file = os.path.join(emoji_path, avatar_type, f"{emotion}.png")

    if os.path.exists(avatar_file):
        avatar_window.image(avatar_file, caption="Avatar")

cap.release()