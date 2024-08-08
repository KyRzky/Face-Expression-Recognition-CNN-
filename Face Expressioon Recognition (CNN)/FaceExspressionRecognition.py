import cv2
import streamlit as st
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
import time

# Load model yang telah dilatih
emotion_model = load_model('expression_detection_modelv29.h5')
emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# Fungsi untuk mendeteksi wajah dalam bingkai
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25, minSize=(30, 30))
    return faces, gray

# Fungsi untuk memprediksi ekspresi wajah
def predict_expression(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)
    prediction = emotion_model.predict(face)
    emotion_label = emotion_labels[np.argmax(prediction)]
    return emotion_label

# Streamlit application
st.title("Face Exspression Recognition")
st.header("Selamat Datang di Sistem Face Exspression Recognition")

st.write("Sistem Ini Merupakan Sistem Untuk Melakukan Pengenalan Dan Klasifikasi Pada Ekspresi Wajah.")
st.write("Petunjuk Penggunaan :")
st.write("1. Tekan Tombol Start Untuk Menyalakan Kamera Untuk Memulai Deteksi Pada Ekspresi Wajah.")
st.write("2. Pastikan Posisi Wajah Berada Pada Frame Kamera")
st.write("3. Tekan Tombol Stop Untuk Mematikan Kamera, Dan Untuk Melihat Ekspresi Apa Saja Yang Terdeteksi Oleh Sistem.")

start_button = st.button('Start')  # Tombol untuk menyalakan kamera
stop_button = st.button('Stop')    # Tombol untuk mematikan kamera
FRAME_WINDOW = st.image([])

# Variabel untuk menyimpan emosi yang terdeteksi
if 'captured_emotions' not in st.session_state:
    st.session_state.captured_emotions = []

# Inisialisasi waktu untuk pembaruan per 5 detik
if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = 0

# Capture video dari kamera
camera = None

if start_button:
    camera = cv2.VideoCapture(0)

while start_button:
    ret, frame = camera.read()
    if not ret:
        st.write("Gagal membuka kamera")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    current_time = time.time()
    
    faces, gray_frame = detect_face(frame)
    for (x, y, w, h) in faces:
        cropped_face = gray_frame[y:y + h, x:x + w]
        emotion_label = predict_expression(cropped_face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Periksa jika lima detik telah berlalu sejak pembaruan terakhir
        if current_time - st.session_state.last_capture_time >= 5:
            st.session_state.captured_emotions.append(emotion_label)
            st.session_state.last_capture_time = current_time

    FRAME_WINDOW.image(frame)

if stop_button:
    st.write("List ekspresi yang muncul dan jumlah kemunculannya:")
    for emotion, count in Counter(st.session_state.captured_emotions).items():
        st.write(f"{emotion}: {count}")

    # Hitung frekuensi setiap ekspresi
    most_common_emotion = Counter(st.session_state.captured_emotions).most_common(1)
    if most_common_emotion:
        most_common_label = most_common_emotion[0][0]
        st.write("Ekspresi yang paling sering muncul setelah kamera dimatikan:", most_common_label)
