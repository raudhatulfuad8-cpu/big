import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ================================
# KONFIGURASI DASAR
# ================================
st.set_page_config(
    page_title="Lion vs Cheetah Detector 🦁🐆",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
    <style>
    body {background-color: #f9fafb;}
    .title {text-align: center; font-size: 38px; font-weight: bold; color: #ff914d;}
    .sub {text-align: center; font-size: 18px; color: #6c757d; margin-bottom: 30px;}
    .result-box {
        background-color: #fff8e1; 
        border-radius: 15px; 
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Lion or Cheetah Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Deteksi objek dengan YOLOv8 dan klasifikasi citra dengan model .keras</p>", unsafe_allow_html=True)

# ================================
# FUNGSI PEMUATAN MODEL
# ================================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("best.pt")
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model YOLOv8: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("classifier_model.keras", compile=False)
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model Keras (.keras): {e}")
        classifier = None

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ================================
# UPLOAD GAMBAR
# ================================
uploaded_file = st.file_uploader("📸 Upload gambar (lion atau cheetah)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        image_data = Image.open(uploaded_file)
        st.image(image_data, caption="Gambar Asli", use_container_width=True)

    # ================================
    # DETEKSI DENGAN YOLO
    # ================================
    if yolo_model:
        results = yolo_model.predict(source=np.array(image_data), verbose=False)
        annotated_frame = results[0].plot()
        annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        with col2:
            st.image(annotated_image, caption="Hasil Deteksi YOLOv8", use_container_width=True)

    # ================================
    # KLASIFIKASI DENGAN MODEL KERAS
    # ================================
    if classifier:
        # Ubah ke grayscale (karena model kamu dilatih dengan 1 channel)
        img = image_data.convert("L")  
        img = img.resize((225, 225))  # sesuaikan dengan input model kamu
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)
        label = "🦁 Lion" if prediction[0][0] < 0.5 else "🐆 Cheetah"
        confidence = float(prediction[0][0]) if label == "🐆 Cheetah" else 1 - float(prediction[0][0])

        st.markdown("---")
        st.markdown("<h3 style='text-align:center;'>Hasil Klasifikasi</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-box'><h2>{label}</h2><p><b>Tingkat Keyakinan:</b> {confidence:.2%}</p></div>",
            unsafe_allow_html=True
        )
