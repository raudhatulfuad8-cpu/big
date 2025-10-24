import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Deteksi & Klasifikasi Gambar AI",
    page_icon="🦁",
    layout="wide"
)

# ==============================
# Styling Custom
# ==============================
st.markdown("""
    <style>
        .title {font-size:2rem; font-weight:700; color:#1E3A8A;}
        .subtitle {font-size:1.1rem; color:#4B5563; margin-bottom:1.5rem;}
        .result-box {
            background-color: #EEF2FF;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            color: #1E3A8A;
            font-weight: bold;
            font-size: 1.2rem;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")
    classifier = tf.keras.models.load_model("classifier_model.keras")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model: {e}")
    st.stop()

# ==============================
# Antarmuka Aplikasi
# ==============================
st.markdown('<p class="title">✨ Deteksi & Klasifikasi Gambar AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Unggah gambar untuk mendeteksi objek dengan YOLOv8 dan mengklasifikasikan spesies dengan model Keras (.h5)</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah gambar di sini (format: JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_input = Image.open(uploaded_file)
    st.image(image_input, caption="📸 Gambar yang diunggah", use_column_width=True)

    if st.button("🚀 Jalankan Prediksi"):
        st.write("⏳ Sedang memproses...")

        # ==============================
        # YOLOv8 DETEKSI OBJEK
        # ==============================
        try:
            results = yolo_model.predict(image_input, conf=0.5)
            annotated_frame = results[0].plot()
            st.image(annotated_frame, caption="🔍 Hasil Deteksi YOLOv8", use_column_width=True)
        except Exception as e:
            st.error(f"⚠️ Kesalahan YOLOv8: {e}")

        # ==============================
        # KLASIFIKASI MODEL .H5
        # ==============================
        try:
            # Pastikan gambar RGB (bukan grayscale)
            if image_input.mode != "RGB":
                image_input = image_input.convert("RGB")

            # Konversi ke array tanpa ukuran tetap
            img_array = image.img_to_array(image_input)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            pred_class = np.argmax(prediction, axis=1)[0]

            # Label contoh (ubah sesuai dengan kelas model kamu)
            labels = ['Cheetah', 'Lion']
            predicted_label = labels[pred_class] if pred_class < len(labels) else "Unknown"

            st.markdown(f"""
            <div class="result-box">
                🧩 Hasil Klasifikasi:<br><br>
                <span style='font-size:1.6rem;'>{predicted_label}</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Gagal menjalankan model Keras: {e}")
else:
    st.info("Silakan unggah gambar untuk memulai prediksi.")

# ==============================
# Footer
# ==============================
st.markdown("""
<hr>
<p style='text-align:center; color:#6B7280;'>
Dibuat oleh <b>Raudah ✨</b> dengan bantuan Upus 💙
</p>
""", unsafe_allow_html=True)
