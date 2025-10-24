import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(
    page_title="Visualisasi Model AI 🧠",
    page_icon="✨",
    layout="wide"
)

# Styling CSS custom untuk tampilan menarik
st.markdown("""
    <style>
        body {
            background-color: #F9FAFB;
            color: #111827;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        .title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
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
# Load Model
# ==============================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("best.pt")  # Model deteksi objek YOLO
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("classifier_model.keras")  # Model klasifikasi
    except Exception as e:
        st.error(f"Gagal memuat model Keras (.h5): {e}")
        classifier = None

    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==============================
# Antarmuka Streamlit
# ==============================
st.markdown('<p class="title">✨ Dashboard Deteksi & Klasifikasi Gambar</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Unggah gambar untuk dideteksi menggunakan YOLOv8 dan diklasifikasikan menggunakan model Keras (.h5)</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah gambar di sini (format: JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_input = Image.open(uploaded_file)
    st.image(image_input, caption="Gambar yang diunggah", use_column_width=True)

    # Tombol Proses
    if st.button("🚀 Jalankan Prediksi"):
        st.write("⏳ Sedang memproses...")

        # ==============================
        # YOLOv8 DETEKSI OBJEK
        # ==============================
        if yolo_model is not None:
            results = yolo_model.predict(image_input, conf=0.5)
            annotated_frame = results[0].plot()
            st.image(annotated_frame, caption="Hasil Deteksi YOLOv8", use_column_width=True)
        else:
            st.warning("⚠️ Model YOLO belum dimuat!")

        # ==============================
        # MODEL KERAS KLASIFIKASI
        # ==============================
        if classifier is not None:
            img_resized = image_input.resize((224, 224))  # sesuaikan ukuran input dengan model kamu
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            pred_class = np.argmax(prediction, axis=1)[0]

            # Label contoh (ubah sesuai dengan kelas model kamu)
            labels = ['Cheetah', 'Lion', 'Leopard', 'Tiger']
            predicted_label = labels[pred_class] if pred_class < len(labels) else "Unknown"

            st.markdown(f"""
            <div class="result-box">
                🧩 Hasil Klasifikasi: <br><br>
                <span style='font-size:1.6rem;'>{predicted_label}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model Keras belum dimuat dengan benar!")

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
