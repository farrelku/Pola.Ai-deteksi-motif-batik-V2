import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import json

st.set_page_config(
    page_title="POLA.AI - Batik Classification",
    page_icon="🎨",
    layout="wide"
)

# ----------------------
# CSS custom (mirip dark theme HTML)
# ----------------------
st.markdown("""
<style>
body {
    background-color: #101828;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
}
.stButton>button {
    background-color: #055dbb;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #033f8c;
    color: white;
}
.stAlert {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Hero Section
# ----------------------
st.title("Temukan Nama Motif Budaya Indonesia")
st.subheader("POLA.AI: Platform AI untuk mendeteksi dan mengenali motif batik Indonesia")

# ----------------------
# Features Section (3 ikon)
# ----------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://i.ibb.co/m5ShRx3g/Upload.png", width=80)
    st.subheader("Upload Foto")
    st.write("Unggah foto batik untuk dianalisis oleh POLA.AI.")

with col2:
    st.image("https://i.ibb.co/Ts6ffG3/Artificial-Intelligence-Brain.png", width=80)
    st.subheader("Deteksi Motif Batik")
    st.write("Pola.AI mengenali berbagai motif batik hanya dari satu foto.")

with col3:
    st.image("https://i.ibb.co/BHYwtb3Y/Time-Machine.png", width=80)
    st.subheader("Riwayat Pencarian")
    st.write("Semua motif yang pernah dicari tersimpan rapi di menu riwayat.")

st.markdown("---")

# ----------------------
# Load model, labels, deskripsi dari file lokal
# ----------------------
MODEL_PATH = "batik_resnet50_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

with open("deskripsi_batik.json", "r", encoding="utf-8") as f:
    DESKRIPSI = json.load(f)

# ----------------------
# Helper
# ----------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# ----------------------
# Upload & Prediksi
# ----------------------
st.subheader("Klasifikasi Motif Batik")
uploaded_file = st.file_uploader("Klik atau drop gambar di sini (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview", use_column_width=True)

    if st.button("Prediksi Motif"):
        with st.spinner("Sedang menganalisis gambar..."):
            try:
                img_input = preprocess_image(image)
                pred = model.predict(img_input, verbose=0)
                idx = int(np.argmax(pred))
                prediction_name = list(labels.keys())[list(labels.values()).index(idx)]
                description = DESKRIPSI.get(prediction_name, "Deskripsi belum tersedia.")
                confidence = float(np.max(pred))

                # Tampilkan hasil
                st.success(f"🎯 Nama Motif: **{prediction_name}**")
                st.info(f"Deskripsi: {description}")
                st.write(f"Confidence: {confidence:.2f}")

                # Simpan riwayat di session_state
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {
                    "image": image,
                    "motif": prediction_name,
                    "deskripsi": description
                })
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop()

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

# ----------------------
# Riwayat
# ----------------------
if 'history' in st.session_state and st.session_state.history:
    st.subheader("Riwayat Pencarian")
    for item in st.session_state.history:
        cols = st.columns([1,3])
        with cols[0]:
            st.image(item["image"], use_column_width=True)
        with cols[1]:
            st.markdown(f"**{item['motif']}**")
            st.write(item["deskripsi"])
