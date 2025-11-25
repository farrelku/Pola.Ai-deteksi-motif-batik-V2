from flask import Flask, render_template, request, session
import tensorflow as tf
import numpy as np
import pickle
import json
from PIL import Image
from datetime import timedelta
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Tambahkan secret key untuk session
app.permanent_session_lifetime = timedelta(days=1)  # Atur lifetime session

# Load model
model = tf.keras.models.load_model("batik_resnet50_model.h5")

# Load labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Load deskripsi batik
with open("deskripsi_batik.json", "r", encoding="utf-8") as f:
    DESKRIPSI = json.load(f)


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    
    # Jika gambar memiliki alpha channel (RGBA), konversi ke RGB
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    image = image / 255.0
    return np.expand_dims(image, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_name = None
    description = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "Tidak ada file"
            return render_template("index.html", error=error)

        file = request.files["file"]
        
        # Cek jika file dipilih
        if file.filename == '':
            error = "Tidak ada file yang dipilih"
            return render_template("index.html", error=error)

        try:
            img = Image.open(file.stream)
            
            # Preprocess
            input_img = preprocess_image(img)

            # Predict
            prediction = model.predict(input_img)
            idx = np.argmax(prediction)

            prediction_name = list(labels.keys())[list(labels.values()).index(int(idx))]

            # Ambil deskripsi
            description = DESKRIPSI.get(prediction_name, "Deskripsi belum tersedia.")

            # Simpan history dengan deskripsi
            session.permanent = True
            if "history" not in session:
                session["history"] = []

            # Hindari duplikat berdasarkan nama motif
            # Cek apakah motif sudah ada di history
            existing_entry = next((item for item in session["history"] if item["nama"] == prediction_name), None)
            
            if not existing_entry:
                # Tambahkan entry baru ke history
                history_entry = {
                    "nama": prediction_name,
                    "deskripsi": description
                }
                session["history"].append(history_entry)
                session.modified = True  # Pastikan session terupdate

        except Exception as e:
            error = f"Error processing image: {str(e)}"
            return render_template("index.html", error=error)

    return render_template(
        "index.html",
        prediction=prediction_name,
        description=description,
        history=session.get("history", []),
        error=error
    )


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Route untuk menghapus history"""
    session["history"] = []
    return render_template(
        "index.html",
        prediction=None,
        description=None,
        history=[],
        error=None
    )


if __name__ == "__main__":
    app.run(debug=True)
