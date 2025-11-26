from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
import pickle
import json
from PIL import Image
import io
import gdown
import os

app = FastAPI(title="POLA.AI Batik Classification")

# -------------------
# CORS middleware
# -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Download model dari Google Drive
# -------------------
def download_model():
    model_path = "batik_resnet50_model.h5"
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        # GANTI DENGAN FILE_ID KAMU
        url = "https://drive.google.com/uc?id=1uXlA_-GJ91FnWL_u6dgitOT8pM6pxD_b"
        gdown.download(url, model_path, quiet=False)
        print("✅ Model downloaded successfully!")
    return model_path

# -------------------
# Load model, labels, deskripsi
# -------------------
try:
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)
    print("✅ Labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    labels = {}

try:
    with open("deskripsi_batik.json", "r", encoding="utf-8") as f:
        DESKRIPSI = json.load(f)
    print("✅ Descriptions loaded successfully!")
except Exception as e:
    print(f"❌ Error loading descriptions: {e}")
    DESKRIPSI = {}

# -------------------
# Helper: preprocess image
# -------------------
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# -------------------
# Routes
# -------------------
@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint untuk prediksi motif batik"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")

        if model is None:
            raise HTTPException(status_code=500, detail="Model tidak tersedia")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_input = preprocess_image(image)

        pred = model.predict(img_input, verbose=0)
        idx = int(np.argmax(pred))
        prediction_name = list(labels.keys())[list(labels.values()).index(idx)]
        description = DESKRIPSI.get(prediction_name, "Deskripsi belum tersedia.")

        return {
            "motif": prediction_name,
            "deskripsi": description,
            "confidence": float(np.max(pred))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# -------------------
# Serve static files (CSS, JS, images)
# -------------------
app.mount("/static", StaticFiles(directory="."), name="static")