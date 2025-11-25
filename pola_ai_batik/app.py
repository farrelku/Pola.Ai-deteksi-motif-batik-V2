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
import os

app = FastAPI(title="POLA.AI Batik Classification")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan data
try:
    model = tf.keras.models.load_model("batik_resnet50_model.h5")
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

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    image = image / 255.0
    return np.expand_dims(image, axis=0)

@app.get("/")
async def read_index():
    """Serve the main HTML page"""
    return FileResponse('index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validasi file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")
        
        # Baca dan process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess dan predict
        img_input = preprocess_image(image)
        pred = model.predict(img_input, verbose=0)  # verbose=0 untuk suppress output
        idx = np.argmax(pred)
        
        # Get prediction result
        prediction_name = list(labels.keys())[list(labels.values()).index(int(idx))]
        description = DESKRIPSI.get(prediction_name, "Deskripsi belum tersedia.")
        
        return {
            "motif": prediction_name,
            "deskripsi": description,
            "confidence": float(np.max(pred))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Serve static files (HTML, CSS, JS, images)
app.mount("/", StaticFiles(directory="."), name="static")

# HAPUS bagian if __name__ == "__main__" untuk Hugging Face
# Hugging Face akan run server secara otomatis