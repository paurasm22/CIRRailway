from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your model
model = load_model("MobileNetv2.h5")  # ensure this file is in the repo

# Define class labels (example)
CLASS_NAMES = ["broken_benches", "garbage", "potholes", "streetlight"]

# Preprocess image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["file"]
    image = preprocess_image(file.read())
    predictions = model.predict(image)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return jsonify({"class": predicted_class, "confidence": confidence})

@app.route("/", methods=["GET"])
def home():
    return "Flask API is running!"

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

