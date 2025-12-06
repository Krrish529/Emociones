from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Cargar modelo
modelo = load_model("modelo_emos.h5")

emociones = ['IRA', 'DESAGRADO', 'MIEDO', 'ALEGRIA', 'NORMAL', 'TRISTEZA', 'SORPRESA']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("emos.html")


@app.route("/predict", methods=["POST"])
def predict():
    print("ARCHIVOS RECIBIDOS:", request.files)

    if 'image' not in request.files:
        return "❌ ERROR: No se envió ninguna imagen."

    file = request.files['image']

    if file.filename == "":
        return "❌ ERROR: No seleccionaste ningún archivo."

    # Guardar imagen
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Leer imagen en color
    img = cv2.imread(path)
    if img is None:
        return "❌ ERROR: No se pudo leer la imagen."

    # Preprocesamiento
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = modelo.predict(img, verbose=0)
    emocion = emociones[np.argmax(pred)]

    return render_template("emos.html", filename=file.filename, emocion=emocion)


if __name__ == "__main__":
    # Tomar el puerto que Render asigna
    port = int(os.environ.get("PORT", 5000))
    # Escuchar en todas las interfaces de red
    app.run(host="0.0.0.0", port=port)
