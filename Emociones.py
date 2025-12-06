import cv2
import numpy as np
from tensorflow.keras.models import load_model


modelo = load_model
(
    r"C:\Users\Usuario\OneDrive\Escritorio\Escritorio ciclo 2025-2\Repositorio\Python machine\.venv\modelo_emos.h5"
)

emociones = ['IRA', 'DESAGRADO', 'MIEDO', 'ALEGRIA', 'NORMAL', 'TRISTEZA', 'SORPRESA']

detector = cv2.CascadeClassifier
(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def main():

    print("▶ Intentando abrir cámara...")
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)   

    if not cam.isOpened():
        print("❌ Error: No se pudo abrir la cámara en índice 0.")
        return

    print("✅ Cámara abierta correctamente.\nPresiona Q para salir.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Error: No se logró leer la cámara.")
            break

        # Convertir a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        rostros = detector.detectMultiScale(
            gris,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(70, 70)
        )

        for (x, y, w, h) in rostros:
            rostro = gris[y:y+h, x:x+w]

            # Preparar para el modelo
            rostro = cv2.resize(rostro, (48, 48))
            rostro = rostro.astype("float32") / 255.0
            rostro = np.expand_dims(rostro, axis=(0, -1))  # (1,48,48,1)

            # Predicción
            pred = modelo.predict(rostro, verbose=0)
            emocion = emociones[np.argmax(pred)]

            # Dibujar
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emocion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Mostrar pantalla
        cv2.imshow("Detector de Emociones", frame)

        # Salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# -----------------------------------------------
# 🔹 Ejecutar script
# -----------------------------------------------
if __name__ == "__main__":
    main()
    