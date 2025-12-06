import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# ----------------------------
# 📌 RUTAS DE TUS CARPETAS RAF-DB
# ----------------------------
train_dir = r"C:\Users\Usuario\OneDrive\Escritorio\machine\Emos\DATASET\train"
test_dir  = r"C:\Users\Usuario\OneDrive\Escritorio\machine\Emos\DATASET\test"

# ----------------------------
# 📌 CONFIGURACIONES
# ----------------------------

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10   # rápido + buena precisión

# ----------------------------
# 📌 GENERACIÓN DE DATOS
# ----------------------------

train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1/255.0)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ----------------------------
# 📌 MODELO (RÁPIDO Y PRECISO)
# ----------------------------

base = MobileNetV2(include_top=False, weights="imagenet",
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))

base.trainable = False   # Congelar pesos → ENTRENAMIENTO RÁPIDO

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
outputs = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 📌 ENTRENAMIENTO
# ----------------------------

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# ----------------------------
# 📌 GUARDAR MODELO
# ----------------------------

model.save("modelo_emos.h5")
print("Modelo guardado como modelo_rafdb_mobilenet.h5 🎉")
