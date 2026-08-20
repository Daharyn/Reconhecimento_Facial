import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────
PASTA_FOTOS   = "fotos_colaboradores"
TAMANHO_IMG   = (224, 224)
EPOCAS        = 20
LEARNING_RATE = 0.0001

# ─────────────────────────────────────────
# 1. CARREGAR FOTOS E CLASSES
# ─────────────────────────────────────────
print("Carregando fotos...")
imagens = []
labels  = []
classes = sorted(os.listdir(PASTA_FOTOS))  # Ex: ["Diego_Felipe", "Jorge_Alan"]

for idx, classe in enumerate(classes):
    caminho_classe = os.path.join(PASTA_FOTOS, classe)
    if not os.path.isdir(caminho_classe):
        continue
    for arquivo in os.listdir(caminho_classe):
        if arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
            caminho_img = os.path.join(caminho_classe, arquivo)
            img = Image.open(caminho_img).convert("RGB").resize(TAMANHO_IMG)
            imagens.append(preprocess_input(np.array(img, dtype=np.float32)))
            labels.append(idx)

X = np.array(imagens)
y = np.array(labels)
print(f"{len(classes)} classes encontradas: {classes}")
print(f"{len(X)} imagens carregadas.")

# ─────────────────────────────────────────
# 2. CONSTRUIR O MODELO
# ─────────────────────────────────────────
print("Construindo modelo...")
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False

x   = GlobalAveragePooling2D()(base.output)
out = Dense(len(classes), activation="softmax")(x)
modelo = Model(inputs=base.input, outputs=out)
modelo.compile(optimizer=Adam(LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ─────────────────────────────────────────
# 3. TREINAR
# ─────────────────────────────────────────
print("Treinando...")
modelo.fit(X, y, epochs=EPOCAS, batch_size=8, verbose=1)

# ─────────────────────────────────────────
# 4. EXPORTAR NO FORMATO TEACHABLE MACHINE
# ─────────────────────────────────────────
print("Exportando modelo...")
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(modelo, ".")

# Atualiza o metadata.json com as classes atuais
metadata = {
    "tfjsVersion":      "3.x",
    "tmVersion":        "2.x",
    "packageVersion":   "0.8.x",
    "packageName":      "@teachablemachine/image",
    "timeStamp":        "",
    "userMetadata":     {},
    "modelName":        "tm-my-image-model",
    "labels":           classes,
    "imageSize":        224
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Modelo exportado com sucesso!")
print(f"Classes: {classes}")
