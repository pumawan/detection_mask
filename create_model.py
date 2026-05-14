import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_PATH = "dataset/images"
ANNOT_PATH = "dataset/annotations"

data = []
labels = []

print("[INFO] Memproses gambar dan label...")

for xml_file in os.listdir(ANNOT_PATH):
    tree = ET.parse(os.path.join(ANNOT_PATH, xml_file))
    root = tree.getroot()
    img_name = root.find("filename").text
    img_path = os.path.join(IMAGE_PATH, img_name)

    image = cv2.imread(img_path)
    if image is None: continue
    
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label == "mask_weared_incorrectly" or label=="without_mask": 
            label = "no_mask"
        else:
            label = "mask"
        
        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)

        face = image[ymin:ymax, xmin:xmax]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(face)

        data.append(face)
        labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] Memulai training...")
opt = Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(trainX, trainY, batch_size=32, steps_per_epoch=len(trainX)//32, 
          validation_data=(testX, testY), epochs=10)

print("[INFO] Menyimpan model...")
model.save("models/mask_detector.keras")
print("[SUCCESS] Model mask_detector.keras berhasil dibuat!")