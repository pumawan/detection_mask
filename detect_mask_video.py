import numpy as np
import cv2
import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("[INFO] Memuat model deteksi wajah...")
face_net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", 
                                    "models/res10_300x300_ssd_iter_140000.caffemodel")

print("[INFO] Memuat model deteksi masker Keras 3...")
mask_net = keras.models.load_model("models/mask_detector.keras")

cap = cv2.VideoCapture(0)
min_confidence = 0.5

while True:
    ret, frame = cap.read()
    if not ret: break

    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            
            if face.size > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                preds = mask_net.predict(face, verbose=0)
                (mask, withoutMask) = preds[0]

                label = "Bermasker" if mask > withoutMask else "Tanpa Masker"
                color = (0, 255, 0) if label == "Bermasker" else (0, 0, 255)

                text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
                cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    cv2.imshow("Mask Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()