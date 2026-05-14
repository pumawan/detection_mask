# Detection Mask
This project is a real-time Face Mask Detection System built using Python, OpenCV, and Keras/TensorFlow. It utilizes a two-stage deep learning pipeline to identify whether individuals in a video stream are wearing masks correctly, not wearing them, or wearing them improperly.

# 🛠️ Technology Stack
> Computer Vision: OpenCV (DNN module for face localization).
> Deep Learning: TensorFlow 2.15+ / Keras 3.
> Model Architecture: MobileNetV2 (Fine-tuned for binary/multi-class classification).
> Dataset: Kaggle Face Mask Detection Dataset.

# 🚀 Key Features
> Real-time Detection: High-speed processing suitable for webcam or IP camera feeds.
> Dual-Model Pipeline: 
1.  Face Detector: ResNet-10 SSD model for robust face localization.
2.  Classifier: A custom-trained MobileNetV2 .keras model for mask state classification.
> Keras 3 Ready: Fully compatible with the latest Python 3.12+ environments, avoiding legacy naming conflicts.

# 📂 Project Structure
> create_model.py: Script to process the Kaggle XML dataset and train the .keras model.
> detect_mask_video.py: The main execution script for real-time inference via webcam.
> models/: Directory containing the pre-trained face detector and the trained mask classifier.
> dataset/: Directory containing images and annotations the pre-trained Data.
