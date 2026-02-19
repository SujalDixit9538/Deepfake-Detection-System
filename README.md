# 🕵️‍♂️ Deepfake Detection System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/SujalDixit1927/Deepfake-Detection-System)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning pipeline designed to detect facial manipulation and artifacts in both static images and video sequences. Built using Transfer Learning (EfficientNetB0) and deployed as an interactive web application.



## 🚀 Live Demo
**[Click here to test the Live Web App](https://huggingface.co/spaces/SujalDixit1927/Deepfake-Detection-System)**

<img width="1349" height="1094" alt="Screenshot 2026-02-19 at 11-50-10 Deepfake Detection System - a Hugging Face Space by SujalDixit1927" src="https://github.com/user-attachments/assets/c83450d7-ee66-47f7-81cc-ec69419b05bb" /><img width="1347" height="1371" alt="Screenshot 2026-02-19 at 11-51-56 Deepfake Detection System - a Hugging Face Space by SujalDixit1927" src="https://github.com/user-attachments/assets/6ff0e2c1-07df-4862-b91a-f035884d74f4" />

---

## 🛠️ Key Features
* **Dual-Mode Inference:** Seamlessly analyzes both single images and full video files.
* **Optimized Video Processing:** Extracts and processes video frames at **1 FPS** to drastically reduce computational latency while maintaining detection accuracy.
* **Temporal Aggregation:** For videos, the model predicts authenticity frame-by-frame and averages the confidence scores to provide a robust, manipulation-resistant final verdict.
* **Production-Ready UI:** User-friendly interface built with Gradio, accessible via web browser.

---

## 🧠 Model Architecture & Training
This project leverages **Transfer Learning** to achieve high accuracy without requiring massive computational resources.

* **Base Model:** `EfficientNetB0` (Pre-trained on ImageNet). Chosen for its optimal balance between accuracy and computational efficiency (fewer parameters than ResNet/VGG).
* **Custom Top Layers:**
  * Global Average Pooling 2D
  * Batch Normalization & Dropout (0.5) to prevent overfitting
  * Dense Output Layer with Sigmoid activation for binary classification (Real vs. Fake)
* **Dataset:** Trained on a subset of a 140,000+ image deepfake dataset.
* **Performance:** Achieved **~92% Test Accuracy** with early stopping and model checkpointing implemented to save the optimal weights.

---

## 📊 Dataset
Due to storage constraints and best practices, the 140,000+ image dataset is not hosted directly in this repository. 

**To reproduce the training pipeline:**
1. Download the dataset from its original source: **https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/data**
2. Extract the downloaded archive.
3. Place the extracted folders into a `data/` directory at the root of this project.

Expected directory structure:
```text
data/
 ┣ train/
 ┃ ┣ real/
 ┃ ┗ fake/
 ┗ test/
   ┣ real/
   ┗ fake/
```

---

## ⚙️ Local Installation & Setup
To run this project on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/Deepfake-Detection-System.git
cd Deepfake-Detection-System
```

**2. Install dependencies:**
```pip install -r requirements.txt```

**3. Run the application:**
```python app.py```
The app will launch locally at http://127.0.0.1:7860.

## 📂 Project Structure
```
📦 Deepfake-Detection-System
 ┣ 📜 Deepfake_Detection_Project.ipynb  # Complete training pipeline and EDA
 ┣ 📜 app.py                            # Gradio web application script
 ┣ 📜 requirements.txt                  # Environment dependencies
 ┣ 📜 final_deepfake_model.keras        # Saved model weights (hosted on HF)
 ┗ 📜 README.md                         # Project documentation
```
