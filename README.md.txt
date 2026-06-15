🧠 Brain Tumor Segmentation using U-Net
A deep learning-based medical image segmentation project that automatically detects and highlights brain tumors from MRI scans using a U-Net architecture.
📌 Project Overview

This project focuses on automatic brain tumor segmentation from MRI images using deep learning.
It helps in identifying tumor regions, estimating tumor area, and visualizing tumor boundaries for better medical understanding.

The system takes an MRI scan as input and outputs:

Tumor segmentation mask
Tumor highlighted image
Tumor area estimation
Performance evaluation metrics
🧠 Problem Statement

Manual tumor detection from MRI scans is:

Time-consuming
Prone to human error
Requires expert radiologists

This project automates the process using deep learning (U-Net) for faster and more accurate results.

📊 Dataset Used
Dataset: LGG MRI Segmentation Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
Dataset Details:
MRI brain scans in .tif format
Corresponding tumor mask images
Binary segmentation (tumor / no tumor)
⚙️ Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Scikit-Image (GLCM features)
Scikit-Learn
Streamlit (for deployment)
🧪 Methodology / Workflow
1. Data Acquisition
Dataset downloaded using Kaggle API
MRI scans and corresponding masks loaded
2. Data Preprocessing
Image resizing (128 × 128)
Normalization (pixel values scaled to 0–1)
Mask binarization (tumor = 1, background = 0)
Filtering non-tumor slices (mask.sum() > threshold)
3. Feature Enhancement
GLCM (Texture Features) extracted to capture image texture patterns
GAN-based augmentation (simulated noise) used to improve robustness
4. Model Architecture
U-Net CNN Architecture
Encoder: Feature extraction
Bottleneck: Deep feature learning
Decoder: Image reconstruction
Skip connections: Preserve spatial details
5. Model Training
Loss: Binary Crossentropy (and Dice Loss in improved version)
Optimizer: Adam
Epochs: 25–50
Batch Size: 8–16
6. Prediction
Model predicts probability mask
Thresholding applied (> 0.5)
Binary tumor segmentation mask generated
7. Post-processing
Removal of noise using morphological operations
Largest connected region extraction
Improved segmentation accuracy
8. Tumor Area Calculation
Tumor pixels counted from mask
Converted to physical area (mm²)
tumor_area = np.sum(pred_mask)
tumor_area_mm2 = tumor_area * (pixel_spacing ** 2)
📈 Evaluation Metrics
Dice Coefficient
IoU (Intersection over Union)
Accuracy
Precision
Recall
F1 Score
📊 Results
High Dice Score achieved on validation data
Good overlap between predicted and actual tumor regions
Effective tumor boundary detection
🖼️ Sample Outputs
MRI Input

(Add image here)

Predicted Mask

(Add image here)

Overlay Result

(Add image here)

📌 Key Features
Automatic tumor segmentation
Tumor area estimation
Visual overlay output
Clinical-style reporting
Post-processing for noise reduction
🚀 Future Improvements
3D CNN for volumetric tumor analysis
Multi-class tumor classification
Real-time Streamlit web application
Clinical deployment support
🌐 Deployment Plan

The trained model can be deployed using Streamlit:

Upload MRI scan
Predict tumor mask
Display overlay image
Show tumor area and report
📂 Repository Structure
Brain-Tumor-Segmentation/
│
├── model/
├── notebooks/
├── screenshots/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
👨‍💻 Author
Name: Bantu Varshasri
Project: Final Year AI/ML Project
Domain: Medical Image Analysis / Deep Learning
📌 Note

This project is for educational and research purposes. Clinical validation is required before real-world medical usage.