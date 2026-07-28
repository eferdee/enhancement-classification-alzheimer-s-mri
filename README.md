# 🧠 MRI Image Enhancement for Alzheimer's Disease Classification

An image enhancement-based deep learning application designed to improve the quality of brain MRI images before Alzheimer's disease classification. This project implements a **Nonlinear Activation Free Network (NAFNet)** as the image enhancement model and integrates it with a Flask web application for inference.

## 📌 Overview

Early detection of Alzheimer's disease plays a crucial role in supporting timely diagnosis and treatment. This project focuses on enhancing MRI images using NAFNet before the classification stage, aiming to improve image quality and overall classification performance.

The application provides a simple web interface where users can upload MRI images and obtain enhanced images for further analysis.

## 📖 Research Summary

Alzheimer's disease is a progressive neurodegenerative disorder and the leading cause of dementia. MRI-based classification using deep learning can be significantly affected by image noise, particularly **Rician noise**, which degrades image quality and reduces classification performance.

This project proposes an image enhancement pipeline using the **Nonlinear Activation Free Network (NAFNet)** to improve MRI image quality before Alzheimer's disease classification. The enhanced images are subsequently classified using **ResNet18**, and the performance is evaluated on three image conditions:

- Original MRI images
- MRI images with synthetic Rician noise
- MRI images enhanced using NAFNet

Image enhancement quality is evaluated using **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index Measure (SSIM)**, while classification performance is assessed using **Accuracy** and **Macro F1-score**.

## ✨ Features

- MRI image enhancement using NAFNet
- Flask-based web application
- Upload and process MRI images
- User-friendly web interface
- Ready for local deployment

## 🛠️ Technologies

- Python
- Flask
- PyTorch
- NumPy
- Pillow
- HTML, CSS, JavaScript

## 🏗️ Model Architecture

### Image Enhancement

- Model: **Nonlinear Activation Free Network (NAFNet)**
- Adapted for MRI image enhancement
- Conservative residual scaling wrapper
- Optional flat architecture (without U-Net configuration)

### Classification

- Backbone: **ResNet18**
- Framework: **PyTorch**

## 📊 Experimental Results

### Image Enhancement Performance

| Image Condition | PSNR (dB) | SSIM |
|----------------|----------:|-----:|
| Noisy MRI | 25.91 ± 5.63 | 0.5361 ± 0.0921 |
| Enhanced MRI | **34.57 ± 3.38** | **0.9626 ± 0.0226** |

### Alzheimer's Disease Classification Performance

| Dataset | Accuracy | Precision | Recall | Macro F1-Score |
|---------|---------:|----------:|-------:|---------------:|
| Raw MRI | **98.34%** | **98.75%** | **99.03%** | **98.87%** |
| Noisy MRI | 79.11% | 89.41% | 73.55% | 76.82% |
| Enhanced MRI | **96.78%** | **97.00%** | **98.34%** | **97.61%** |

### Key Findings

- Image enhancement increased **PSNR** from **25.91 dB** to **34.57 dB**.
- **SSIM** improved from **0.5361** to **0.9626**, indicating significantly better structural similarity.
- Classification performance recovered from **79.11%** accuracy on noisy MRI images to **96.78%** after enhancement.
- The proposed enhancement pipeline effectively mitigates the impact of Rician noise while preserving diagnostic information for Alzheimer's disease classification.

## 📦 Pre-trained Models

The trained model weights are **not included** in this repository due to GitHub's file size limitations.

Please download the pre-trained models from the following links:

| Model | Description | Download |
|-------|-------------|----------|
| NAFNet | MRI image enhancement model | **[Google Drive](https://drive.google.com/file/d/1bmBtFX6zEWcD6Qnct1q2nB9Guz2IBiq8/view?usp=drive_link)** |
| Baseline Classifier | Alzheimer's disease classification model | **[Google Drive](https://drive.google.com/file/d/14uF8OsThus7ZD1D_2QIOXapjfaB0n4Dn/view?usp=drive_link)** |

After downloading, place the model files in the project root:

## 📂 Project Structure

```text
my_flask_app/
│
├── app.py
├── best_baseline_classifier.pth
├── .gitignore
├── inference_service.py
├── nafnet_best.pth
├── nafnet_mri.py
├── requirements.txt
├── README.md
├── static/
└── templates/
```

## 🚀 Installation

Clone this repository:

```bash
git clone https://github.com/eferdee/enhancement-classification-alzheimer-s-mri.git
cd enhancement-classification-alzheimer-s-mri
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

## 📊 Dataset

This project uses the **Augmented Alzheimer MRI Dataset** available on Kaggle.

Dataset:
https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

**Note:** The dataset is **not included** in this repository. Please download it directly from the official Kaggle page.

## 📈 Research Summary

- Image Enhancement Model: NAFNet
- Medical Imaging: Brain MRI
- Programming Language: Python
- Framework: Flask
- Deep Learning Framework: PyTorch

## 📄 Publication

This project was developed as part of an undergraduate thesis:

**Image Enhancement Using Nonlinear Activation Free Network for Alzheimer's Disease Classification in MRI Images**

## 📜 License

This repository contains only the source code developed for this project.

The dataset belongs to its respective author and follows the license specified on its official Kaggle page.

## 👨‍💻 Author

**M. Farid Saputra**

- GitHub: https://github.com/eferdee