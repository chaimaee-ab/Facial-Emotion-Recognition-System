# Emojify | Facial Emotion Recognition

**A Machine Learning and Deep Learning application for facial emotion recognition and real-time emoji prediction.**

---

## Overview

Emojify is an intelligent facial emotion recognition application that analyzes facial images, predicts human emotions, and maps each prediction to a corresponding emoji.

The project explores both traditional Machine Learning algorithms and Deep Learning approaches to compare their performance in a multiclass facial emotion classification task. It also integrates real-time emotion detection through a webcam.

## Key Features

- **Facial Emotion Classification:** Recognition of six facial emotion categories.
- **Model Comparison:** Evaluation of traditional Machine Learning algorithms and CNN-based approaches.
- **Real-Time Detection:** Facial emotion prediction using a webcam.
- **Emoji Mapping:** Visualization of the predicted emotion through a corresponding emoji.
- **Interactive Interface:** Webcam interaction and selection between Male and Female emoji categories.

## Emotion Categories

| Class | Emotion |
|---|---|
| 0 | Angry |
| 1 | Fear |
| 2 | Happy |
| 3 | Neutral |
| 4 | Sad |
| 5 | Surprise |

## Technical Approach

### 1. Data Preprocessing

Facial images undergo several preprocessing operations:

- Resizing to 48 × 48 pixels.
- Grayscale conversion.
- Pixel normalization to the [0, 1] range.
- Image flattening for traditional Machine Learning models.
- Histogram of Oriented Gradients (HOG) feature extraction.
- Principal Component Analysis (PCA) for dimensionality reduction.

### 2. Model Development

**Deep Learning**
- Convolutional Neural Network (CNN).
- Conv2D and MaxPooling layers.
- Dense and Dropout layers.
- Softmax output for multiclass classification.

**Traditional Machine Learning**
- K-Nearest Neighbors (KNN).
- Support Vector Machine (SVM): Linear, RBF and Polynomial kernels.
- Decision Tree.
- Random Forest.

### 3. Model Evaluation

The models are evaluated on a separate test dataset to compare their classification performance and generalization capabilities.

The experiments highlight the potential of CNN-based feature extraction combined with traditional classifiers, particularly SVM.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| Computer Vision | OpenCV |
| Data Processing | NumPy |
| Visualization | Matplotlib |
| Web Interface | Streamlit |

## Project Structure

```text
Emojify/
│
├── README.md
├── .gitignore
│
└── Projet/
    ├── app/
    │   ├── realtime_cam.py
    │   └── streamlit_app.py
    │
    ├── notebooks/
    │   └── projet.py
    │
    ├── static/
    │   └── emojis/
    │       ├── female/
    │       └── male/
    │
    ├── Emojify.ipynb
    ├── app.py
    ├── create_notebook.py
    ├── webcam_realtime.py
    └── requirements.txt
```

*Note: The dataset and trained model files are excluded from this repository.*

## Installation and Setup

### Prerequisites

- Python 3.8 or higher.
- A webcam for real-time detection.
- The required facial emotion dataset.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Emojify.git
cd Emojify
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
cd Projet
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the training and testing images in the expected `data/` directory structure, organized by emotion category.

The dataset is not included in this repository.

## Usage

### Model Training

From the `Projet` directory:

```bash
cd notebooks
python projet.py
