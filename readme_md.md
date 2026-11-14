# Deep Learning for Computer Vision: CNNs, InceptionNet & ResNet-50

A hands-on Jupyter Notebook guide covering the foundational theory of Convolutional Neural Networks (CNNs) and practical image classification implementations using **TensorFlow / Keras**.

---

## 📋 Overview

This repository walks through computer vision concepts from the ground up—starting with how digital images and kernels work, moving through custom CNN architectures for handwritten digit and object classification, and concluding with inference using pre-trained **ResNet-50**.

### Key Topics Covered
1. **Computer Vision Fundamentals**
   * Digital image representation (pixel grids and numerical vectors)
   * Image kernels & convolutions (edge detection, blurring, smoothing)
   * Perceptron recap (weights, bias, net sum, activation functions)
   * Core CNN building blocks: **Convolutional Layers**, **Pooling Layers (Max Pooling)**, and **Fully Connected Layers**
2. **CNN for MNIST Digit Classification**
   * Data normalization (`0–255` scaled to `0–1`), reshaping for single-channel input `(28, 28, 1)`, and one-hot encoding (`to_categorical`)
   * Building a Sequential CNN (`Conv2D` → `MaxPool2D` → `Flatten` → `Dense`)
   * Training with `EarlyStopping` callbacks
   * Model evaluation using accuracy/loss curves, classification reports, and a Seaborn confusion matrix heatmap
3. **InceptionNet Module on CIFAR-10**
   * Multi-scale feature extraction using parallel `1x1`, `3x3`, and `5x5` convolutions alongside a `3x3` max-pooling tower
   * Building a custom architecture via the Keras Functional API (`Concatenate`)
   * Training across 10 RGB image categories `(32, 32, 3)`
4. **Pre-trained ResNet-50 Inference**
   * Loading `ResNet50` with `imagenet` weights
   * Preprocessing custom images (`224x224`) and decoding top-3 ImageNet predictions

---

## 🛠️ Tech Stack & Dependencies

* **Python 3.8+**
* **TensorFlow / Keras** — Model building, datasets (`mnist`, `cifar10`), and pre-trained `ResNet50`
* **NumPy & Pandas** — Data manipulation and training history tracking
* **Matplotlib & Seaborn** — Image visualization, training curves, and confusion matrices
* **Scikit-Learn** — `classification_report` and `confusion_matrix`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook
```
*(Alternatively, you can upload the `.ipynb` file directly to **Google Colab** to leverage free GPU acceleration for faster CIFAR-10 training.)*

---

## 📊 Model Highlights & Results

| Model / Architecture | Dataset | Input Shape | Trainable Params | Highlights |
| :--- | :--- | :--- | :--- | :--- |
| **Custom Sequential CNN** | MNIST (60k train / 10k test) | `(28, 28, 1)` | ~1.81M | Achieved **~98.5% test accuracy** in 5 epochs with Early Stopping |
| **Custom Inception Block** | CIFAR-10 (50k train / 10k test) | `(32, 32, 3)` | ~2.11M | Parallel `1x1`, `3x3`, `5x5` conv towers + pooling concatenated |
| **Pre-trained ResNet-50** | Custom Image (ImageNet) | `(224, 224, 3)` | Pre-trained | Zero-shot top-3 class prediction (`Labrador_retriever`, `golden_retriever`, `kuvasz`) |

> **Note for ResNet-50 Section:** Update `img_path` in the final code block to point to your own local image file before running inference.

---

## 🔗 Useful Resources Referenced in the Notebook

* [Interactive Image Kernels Visualization (Setosa)](https://setosa.io/ev/image-kernels/)
* [Keras Metrics Documentation](https://keras.io/metrics/)
* [Keras Pre-trained Applications API](https://keras.io/api/applications/)