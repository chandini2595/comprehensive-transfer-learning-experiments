# 🔥 Deep Learning Project Series - Transfer Learning, Supervised Contrastive Learning & SOTA Vision Models

This repository contains a collection of Colab notebooks demonstrating the power of supervised contrastive learning, transfer learning on various modalities (images, video, audio, text), zero-shot models like CLIP, and end-to-end pipelines using SOTA models on real datasets.

---

## 📌 Table of Contents

1. [Supervised Contrastive Learning vs Softmax Classification](#1-supervised-contrastive-learning)
2. [Transfer Learning on Various Modalities](#2-transfer-learning-on-various-modalities)
   - [Images](#images)
   - [Video](#video)
   - [Audio](#audio)
   - [NLP](#nlp)
3. [Zero-Shot Transfer Learning](#3-zero-shot-transfer-learning)
4. [Image Classification on MNIST, FashionMNIST, CIFAR10](#4-mnist-fashionmnist-cifar10)
   - EfficientNet, BiT, MLP-Mixer, ConvNeXt V2
5. [X-ray and CT Scan Classification](#5-xray-and-3d-ctscan-classification)

---

## ✅ 1. Supervised Contrastive Learning

**🔗 Notebook:** [Supervised Contrastive vs Softmax Classification](https://colab.research.google.com/drive/1cQAE2wWuHXQYqCcbvqY2lXvB9ThWcLTb?usp=sharing)

This notebook compares:
- Traditional classification using softmax loss
- Supervised contrastive learning using a contrastive loss (`SupConLoss`)
- Visualizations using t-SNE for embedding spaces

**📊 Visualizations**:
- Embedding space before/after contrastive training
- Accuracy comparisons
- Training loss plots

---

## ✅ 2. Transfer Learning on Various Modalities

### 🖼️ Images
- **Colab:** [Transfer Learning on Dogs vs Cats](https://colab.research.google.com/drive/1-K-MmFg8lWkZnWauHhOSjFlY4CdxV6Ey?usp=sharing)
- Use both:
  - Feature Extraction (frozen base model)
  - Fine Tuning (unfreezing top layers)
- Models used: `MobileNetV2`, `EfficientNetB0`

---

### 🎥 Video
- **Colab:** [Action Recognition with TFHub](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/action_recognition_with_tf_hub.ipynb)

---

### 🔊 Audio
- **Colab:** [YAMNet + Transfer Learning on ESC-50](https://colab.research.google.com/github/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_tf2.ipynb)

---

### ✍️ NLP
- **Colab:** [Text Classification with TFHub](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)

---

## ✅ 3. Zero-Shot Transfer Learning

### 🔍 CLIP: Contrastive Language–Image Pretraining

- **Colab:** [Zero-shot classification with CLIP](https://colab.research.google.com/drive/1UudS2pRPkEmfWfRIak6vOdCKS_dyDEtS?usp=sharing)
- Example prompts and categories used in zero-shot mode on custom images

---

### 🌼 BigTransfer (BiT)
- **Colab:** [Flower Classification with BiT](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/bit.ipynb)

---

## ✅ 4. MNIST, FashionMNIST, CIFAR10 with SOTA Vision Models

### 📚 Each dataset includes:

- Pre-trained model (EfficientNetB0, BiT as feature extractor + fine-tune)
- SOTA model variants: MLP-Mixer, ConvNeXt V2

#### 🔢 MNIST
- [Colab MNIST Transfer Learning](https://colab.research.google.com/drive/1HzSPbg9Wzv3kShCdfjykGuVEQb-UT3Ls?usp=sharing)

#### 👗 FashionMNIST
- [Colab FashionMNIST Transfer Learning](https://colab.research.google.com/drive/1_0WVLZFdLyEcRG1SvZzNJKhGQzkEFyAd?usp=sharing)

#### 🧸 CIFAR10
- [Colab CIFAR10 with BiT and MLP-Mixer](https://colab.research.google.com/drive/1L5blblpGdqNZAlxRUMQ7sFzHO7tvZTVS?usp=sharing)

---

## ✅ 5. X-ray and 3D CT Scan Classification

### 🫁 X-ray Pneumonia Detection
- **Colab:** [X-ray Classification using ConvNet](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/xray_classification_with_tpus.ipynb)

### 🧠 3D CT Scan Classification
- **Colab:** [3D CT Scan Tumor Classification](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/3D_image_classification.ipynb)

---

## 🧠 Bonus: Resources & Credits

- [Supervised Contrastive Learning Keras](https://keras.io/examples/vision/supervised-contrastive-learning/)
- [BigTransfer](https://keras.io/examples/vision/bit/)
- [CLIP Model](https://github.com/openai/CLIP)
- [TensorFlow Hub Tutorials](https://tfhub.dev/s?deployment-format=colab)

---

