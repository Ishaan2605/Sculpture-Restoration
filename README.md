# GAN-Based Sculpture Restoration using Image-to-Image Translation

## Overview

This project focuses on **restoring eroded or damaged sculptures** using a deep learning pipeline built on **Generative Adversarial Networks (GANs)** and **Computer Vision (CV)** techniques. It utilizes a **Pix2Pix-based architecture enhanced with Self-Attention, AdaIN normalization, and dual discriminators** for high-quality structural and textural restoration.

> Inspired by Indian heritage sites, the model aims to reconstruct missing or deteriorated parts of sculptures and temple artifacts — preserving history through AI.

---

## Features

- Preprocessing pipeline with synthetic damage generation
- Structural guidance using **Harris Corner Detection**
- GAN architecture with:
  - Conditional Pix2Pix base
  - Self-Attention (SAGAN-style) in Generator
  - Adaptive Instance Normalization (AdaIN) for texture handling
  - Dual discriminators: **Global** + **Patch-based**
- Custom loss functions: **L1**, **Adversarial**, **Perceptual (VGG)**, and **SSIM**
- Post-processing using **SRGAN** and traditional **inpainting**
- Evaluation using **SSIM**, **PSNR**, **FID**, and side-by-side visual comparison

---

## Project Structure

```
gan-sculpture-restoration/
│
├── data/                      # Dataset folder (original and damaged images)
├── preprocessing/
│   ├── damage_generator.py   # Adds artificial damage
│   └── preprocess.py         # Corner detection, resizing, etc.
├── models/
│   ├── generator.py          # Pix2Pix Generator with Self-Attention + AdaIN
│   ├── discriminator.py      # Global and PatchGAN Discriminators
│   └── srgan_postprocess.py  # Super-resolution enhancement
├── train.py                  # Training script with multi-loss integration
├── evaluate.py               # Evaluation script with metrics
├── app/
│   ├── webapp.py             # Frontend for image upload and restoration
│   └── templates/            # HTML interface
├── utils/
│   └── losses.py             # Custom loss functions
│
├── checkpoints/              # Saved model weights
├── requirements.txt
└── README.md                 # Project documentation
```

---

## Model Architecture

### Generator

- U-Net based encoder-decoder
- Self-Attention blocks in intermediate layers
- AdaIN for dynamic style-texture adaptation

### Discriminators

- **Global Discriminator**: Evaluates entire image realism
- **Patch Discriminator (PatchGAN)**: Focuses on localized texture realism

---

## Loss Functions

- **Adversarial Loss** — Encourages realistic output generation
- **L1 Loss** — Pixel-wise reconstruction for structural similarity
- **Perceptual Loss (VGG19)** — Preserves high-level texture and features
- **SSIM Loss** — Optimizes for structural similarity between real and generated images

---

## Evaluation Metrics

| Metric        | Description                                |
|---------------|--------------------------------------------|
| SSIM          | Structural Similarity Index                |
| PSNR          | Peak Signal-to-Noise Ratio                 |
| FID Score     | Fréchet Inception Distance                 |
| Visual Output | Side-by-side input and restoration results |

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gan-sculpture-restoration.git
cd gan-sculpture-restoration
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Place original and damaged images inside the `data/` directory. Then run:

```bash
python preprocessing/damage_generator.py
python preprocessing/preprocess.py
```

### 4. Train the Model

```bash
python train.py
```

### 5. Evaluate the Model

```bash
python evaluate.py
```

### 6. Run the Web App

```bash
cd app
python webapp.py
```

---


## Future Work

- Apply on **real archaeological photographs** for field testing
- Integrate **3D structure recovery** using NeRF or depth prediction
- Export model to **ONNX/TensorRT** for deployment on mobile or web

---
