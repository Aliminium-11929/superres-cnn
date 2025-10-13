# Super-Resolution CNN (2× Upscaling)

This repository contains a baseline implementation of a **Convolutional Neural Network (CNN)** for **2× image super-resolution**, developed as part of the EECE course project proposal.

---

## 🧠 Project Overview
The goal is to reconstruct high-resolution (HR) images from their low-resolution (LR) counterparts using a deep CNN trained on LR–HR image pairs.  
Each high-resolution image is divided into four quadrants, downsampled, and converted to grayscale to form training pairs.  
The project currently focuses on grayscale upscaling, with color restoration considered as future work.

---

## 🧩 Baseline Model
The baseline follows the **SRCNN architecture**:
- 3 convolutional layers  
- ReLU activations  
- Mean Squared Error (MSE) loss  
- Metrics: PSNR, SSIM  

This model serves as the starting point for deeper architectures (e.g., EDSR-style residual networks).

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/superres-cnn.git
cd superres-cnn
