# Soil–Crop Recommendation System using Convolutional Autoencoder (CAE)

## Abstract / Overview

This project builds a **soil–crop recommendation system** that predicts the most suitable crop based on soil nutrients and environmental conditions. We use a **Convolutional Autoencoder (CAE)–based classifier** as the main model and a **Logistic Regression** baseline for comparison. Models are trained on the Kaggle **Crop Recommendation Dataset** and evaluated using accuracy, precision, recall, and F1-score. The work supports **SDG 2 (Zero Hunger)** and **SDG 12 (Responsible Consumption and Production)** by enabling data‑driven, sustainable crop planning.

---

## 1. Introduction

Crop choice is often based on experience or generic advisories, which may not fully reflect **field‑specific soil and climate conditions**. This can lead to suboptimal yields, wasted fertilizers and water, and economic losses. With soil testing and meteorological data becoming widely available, there is an opportunity to use **machine learning** for objective crop recommendation.

**Objectives:**

1. Design a **CAE‑based deep learning model** for soil–crop recommendation.
2. Implement a **Logistic Regression baseline** on standardized raw features.
3. Evaluate both models using accuracy, precision, recall, and F1-score.
4. Provide a simple **inference pipeline** to recommend crops for new soil inputs.
5. Relate the solution to **SDG 2** and **SDG 12**.

---

## 2. Dataset

- **Source:** Kaggle – Crop Recommendation Dataset (`Crop_recommendation.csv`).
- **Path:** `data/soil_crop/Crop_recommendation.csv`.
- **Size:** ~2200 samples, **22 crop classes**.
- **Features (7):**  
  `N, P, K, temperature, humidity, ph, rainfall`
- **Label:** `label` – crop name (22 classes).

**Preprocessing (implemented in `utils/data_loader.py`):**

- Stratified **train/validation/test split** (~70%/15%/15%).
- **Standardization** of features using `StandardScaler`.
- **Label encoding** using `LabelEncoder`.
- For CAE, features reshaped from `(7,)` to `(7, 1)` for Conv1D input.

---
