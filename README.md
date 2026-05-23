# 🚗 Vehicle Damage Detection Dataset & Benchmark

> **A Novel Dataset for Vehicle Damage Detection with Benchmark Evaluation of Deep Learning Models**

[![IEEE LATAM](https://img.shields.io/badge/IEEE-LATAM-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green)]()
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v10%20%7C%20v11-orange)]()
[![RT-DETR](https://img.shields.io/badge/RT--DETR-Transformer-red)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## Overview

This repository contains the official implementation, experiments, and dataset associated with the paper:

### **A Novel Dataset for Vehicle Damage Detection with Benchmark Evaluation of Deep Learning Models**

The project introduces a **publicly available dataset** for vehicle damage detection designed to represent **real-world logistics and inspection scenarios**.  
The dataset includes challenging conditions such as:

- Small damage regions
- Lighting variation
- Perspective distortion
- Class imbalance
- High intra-class variability

Additionally, this work provides a benchmark evaluation of several state-of-the-art object detection models, including:

- YOLOv8
- YOLOv10
- YOLO11
- RT-DETR

The results demonstrate that lightweight YOLO models provide the best trade-off between detection accuracy and computational efficiency. 

---

# Authors

- **Marcello Henrique da Costa Grec**
- **Charles Henrique Porto Ferreira**
- **Danilo Hernani Perico** *(Member, IEEE)*

---

# Paper Abstract

Vehicle damage during transportation and logistics operations remains a critical issue, leading to increased operational costs, insurance claims, and delays in product availability.

This work introduces a novel publicly available dataset for vehicle damage detection comprising **2,290 annotated images** across three damage categories:

- Scratched
- Dented
- Broken

The dataset reflects realistic inspection conditions and includes benchmark evaluations using multiple deep learning object detection models. Experimental results show that lightweight YOLO-based models achieve the best balance between accuracy and computational efficiency, reaching **mAP@0.5 up to 0.64**. 

---

# Key Contributions

- ✅ Publicly available vehicle damage dataset
- ✅ Real-world inspection conditions
- ✅ Benchmark evaluation of modern object detectors
- ✅ Computational efficiency analysis
- ✅ Privacy-preserving anonymization process
- ✅ Reproducible experiments

---

# Dataset Information

## Dataset Statistics

| Class | Images | Objects | Avg. per Image | Avg. Area |
|---|---|---|---|---|
| Scratched | 1622 | 3465 | 2.14 | 2.65% |
| Dented | 550 | 870 | 1.58 | 8.74% |
| Broken | 383 | 524 | 1.37 | 12.46% |

The dataset contains:

- **2,290 annotated images**
- Bounding box annotations
- Multiple damages per image
- Real-world logistics scenarios
- RGB images with diverse resolutions


---

# Damage Categories

The dataset contains three vehicle damage categories:

| Class | Description |
|---|---|
| Scratched | Superficial paint damage |
| Dented | Structural deformation |
| Broken | Detached or missing components |


---

# Benchmark Results

## Detection Performance

| Model | Precision | Recall | F1-score | mAP@0.5 |
|---|---|---|---|---|
| YOLOv8n | 0.85 | 0.59 | 0.67 | 0.62 |
| YOLOv8s | 0.82 | 0.62 | 0.65 | **0.64** |
| YOLOv10n | 0.78 | 0.60 | 0.60 | 0.50 |
| YOLOv10s | 0.85 | 0.59 | 0.64 | 0.61 |
| YOLOv10m | 0.66 | 0.47 | 0.53 | 0.52 |
| YOLO11n | 0.80 | 0.52 | 0.65 | 0.63 |
| YOLO11s | 0.74 | 0.54 | 0.63 | **0.64** |
| RT-DETR | 0.84 | 0.54 | 0.64 | 0.62 |


---


# Experimental Setup

- **GPU:** NVIDIA RTX 4050
- **CPU:** Intel Core i7
- **Framework:** Ultralytics
- **Language:** Python
- **Training:** 100 epochs
- **Learning Rate:** 0.001429 (YOLO models)


