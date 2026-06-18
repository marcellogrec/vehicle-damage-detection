# 🚗 Vehicle Damage Detection Dataset & Benchmark

> **A Novel Dataset for Vehicle Damage Detection with Benchmark Evaluation of Deep Learning Models**

[![IEEE LATAM](https://img.shields.io/badge/IEEE-LATAM-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green)]()

---

## Overview

This repository contains the official implementation, experiments, and dataset associated with the paper: 10964

### **A Novel Dataset for Vehicle Damage Detection with Benchmark Evaluation of Deep Learning Models**

The project introduces a **publicly available dataset** for vehicle damage detection designed to represent **real-world logistics and inspection scenarios**.  
The dataset includes challenging conditions such as:

- Small damage regions
- Lighting variation
- Perspective distortion
- Class imbalance
- High intra-class variability

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

The dataset reflects realistic inspection conditions and includes benchmark evaluations using multiple deep learning object detection models.

---

# Key Contributions

- Publicly available vehicle damage dataset
- Real-world inspection conditions
- Benchmark evaluation of modern object detectors
- Computational efficiency analysis
- Privacy-preserving anonymization process
- Reproducible experiments

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



# Citation

```bibtex
@article{grec2026vehicle,
  title={A Novel Dataset for Vehicle Damage Detection with Benchmark Evaluation of Deep Learning Models},
  author={Grec, Marcello Henrique da Costa Grec and Ferreira, Charles Henrique Porto and Perico, Danilo Hernani},
  journal={IEEE Latin America Transactions},
  year={2026}
}

