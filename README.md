# Vehicle Damage Detection: Dataset & Benchmark

Code for the paper **"Vehicle damage detection: a public dataset and benchmark of deep learning models"**

**Dataset:** [Harvard Dataverse — doi:10.7910/DVN/MV5EME](https://doi.org/10.7910/DVN/MV5EME)

---

## Overview

This repository contains the code used to train and evaluate the object detectors reported in the paper.

The paper introduces a **publicly available dataset of 2,290 images** for vehicle damage detection, acquired directly from **real-world logistics and transportation operations** rather than from web image repositories. The images reflect realistic inspection conditions, including:

- Small damage regions (Scratched instances occupy on average 1.29% of the image)
- Variation in lighting, perspective, and background
- Non-standardized capture devices and resolutions
- Class imbalance
- High intra-class variability

We benchmark **twelve YOLO detectors** (YOLOv8, v9, v10, 11, 12, and 26, in Nano/Tiny and Small configurations) using **10 random seeds per model** and statistical significance testing. We also evaluate **class-aware loss weighting** as a way to address class imbalance.

---

## Authors

- **Marcello Henrique da Costa Grec** (corresponding author), Centro Universitário FEI
- **Charles Henrique Porto Ferreira**, Centro Universitário FEI
- **Danilo Hernani Perico**, Centro Universitário FEI

---

## Dataset

### Damage Categories

| Class | Description |
|---|---|
| Scratched | Superficial damage confined to the paint or coating layer (scratches, scuffs), with no deformation or breach of the material |
| Dented | Surface deformation (depressions, creases, waviness) where the part remains continuous and attached |
| Broken | Loss of material continuity or structural integrity: cracks and tears, displaced parts, punctures, broken lamps or glass, missing components |

### Statistics

| Class | Images containing class | Objects | Avg. objects/image | Avg. area |
|---|---|---|---|---|
| Scratched | 1,623 | 3,467 | 2.14 | 1.29% |
| Dented | 549 | 867 | 1.58 | 6.27% |
| Broken | 384 | 525 | 1.37 | 10.19% |
| **Total** | **2,290** | **4,859** | **2.12** | — |

- Bounding-box annotations, created by a single expert annotator in Supervisely
- An image may contain several damages, of the same class or of different classes
- RGB images at varied resolutions (the most frequent is 1280×719)
