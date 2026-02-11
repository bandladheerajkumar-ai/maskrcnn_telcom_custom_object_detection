# 🏗️ Custom Instance Segmentation for Telecom Infrastructure
### A High-Precision Mask R-CNN Pipeline

This repository showcases the end-to-end development of a custom Object Detection and Instance Segmentation model. Using the **Mask R-CNN** architecture, I engineered a solution to identify, segment, and validate critical components on telecom towers—transforming raw field imagery into actionable business intelligence.

---

## 🛠️ The Technical Workflow: Building from Scratch

### 1. Custom Dataset Engineering
To achieve production-grade accuracy, I managed the complete data lifecycle:
* **Pixel-Level Annotation:** Utilized polygon masking (rather than standard bounding boxes) to ensure the model captures the exact geometry of complex objects like jumpers and weatherproofing seals.
* **Class Balancing:** Engineered a dataset targeting niche classes: `GPS_Antenna`, `Blower`, `EMF_Signboard`, and `Jumper_Labeling`.
* **Robust Augmentation:** Implemented a specialized pipeline (rotation, contrast scaling, and Gaussian noise) to ensure model stability across varying weather conditions and drone camera angles.

### 2. Model Architecture & Training
I leveraged a **Mask R-CNN** framework with a **ResNet-101 backbone** for deep feature extraction.
* **Transfer Learning:** Initialized with COCO weights and fine-tuned on custom telecom-specific layers.
* **RPN Optimization:** Customized Region Proposal Network (RPN) anchor scales to detect both large equipment (blowers) and micro-components (labels).
* **Multi-Task Loss Tuning:** Optimized the combined loss function (classification, localization, and mask accuracy) to ensure high $mAP$ (mean Average Precision).

### 3. Solving Industry-Specific Challenges
| Requirement | Technical Solution | Business Impact |
| :--- | :--- | :--- |
| **GPS Integrity** | Segmentation of antennas + cable weatherproofing. | Prevents network sync failure & equipment damage. |
| **Maintenance Audit** | Detection of blower equipment during cleaning. | 100% automated validation of site maintenance. |
| **Legal Compliance** | Legibility check of EMF Safety Signboards. | Eliminates regulatory fines & legal liability. |

---

## 🚀 Capabilities Demonstrated
This project proves the capability to deliver a "Production-Ready" CV solution based on any custom requirement:
* **Requirement Analysis:** Translating business pain points into computer vision tasks.
* **Custom Training:** Building models that handle industrial edge cases (occlusions, low lighting).
* **Scalable Inference:** Architecting code that can be deployed for high-volume batch processing.

---

## 📂 Project Structure
```text
├── dataset/             # Custom annotated data (Images & JSON)
├── mrcnn/               # Core Mask R-CNN architecture files
├── src/
│   ├── train.py         # Custom training script & hyperparameter config
│   └── inference.py     # Production-ready detection script
├── weights/             # (Proprietary) Trained .h5 model weights
└── README.md
