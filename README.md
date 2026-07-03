# 🧬 Automated Semen Viability and Counting Analysis using YOLOv8

## 📌 Project Overview

This project is being developed as part of my internship at **eVerse.AI**. The objective is to automate semen analysis using Artificial Intelligence and Computer Vision techniques.

The current phase of the project focuses on training a **YOLOv8 object detection model** to detect and count sperm cells from microscopic images. This serves as the foundation for future modules such as sperm tracking, motility analysis, and viability assessment.

---

## 🎯 Project Objectives

- Detect sperm cells using YOLOv8
- Count sperm cells automatically
- Build a reusable preprocessing pipeline
- Prepare datasets in YOLO format
- Create a scalable pipeline for future sperm tracking and viability analysis

---

## 🛠 Technologies Used

- Python 3.x
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML

---

## 📁 Project Structure

```
Automated_Semen_Analysis/
│
├── configs/
│   └── config.yaml              # Project configuration
│
├── src/
│   ├── detection/
│   │   └── train_yolo.py        # YOLOv8 training script
│   │
│   ├── preprocessing/
│   │   ├── xml_to_yolo.py
│   │   ├── split_dataset.py
│   │   ├── dataset_analysis.py
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── helpers.py
│       ├── logger.py
│       └── __init__.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙ Dataset Preparation

The dataset is preprocessed before training using the following pipeline:

1. Analyze the dataset
2. Convert XML annotations to YOLO format
3. Split the dataset into:
   - Training
   - Validation
   - Testing
4. Generate the required YOLO directory structure

> **Note:** The dataset is not included in this repository because of its large size and privacy considerations.

---

## 🚀 Model Training

The YOLOv8 model is trained using the training script:

```bash
python src/detection/train_yolo.py
```

Training configuration is managed through:

```
configs/config.yaml
```

---

## 📊 Current Progress

✔ Dataset preprocessing pipeline completed

✔ XML to YOLO annotation conversion

✔ Dataset splitting

✔ YOLOv8 training pipeline

✔ Project structure and configuration

---

## 🔜 Future Work

- ByteTrack integration for sperm tracking
- DeepSORT/BoT-SORT evaluation
- Sperm motility analysis
- Viability classification
- Morphology analysis
- Streamlit-based deployment
- Performance optimization

---

## 📌 Repository Notes

This repository contains only the project source code.

The following files are intentionally excluded:

- Dataset
- Trained model weights (`*.pt`)
- Training outputs (`runs/`)
- Logs
- Virtual environment (`venv/`)

These files are excluded using `.gitignore`.

---

## 👩‍💻 Author

**Jaya Bijore**

Intern – eVerse.AI

Project: **Automated Semen Viability and Counting Analysis**

---

## 📄 License

This repository is intended for educational and internship purposes.