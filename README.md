# 🧬 Automated Semen Viability and Counting Analysis using AI & Computer Vision

## 📌 Project Overview

The **Automated Semen Viability and Counting Analysis** project is an end-to-end Artificial Intelligence and Computer Vision system developed during my internship at **eVerse.AI**.

The objective of this project is to automate microscopic semen analysis by detecting sperm cells, tracking their movement, analyzing motility, classifying morphology, estimating viability, and generating a comprehensive semen quality assessment based on **WHO 2021 reference guidelines**.

The system provides an interactive **Streamlit web application** that allows users to upload microscope videos and automatically generate professional reports in **PDF**, **CSV**, and **JSON** formats.

---

# 🎯 Project Objectives

- Detect sperm cells from microscopic videos using YOLOv8
- Track individual sperm using ByteTrack
- Analyze sperm motility using CASA-inspired metrics
- Classify sperm morphology using EfficientNetB0
- Predict sperm viability using Machine Learning
- Calculate WHO 2021 quality score
- Generate professional reports
- Deploy the complete pipeline using Streamlit

---

# 🚀 Features

- ✅ YOLOv8-based sperm detection
- ✅ ByteTrack multi-object tracking
- ✅ CASA-inspired motility analysis
- ✅ EfficientNetB0 morphology classification
- ✅ Random Forest viability prediction
- ✅ WHO 2021 quality scoring
- ✅ Clinical recommendations generation
- ✅ Professional PDF reports
- ✅ CSV summary export
- ✅ JSON report export
- ✅ Interactive Streamlit dashboard
- ✅ Video upload and automatic inference

---

# 🛠 Technologies Used

### Programming

- Python 3.x

### Computer Vision

- YOLOv8 (Ultralytics)
- OpenCV
- ByteTrack
- Pillow
- Scikit-Image

### Deep Learning

- PyTorch
- Torchvision
- EfficientNetB0 (timm)

### Machine Learning

- Scikit-learn
- Random Forest
- XGBoost

### Data Processing

- NumPy
- Pandas
- SciPy

### Visualization

- Matplotlib
- Plotly

### Report Generation

- ReportLab
- FPDF2

### Web Application

- Streamlit

---

# 📁 Project Structure

```text
Automated_Semen_Analysis/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── app.py
│   └── assets/
│
├── configs/
│   └── config.yaml
│
├── models/
│   ├── detection/
│   │   └── best.pt
│   ├── morphology/
│   │   └── efficientnet_morphology.pth
│   └── viability/
│       └── viability_model.pkl
│
├── outputs/
│   ├── csv/
│   ├── detections/
│   ├── json/
│   ├── plots/
│   ├── reports/
│   ├── tracking/
│   └── videos/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── motility/
│   ├── morphology/
│   ├── viability/
│   ├── scoring/
│   ├── reporting/
│   ├── preprocessing/
│   └── utils/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔬 AI Pipeline

```text
Microscope Video
        │
        ▼
YOLOv8 Detection
        │
        ▼
ByteTrack Tracking
        │
        ▼
Motility Analysis
        │
        ▼
Morphology Classification
        │
        ▼
Viability Prediction
        │
        ▼
WHO 2021 Quality Score
        │
        ▼
Generate PDF / CSV / JSON Reports
        │
        ▼
Streamlit Dashboard
```

---

# 📊 Output Metrics

The system automatically calculates:

- Total sperm count
- Unique sperm tracks
- Progressive motility (%)
- Non-progressive motility (%)
- Immotile sperm (%)
- Curvilinear Velocity (VCL)
- Straight Line Velocity (VSL)
- Average Path Velocity (VAP)
- Normal morphology (%)
- Abnormal morphology (%)
- Predicted viability (%)
- WHO 2021 Quality Score
- WHO threshold validation
- Clinical recommendations

---

# 📋 WHO 2021 Reference Criteria

| Parameter | Reference Value |
|------------|-----------------|
| Concentration | ≥16 million/mL |
| Progressive Motility | ≥30% |
| Normal Morphology | ≥4% |
| Vitality | ≥54% |

---

# ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/jayabijore20/Automated_Semen_Analysis.git

cd Automated_Semen_Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app/app.py
```

The application will open in your browser where you can:

- Upload a microscope semen video
- Run AI analysis
- View all metrics
- Download PDF, CSV and JSON reports

---

# 📄 Generated Outputs

The application automatically generates:

- Annotated detection images
- Tracking trajectory visualizations
- Motility analysis plots
- Morphology statistics
- Viability predictions
- WHO quality assessment
- PDF report
- CSV summary
- JSON report

---

# 🤖 AI Models Used

| Module | Model |
|----------|-------------------------|
| Detection | YOLOv8 |
| Tracking | ByteTrack |
| Motility | CASA-inspired Algorithm |
| Morphology | EfficientNetB0 |
| Viability | Random Forest |
| Quality Assessment | WHO 2021 Rule-Based Scoring |

---

# 📂 Dataset

The project was developed using publicly available microscopy datasets.

The dataset is **not included** in this repository because of its large size and licensing considerations.

The preprocessing pipeline includes:

- Dataset analysis
- XML to YOLO conversion
- Dataset splitting
- YOLO dataset generation

---

# 📈 Current Progress

- ✅ Dataset preprocessing completed
- ✅ XML to YOLO annotation conversion
- ✅ Dataset splitting
- ✅ YOLOv8 model training
- ✅ ByteTrack integration
- ✅ Motility analysis
- ✅ Morphology classification
- ✅ Viability prediction
- ✅ WHO quality scoring
- ✅ Report generation
- ✅ Streamlit deployment
- ✅ End-to-end inference pipeline

---

# 🔮 Future Enhancements

- Real-time microscope integration
- GPU acceleration
- Bull-specific semen datasets
- Multi-class morphology classification
- REST API deployment
- Docker containerization
- Cloud deployment
- Performance optimization

---

# ⚠ Disclaimer

This project is intended for **research, educational, and internship purposes only**.

The generated results should **not** be considered a substitute for professional laboratory analysis or medical diagnosis. All results should be interpreted by qualified experts.

---

# 👩‍💻 Author

**Jaya Bijore**

MBA (Business Analytics)

Computer Vision Intern — **eVerse.AI**

Project:
**Automated Semen Viability and Counting Analysis using AI & Computer Vision**

---

# 📜 License

This repository is provided for educational, research, and internship purposes.

Please respect the licenses of the original datasets and third-party libraries used in this project.