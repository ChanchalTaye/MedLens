# MedLens 

**MedLens** is an AI-powered medical imaging project focused on assisting diagnosis using deep learning models across multiple radiology modalities such as **X-ray, MRI, and Ultrasound**.

The project is designed with a **clean, reproducible, and professional ML workflow**, separating:
- source code
- training logic
- explainability
- models & datasets (external)

---

##  Features

-  Deep learning–based medical image classification
-  Multi-modality support (X-ray, MRI, Ultrasound)
-  Model training pipelines
-  Explainability support (e.g., Grad-CAM / visual insights)
-  Backend + frontend architecture
-  Clean GitHub structure (no datasets, no virtual environments)

---

##  Project Structure

```text
MedLens/
├── backend/            # API / inference logic
├── frontend/           # Frontend UI
├── training/           # Model training scripts
├── explainability/     # Model explainability & visualization
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Ignored files (venv, data, models, etc.)
```
---

##  Setup Instructions
##  Clone the repository
```bash
git clone https://github.com/ChanchalTaye/MedLens.git
cd MedLens
```
##  Create and activate virtual environment
```bash
python -m venv venv
```
##  Terminal
```bash
venv\Scripts\activate
```
##  Install dependencies
```bash
pip install -r requirements.txt
```
---

##  Datasets

MedLens uses multiple publicly available medical imaging datasets from **Kaggle**, covering different radiology modalities such as **X-ray, MRI, CT, and Ultrasound**.

**Datasets are NOT included in this repository** to keep it lightweight and compliant with GitHub storage limits.

---

###  Chest X-ray Datasets

#### 1. COVID-19 Radiography Database
- **Classes**: COVID-19, Normal, Viral Pneumonia
- **Link**: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

#### 2. Tuberculosis Chest X-rays (Montgomery)
- **Classes**: Tuberculosis, Normal
- **Link**: https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery

#### 3. Chest X-ray Pneumonia
- **Classes**: Pneumonia, Normal
- **Link**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

###  MRI / CT Dataset

#### 4. Kidney Stones MRI and CT Scans
- **Classes**: Kidney Stone, Normal
- **Modalities**: MRI, CT
- **Link**: https://www.kaggle.com/datasets/mohammedrizwanmalik/kidney-stones-mri-and-ct-scans

---

###  Ultrasound Dataset

#### 5. Breast Ultrasound Images Dataset
- **Classes**: Benign, Malignant, Normal
- **Link**: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

---

##  Downloading Datasets (Using Kaggle API)

###  Install Kaggle CLI
```bash
pip install kaggle
```
###  Download Datasets
```bash
kaggle datasets download tawsifurrahman/covid19-radiography-database -p data/covid19
kaggle datasets download raddar/tuberculosis-chest-xrays-montgomery -p data/tuberculosis
kaggle datasets download paultimothymooney/chest-xray-pneumonia -p data/pneumonia
kaggle datasets download mohammedrizwanmalik/kidney-stones-mri-and-ct-scans -p data/kidney
kaggle datasets download aryashah2k/breast-ultrasound-images-dataset -p data/breast_ultrasound
```
###  Recommended Dataset Structure
```bash
data/
├── covid19/
├── tuberculosis/
├── pneumonia/
├── kidney/
└── breast_ultrasound/
```











