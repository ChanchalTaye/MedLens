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
