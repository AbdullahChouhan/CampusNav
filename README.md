# CampusNav: Reproducible Computer Vision Pipeline for Indoor Location Recognition

An MLOps-driven indoor location recognition system for university buildings, starting with **Building F, Floor 3 (F3)**.

- **GitHub Repository:** https://github.com/AbdullahChouhan/CampusNav
- **DagsHub / MLflow Tracking:** https://dagshub.com/AbdullahChouhan/CampusNav

## Target Locations (Building F, Floor F3)
- `f_f3_west_corridor`
- `f_f3_east_corridor`
- `f_f3_south_junction`
- `f_f3_north_junction`
- `f_f3_south_stairwell`
- `f_f3_north_stairwell`

## Project Structure
```text
CampusNav/
├── configs/locations.yaml   # Building, floor, and target class labels
├── params.yaml              # Pipeline & model hyperparameters
├── dvc.yaml                 # Reproducible DVC pipeline DAG
├── data/
│   ├── raw_videos/          # Recorded hallway clips (DVC tracked)
│   ├── annotations/         # Timestamp interval CSV
│   ├── processed/           # Extracted 224x224 frames & manifest
│   └── splits/              # Leak-free train/val/test CSVs
├── src/
│   ├── data/                # Frame extraction, validation, group splitting
│   ├── models/              # Baseline (HOG+SVM) & Main (ResNet-18)
│   ├── evaluation/          # Metrics & confusion matrices
│   └── tracking/            # MLflow & DagsHub helpers + model registration
├── models/                  # Trained model checkpoints (DVC tracked)
├── reports/                 # Metrics JSONs & confusion matrix figures
└── tests/                   # Unit & pipeline sanity tests
```

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and insert your DagsHub token:
```bash
cp .env.example .env
```
Configure local DVC authentication (never committed to Git):
```bash
dvc remote modify --local origin auth basic
dvc remote modify --local origin user <YOUR_DAGSHUB_USERNAME>
dvc remote modify --local origin password <YOUR_DAGSHUB_TOKEN>
```

### 3. Reproduce the Pipeline
```bash
dvc pull
dvc repro
dvc dag
```
