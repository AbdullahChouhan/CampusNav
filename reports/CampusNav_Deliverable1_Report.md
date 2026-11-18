# CampusNav: Reproducible Computer Vision Pipeline for Indoor Location Recognition
**MLOps Deliverable 1 — System Architecture & Implementation Report**

- **GitHub Repository:** https://github.com/AbdullahChouhan/CampusNav
- **DagsHub Workspace:** https://dagshub.com/AbdullahChouhan/CampusNav
- **MLflow Tracking URI:** https://dagshub.com/AbdullahChouhan/CampusNav.mlflow
- **DVC Remote Endpoint:** https://dagshub.com/AbdullahChouhan/CampusNav.dvc

---

## 1. Project Overview and Scope
**CampusNav** is an AI-assisted indoor navigation system designed to localize users inside university buildings from hallway imagery and guide them between destinations. Rather than relying on computationally heavy 3D reconstruction, the system decouples indoor navigation into two modular components:
1. **Visual Global Localization:** A computer vision pipeline that classifies an input camera frame into a distinct topological building zone or landmark.
2. **Topological Route Finding:** A graph-based shortest-path engine that maps zones and room offsets across corridors.

For Deliverable 1, the implementation establishes the end-to-end reproducible MLOps pipeline for **indoor location recognition** on **Building F, Floor 3 (F3)**.

### Hierarchical Location Taxonomy
To ensure forward compatibility across multiple floors and buildings without refactoring schemas in later phases, location labels follow a strict hierarchical naming convention (`<building>_<floor>_<zone>`), defined in `configs/locations.yaml`:
- `f_f3_west_corridor` — West wing classroom corridor on Floor 3.
- `f_f3_east_corridor` — East wing classroom corridor on Floor 3.
- `f_f3_south_junction` — Southern connecting crossway and lobby junction.
- `f_f3_north_junction` — Northern connecting crossway and lobby junction.
- `f_f3_south_stairwell` — Southern stairwell landing and vertical access point.
- `f_f3_north_stairwell` — Northern stairwell landing and vertical access point.

---

## 2. Repository Architecture and MLOps Hygiene
The project enforces a strict separation between source code, configuration, credentials, and large binary artifacts:
```text
CampusNav/
├── configs/locations.yaml       # Target building, floor, and canonical class list
├── params.yaml                  # Centralized pipeline & model hyperparameters
├── dvc.yaml                     # Multi-stage reproducible DVC pipeline definition
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Template for DagsHub and MLflow environment variables
├── data/
│   ├── raw_videos/              # Raw hallway recordings (tracked via DVC)
│   ├── annotations/             # Timestamp interval annotations (video_intervals.csv)
│   ├── processed/               # Extracted 224x224 JPEG frames & dataset_manifest.csv
│   └── splits/                  # Leak-free train.csv, val.csv, and test.csv splits
├── src/
│   ├── data/                    # extract_frames.py, validate_data.py, split_data.py
│   ├── models/                  # train_svm.py (Baseline) & train_resnet.py (Main)
│   ├── evaluation/              # evaluate.py (Metrics JSON & confusion matrix plots)
│   └── tracking/                # mlflow_utils.py & register_model.py
├── models/                      # Serialized checkpoints (tracked via DVC)
├── reports/                     # Evaluation metrics and generated figures
└── tests/                       # Automated pytest suite for configs, models, and DAG
```

**Credential and Artifact Isolation:** `.gitignore` excludes `.env`, `.dvc/config.local`, `.venv/`, `mlruns/`, raw video files (`*.mp4`), extracted frame directories, and model weights (`*.pth`, `*.joblib`). DVC remote authentication is configured locally via `dvc remote modify --local` and backed by DagsHub storage linked to Hugging Face S3.

---

## 3. Data Engineering and Leak-Free Splitting Workflow

### 3.1 Interval Annotation and Frame Extraction (`src/data/extract_frames.py`)
Videos are annotated using timestamp intervals in `data/annotations/video_intervals.csv` (`video_file, start_time, end_time, location_label`) with 3–5 second guard bands at doorway transitions to exclude motion blur and ambiguous boundary frames.

The extraction stage reads `params.yaml`, opens each video via OpenCV (`cv2.VideoCapture`), samples frames every `frame_interval_sec: 2.0` seconds within valid annotated intervals, downscales frames immediately to `224x224` JPEG (`jpeg_quality: 90`), and outputs `data/processed/dataset_manifest.csv`.

### 3.2 Automated Data Validation (`src/data/validate_data.py`)
The validation stage verifies dataset integrity and outputs `reports/metrics/validation_report.json`:
1. **Schema & Null Check:** Ensures no missing values exist across `image_path`, `location_label`, or `recording_id`.
2. **Canonical Label Verification:** Asserts every label belongs to `configs/locations.yaml`.
3. **Image Integrity Check:** Verifies every extracted JPEG via `PIL.Image.verify()`.
4. **Duplicate Detection:** Computes MD5 digests across all frames to flag duplicate images.
5. **Recording Coverage Check:** Verifies each class appears across at least `min_recordings_per_class` independent recordings.

### 3.3 Preventing Data Leakage via Group Splitting (`src/data/split_data.py`)
To prevent frames from the same video from appearing across both training and evaluation splits, `split_data.py` uses 5-fold `StratifiedGroupKFold` grouped strictly by `recording_id` and stratified by `location_label` (60% train, 20% validation, 20% test), with explicit disjointness assertions across all three splits.

---

## 4. Machine Learning Models and Evaluation

### 4.1 Baseline Model: HOG + SVM (`src/models/train_svm.py`)
Converts images to grayscale, resizes to `128x128`, extracts Histogram of Oriented Gradients (HOG) descriptors (`orientations=9`, `pixels_per_cell=[16, 16]`, `cells_per_block=[2, 2]`), standardizes features via `StandardScaler`, and trains an RBF-kernel `SVC` pipeline saved to `models/svm_baseline.joblib`.

### 4.2 Main Model: Fine-Tuned ResNet-18 (`src/models/train_resnet.py`)
Uses a pretrained **ResNet-18** (`ResNet18_Weights.DEFAULT`) in PyTorch with `224x224` input resolution, training augmentations (`RandomHorizontalFlip`, `ColorJitter`, ImageNet normalization), frozen convolutional backbone (`freeze_backbone: true`), and a 6-class linear classification head trained with Cross-Entropy loss and Adam (`lr=0.001`, `batch_size=32`, `epochs=10`).

### 4.3 Decoupled Evaluation (`src/evaluation/evaluate.py`)
Computes test **Accuracy**, **Macro Precision**, **Macro Recall**, **Macro F1-score**, and full classification reports (`reports/metrics/<model>_metrics.json`), and saves labeled confusion matrix heatmaps (`reports/figures/<model>_confusion_matrix.png`).

---

## 5. DVC Pipeline and MLflow Tracking Integration

### 5.1 DVC Pipeline DAG (`dvc dag`)
```text
                                 +----------------+
                                 | extract_frames |
                                 +----------------+
                                          |
                                 +---------------+
                                 | validate_data |
                                 +---------------+
                                          |
                                   +------------+
                             ******| split_data |******
                       ******      +------------+      ******
                 ******                                      ******
        +----------------+                                  +------------+
        | train_baseline |                                  | train_main |
        +----------------+                                  +------------+
                 |                                                 |
       +-------------------+                              +---------------+
       | evaluate_baseline |                              | evaluate_main |
       +-------------------+                              +---------------+
                                                                   |
                                                          +----------------+
                                                          | register_model |
                                                          +----------------+
```

### 5.2 Remote MLflow Tracking & Model Registry (`src/tracking/`)
All runs log hyperparameters, metrics, model artifacts, and confusion matrices to the DagsHub MLflow server (`CampusNav-Indoor-Localization`). `register_model.py` gates registration behind `register_model` and `min_f1_to_register: 0.70` in `params.yaml`, registering `CampusNav-ResNet18` and applying both the `Staging` stage transition and modern `@staging` alias.
