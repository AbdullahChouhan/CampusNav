# Deliverable 1 — Master Checklist

## 1. Data Collection & Annotation (Morning Recordings)
- [ ] Record ~20–25 unidirectional videos in **Building F, Floor 3 (F3)** across all 6 target locations:
  - `f_f3_west_corridor`
  - `f_f3_east_corridor`
  - `f_f3_south_junction`
  - `f_f3_north_junction`
  - `f_f3_south_stairwell`
  - `f_f3_north_stairwell`
- [ ] Ensure multiple group members record independent sweeps/clips (different times/directions, no 180° double-backing mid-video).
- [ ] Place `.mp4` files in `data/raw_videos/` and track them with DVC (`dvc add data/raw_videos`).
- [ ] Annotate video timestamp intervals in Google Sheets (leaving 3–5s guard bands at transitions) and export to `data/annotations/video_intervals.csv`.
- [ ] Create/finalize `configs/map_metadata.yaml` for room-to-zone mapping (for Phase 2 readiness).

## 2. Pipeline Implementation
- [ ] Implement `src/data/extract_frames.py` (OpenCV sampling every 2s, resize to 224x224, output `dataset_manifest.csv`).
- [ ] Implement `src/data/validate_data.py` (file existence, corrupt image check, label validation, duplicate check, class/recording count check).
- [ ] Implement `src/data/split_data.py` (`StratifiedGroupKFold` grouped by `recording_id` to guarantee zero data leakage).
- [ ] Implement `src/models/train_svm.py` (HOG feature extraction + scikit-learn SVM + MLflow logging).
- [ ] Implement `src/models/train_resnet.py` (Pretrained ResNet-18 fine-tuning in PyTorch + MLflow logging).
- [ ] Implement `src/evaluation/evaluate.py` (Accuracy, Precision, Recall, Macro F1, Confusion Matrix PNG, Classification Report).
- [ ] Implement `src/tracking/register_model.py` (Register best model to MLflow & assign `Staging` stage / `@staging` alias).
- [ ] Add unit & integration tests in `tests/`.

## 3. MLOps & Reproducibility Verification
- [ ] Run full pipeline end-to-end with `dvc repro`.
- [ ] Verify `dvc dag` displays the clean parallel pipeline graph.
- [ ] Push DVC-tracked data and models (`dvc push`).
- [ ] Verify MLflow experiment runs, hyperparameters, metrics, and artifacts appear on DagsHub.
- [ ] Verify registered model appears in `Staging` (or `@staging` alias) on DagsHub MLflow.
- [ ] Test a clean `git clone` + `dvc pull` + `dvc repro` on another machine/folder to confirm reproducibility.
- [ ] Ensure all 3 group members have meaningful commits on GitHub.

## 4. Final PDF Report (3–4 Pages)
- [ ] Include GitHub repository link.
- [ ] Include DagsHub / MLflow tracking link.
- [ ] Include screenshots of at least two MLflow experiment runs (HOG+SVM vs ResNet-18).
- [ ] Include screenshot of the registered model in `Staging`.
- [ ] Include screenshot of `dvc dag`.
- [ ] Write concise explanation of dataset design, leak-free splitting, pipeline stages, and model comparison.
