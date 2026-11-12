# CampusNav — Project Context and Current Plan

I'm working on a university MLOps project called **CampusNav**, an AI-assisted indoor navigation system. Below is the project background, assignment requirements, decisions made so far, and the proposed implementation.

Please use this as context for future questions. Treat the decisions and assignment requirements below as the current plan unless I explicitly change them.

## 1. Project idea

CampusNav is intended to help users navigate inside university buildings.

The original idea involved:
- Recording videos of university hallways.
- Potentially reconstructing 3D models of the building.
- Using AI to recognize locations.
- Eventually finding routes between rooms and locations.

However, we decided to simplify the initial implementation.

**Current technical approach:**
1. A computer vision model recognizes the user's current indoor location from an image.
2. A navigation graph represents connections between locations.
3. A graph algorithm such as Dijkstra's algorithm finds a route between two locations.

For the first deliverable, we will focus on **indoor location recognition**, not the complete navigation system.

We do not need 3D reconstruction.

## 2. Scope of the first deliverable

The first deliverable is primarily about demonstrating MLOps practices and building a reproducible machine learning pipeline.

The proposed project title is:

**"CampusNav: Reproducible Computer Vision Pipeline for Indoor Location Recognition"**

The initial system will:
- Focus on one university building.
- Recognize approximately 5–8 distinct indoor locations.
- Accept an image extracted from a hallway video.
- Predict the location represented in that image.

Example output:

Input image → Predicted location: `Computer_Lab_Entrance`

The navigation graph and actual route-finding functionality can be developed in a later phase.

## 3. Assignment requirements

The first deliverable requires:

### GitHub
- A GitHub repository with a clean project structure.
- A useful README.
- All group members must contribute commits.
- A `.gitignore` file.
- Do not commit raw datasets, model weights, or secrets directly to Git.

### Data management
- Data ingestion.
- Data validation.
- Data versioning using DVC.
- A DVC remote, preferably through DagsHub or AWS S3.
- A reproducible data-processing workflow.

### Machine learning
Implement and evaluate:
1. A baseline model.
2. A main model.

Both models must be evaluated using the same train/validation/test split.

### MLflow
- Use remote MLflow tracking, preferably through DagsHub.
- Log hyperparameters, metrics, and artifacts for every experiment run.
- Register at least one model/checkpoint.
- The registered model should have the requested `Staging` or `Production` status.

Note: If the MLflow version or interface no longer supports traditional model stages, check with the lecturer whether an equivalent alias or status is acceptable.

### DVC pipeline
- Create a `dvc.yaml` containing named pipeline stages.
- The complete pipeline must be reproducible using `dvc repro`.
- The pipeline structure must be visible using `dvc dag`.

### Final PDF
Submit a 3–4 page PDF containing:
- GitHub repository link.
- DagsHub/MLflow link.
- Screenshots of at least two MLflow experiment runs.
- Screenshot of the registered model in Staging.
- Screenshot of the DVC pipeline using `dvc dag`.
- A short explanation of the implementation.

**Important:** Docker, cloud deployment, REST APIs, Flask, EC2, and monitoring are not required for this first deliverable. Do not expand the scope to include them unless the assignment requirements change.

## 4. Dataset collection strategy

We plan to record videos of university corridors and recognizable indoor locations.

The recommended approach is a **hybrid recording strategy**, rather than recording only complete floor walkthroughs or only random short clips.

Suggested distribution:

- 40% full corridor sweeps.
- 35% random partial walks.
- 25% location-focused recordings.

These percentages are guidelines, not strict requirements.

### Recording types

**A. Full corridor sweeps**
- Walk through a complete corridor or connected section.
- Suggested duration: 1–3 minutes.
- Record the different locations encountered along the way.

**B. Random partial walks**
- Start at different points and walk through part of a corridor.
- Vary starting and ending points.
- Suggested duration: 20–60 seconds.

**C. Location-focused recordings**
- Record recognizable areas such as laboratory entrances, staircases, corridor junctions, and library entrances.
- Capture different viewing angles.
- Suggested duration: 20–40 seconds.

### Recording settings

Recommended initial settings:
- 1080p resolution.
- Landscape orientation.
- 30 FPS.
- Camera held approximately at eye level.
- Normal walking speed.
- Avoid excessive camera shaking.

Vary:
- Walking direction.
- Starting and ending points.
- Camera angles.
- Recording sessions.
- People recording the videos.
- Lighting conditions and times of day where practical.

Avoid recording identifiable people where possible. Obtain university permission and blur faces if necessary.

### Dataset size targets

Initial practical targets:
- 5–8 location classes.
- 3–5 independent recording sessions per location.
- Approximately 20–35 total videos.
- Approximately 1,000–2,000 extracted frames.
- Approximately 150–300 frames per location.

These are starting targets, not strict requirements. Independent recordings and useful visual variation are more important than simply having many nearly identical frames.

Every group member should record multiple locations. Avoid having one person record only one location, as the model might learn differences in camera or recording style instead of the actual location.

## 5. How video labeling works

An important decision is that **we should not assign one location label to an entire video if that video passes through multiple locations.**

Instead, annotate videos using timestamp intervals.

For example, suppose `sweep_01.mp4` contains:

| Start time | End time | Location |
|---|---|---|
| 0 sec | 30 sec | Main_Corridor |
| 30 sec | 55 sec | Computer_Lab_Entrance |
| 55 sec | 90 sec | Classroom_Corridor |
| 90 sec | 120 sec | Staircase |

We can store these annotations in a separate CSV:

```csv
video,start_time,end_time,location
sweep_01.mp4,0,30,Main_Corridor
sweep_01.mp4,30,55,Computer_Lab_Entrance
sweep_01.mp4,55,90,Classroom_Corridor
sweep_01.mp4,90,120,Staircase
```

The frame extraction script uses these timestamp intervals to determine the correct label for each extracted image.

Avoid extracting or labeling ambiguous frames at transitions between locations. Initially, it is better to exclude these frames rather than introduce a `Transition` class.

Location classes should represent visually distinguishable areas. For example, a long corridor might need to be divided into multiple zones if different sections can be recognized reliably.

## 6. How frame extraction and the image CSV work

**Each extracted image becomes one individual training example and one row in the image dataset CSV.**

We do not extract only:
- The first frame of a location.
- The last frame of a location.
- The midpoint frame.

Instead, extract multiple frames while the location is visible.

Recommended starting extraction frequency:

**One frame every 2 seconds.**

We do not need every frame from the original 30 FPS video because consecutive frames are often almost identical and would create unnecessary redundancy.

For example, if `Main_Corridor` appears from 0–30 seconds, extract frames at approximately:

0, 2, 4, 6, 8, 10, ... 28 seconds.

Each of these images receives the `Main_Corridor` label.

The main image dataset CSV should have this structure:

```csv
image_path,location_label,recording_id
data/frames/sweep_01/frame_000.jpg,Main_Corridor,sweep_01
data/frames/sweep_01/frame_002.jpg,Main_Corridor,sweep_01
data/frames/sweep_01/frame_004.jpg,Main_Corridor,sweep_01
data/frames/sweep_01/frame_006.jpg,Main_Corridor,sweep_01
data/frames/sweep_01/frame_008.jpg,Main_Corridor,sweep_01
```

Explanation:

- `image_path`: Path to the individual extracted image.
- `location_label`: Correct location class for that image.
- `recording_id`: Identifies the original video or recording session.

The timestamp annotation CSV and image dataset CSV serve different purposes:

- Timestamp CSV: Describes which location is visible during each video interval.
- Image CSV: Describes each extracted image and its assigned label.

The extraction pipeline should automatically generate the image CSV from the timestamp annotations and extracted frames.

## 7. Avoiding data leakage

This is particularly important because multiple frames from the same video are visually very similar.

**Never randomly distribute individual frames from the same original recording across training, validation, and testing.**

For example, if 100 frames come from `sweep_01.mp4`, all 100 must belong to the same split.

Otherwise, the model might see almost identical images during training and testing, making the test performance misleadingly high.

Instead:
- Split by original recording ID or independent recording session.
- Keep recordings together within one split.
- Ensure every location class appears in training, validation, and testing.
- Where possible, distribute recording sessions from different people across the splits to avoid camera/person bias.

Use the same split for both the baseline and main model.

## 8. Proposed machine learning models

We plan to compare two approaches.

### Baseline: HOG + SVM
- Extract Histogram of Oriented Gradients (HOG) features.
- Train a Support Vector Machine classifier.
- Use scikit-learn.

This provides a relatively simple traditional computer vision baseline.

### Main model: ResNet-18
- Use a pretrained ResNet-18.
- Fine-tune it on our indoor location dataset.
- Use PyTorch.

Both models should use identical dataset splits.

### Evaluation metrics

Log and compare:
- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Confusion matrix.
- Classification report.

The purpose is to demonstrate a meaningful model comparison and reproducible MLOps workflow, not simply to maximize accuracy.

## 9. Proposed MLOps pipeline

The pipeline should contain named stages similar to:

1. `build_manifest`: Build the dataset manifest from video annotations.
2. `extract_frames`: Extract images from videos at the chosen interval.
3. `validate_data`: Check image integrity, labels, duplicates, and class distribution.
4. `split_data`: Create train/validation/test splits grouped by recording ID.
5. `train_baseline`: Train HOG + SVM.
6. `train_main`: Train ResNet-18.
7. `evaluate`: Evaluate both models and save metrics/artifacts.
8. `register_model`: Register the selected model/checkpoint in MLflow.

The pipeline should be reproducible through `dvc repro`.

Use `params.yaml` for configurable values such as:
- Frame extraction interval.
- Image dimensions.
- Dataset split ratios.
- Random seed.
- Model hyperparameters.
- Batch size.
- Learning rate.
- Number of epochs.

Data validation should check:
- Files exist and can be opened.
- Labels belong to the predefined location classes.
- No missing labels or paths.
- Images have valid dimensions and channels.
- Duplicate images are detected.
- Every class has sufficient recordings.
- Every class appears in each dataset split.

## 10. Suggested project structure

```text
CampusNav/
│
├── README.md
├── requirements.txt
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
│
├── configs/
│   └── locations.yaml
│
├── data/
│   ├── raw_videos/
│   ├── processed/
│   ├── annotations/
│   └── splits/
│
├── src/
│   ├── data/
│   │   ├── build_manifest.py
│   │   ├── extract_frames.py
│   │   ├── validate_data.py
│   │   └── split_data.py
│   │
│   ├── models/
│   │   ├── train_svm.py
│   │   └── train_resnet.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   └── tracking/
│       └── mlflow_utils.py
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_pipeline.py
│
└── reports/
    └── metrics/
```

The actual data directories can be managed using DVC rather than committing large datasets to Git.

The `.gitignore` should exclude things such as:
- `.venv/`
- `__pycache__/`
- `.env`
- `.dvc/config.local`
- `mlruns/`

Do not commit credentials or secrets.

## 11. Remote services

We are considering **DagsHub** because it can provide:
- GitHub repository integration.
- DVC remote storage.
- Remote MLflow tracking.

The hosted MLflow tracking URL generally follows this pattern:

```text
https://dagshub.com/USERNAME/REPOSITORY.mlflow
```

All group members should have the necessary repository permissions and credentials.

The project should be tested from a clean clone to verify that the DVC remote and `dvc repro` work correctly.

## 12. Suggested division of work for three group members

### Member 1 — Data Engineering
Responsible for:
- Recording videos.
- Defining location labels.
- Creating timestamp annotations.
- Extracting frames using OpenCV.
- Generating the image CSV.
- Data validation.
- DVC data tracking.

Possible files:
- `src/data/build_manifest.py`
- `src/data/extract_frames.py`
- `src/data/validate_data.py`
- `src/data/split_data.py`
- `configs/locations.yaml`

### Member 2 — Machine Learning
Responsible for:
- HOG + SVM baseline.
- ResNet-18 training.
- Evaluation.
- Metrics and confusion matrices.
- MLflow experiment logging.

Possible files:
- `src/models/train_svm.py`
- `src/models/train_resnet.py`
- `src/evaluation/evaluate.py`
- `src/tracking/mlflow_utils.py`

### Member 3 — MLOps Integration
Responsible for:
- GitHub and DagsHub setup.
- DVC remote configuration.
- MLflow remote tracking configuration.
- `dvc.yaml`.
- `params.yaml`.
- README.
- `.gitignore`.
- Integration testing.
- Model registration.
- Final screenshots and documentation.

All three members should contribute meaningful commits using their own GitHub accounts. Agree on the CSV schema and script interfaces before implementing separate components.

## 13. What comes after the first deliverable?

The longer-term CampusNav system can eventually combine:

1. Image-based indoor location recognition.
2. A manually constructed indoor navigation graph.
3. Dijkstra's shortest-path algorithm or NetworkX.
4. A user interface that displays the predicted current location and route to a destination.

For example:

```text
Main Entrance
      |
Main Corridor
   /      \
Library   Staircase
              |
       First Floor Corridor
          /         \
     Classroom    Laboratory
```

We can maintain a simple spreadsheet of which locations connect to each other while recording the videos.

However, implementing graph-based navigation is not necessary for the current first deliverable.

---

## Current overall recommendation

The immediate priorities are:

1. Choose 5–8 visually distinguishable locations in one building.
2. Agree on their exact class names.
3. Record a mixture of full corridor sweeps, partial walks, and focused location clips.
4. Create timestamp annotations for each video.
5. Extract one frame approximately every 2 seconds within each annotated location interval.
6. Automatically generate one CSV row per extracted image.
7. Split the data by recording session, not individual image.
8. Build the HOG + SVM and ResNet-18 models.
9. Integrate DVC, MLflow, and DagsHub.
10. Verify the complete pipeline using `dvc repro`.

**When helping me with this project, prioritize practical implementation advice, reproducibility, and the stated first-deliverable requirements. Avoid unnecessarily expanding the scope or introducing deployment features that are not required yet.**