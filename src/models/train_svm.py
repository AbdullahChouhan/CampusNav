"""Train baseline HOG + SVM classifier and log to MLflow."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import yaml
import mlflow

from src.tracking.mlflow_utils import configure_mlflow


def extract_hog_features(image_paths: list[str], resize: tuple[int, int], svm_cfg: dict) -> np.ndarray:
    features = []
    for p in image_paths:
        img = Image.open(p).convert("L").resize(resize)
        feat = hog(
            np.array(img),
            orientations=int(svm_cfg["hog_orientations"]),
            pixels_per_cell=tuple(svm_cfg["hog_pixels_per_cell"]),
            cells_per_block=tuple(svm_cfg["hog_cells_per_block"]),
            feature_vector=True,
        )
        features.append(feat)
    return np.array(features)


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    svm_cfg = cfg["baseline_svm"]
    seed = int(cfg["seed"])
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")

    resize = tuple(svm_cfg["hog_resize"])
    X_train = extract_hog_features(train_df["image_path"].tolist(), resize, svm_cfg)
    X_val = extract_hog_features(val_df["image_path"].tolist(), resize, svm_cfg)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["location_label"])
    y_val = le.transform(val_df["location_label"])

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=float(svm_cfg["svm_c"]),
                    kernel=svm_cfg["svm_kernel"],
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )

    configure_mlflow(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="baseline_hog_svm"):
        mlflow.log_params(svm_cfg)
        mlflow.log_param("seed", seed)

        clf.fit(X_train, y_train)
        val_preds = clf.predict(X_val)

        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        mlflow.log_metrics({"val_accuracy": val_acc, "val_macro_f1": val_f1})

        out_path = Path("models/svm_baseline.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": clf, "label_encoder": le}, out_path)
        mlflow.log_artifact(str(out_path), artifact_path="model")

        print(f"Baseline HOG+SVM -> val_acc={val_acc:.4f}, val_macro_f1={val_f1:.4f}")


if __name__ == "__main__":
    main()
