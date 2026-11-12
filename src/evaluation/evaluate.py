"""Evaluate baseline or main model on the test split and log metrics/plots."""

import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import yaml
import mlflow

from src.models.train_resnet import FrameDataset
from src.models.train_svm import extract_hog_features
from src.tracking.mlflow_utils import configure_mlflow


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "main"], required=True)
    args = parser.parse_args()

    with open("params.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_df = pd.read_csv("data/splits/test.csv")
    y_true_labels = test_df["location_label"].tolist()

    if args.model == "baseline":
        bundle = joblib.load("models/svm_baseline.joblib")
        clf = bundle["pipeline"]
        le = bundle["label_encoder"]
        svm_cfg = cfg["baseline_svm"]
        X_test = extract_hog_features(
            test_df["image_path"].tolist(), tuple(svm_cfg["hog_resize"]), svm_cfg
        )
        y_pred_idx = clf.predict(X_test)
        y_pred_labels = le.inverse_transform(y_pred_idx).tolist()
        classes = sorted(le.classes_.tolist())
    else:
        ckpt = torch.load("models/resnet18_main.pth", map_location="cpu")
        class_to_idx = ckpt["class_to_idx"]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        classes = sorted(class_to_idx.keys())
        img_size = tuple(ckpt["img_size"])

        tf = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        ds = FrameDataset(test_df, class_to_idx, tf)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        pred_indices = []
        with torch.no_grad():
            for x, _ in loader:
                out = model(x.to(device))
                pred_indices.extend(out.argmax(dim=1).cpu().tolist())
        y_pred_labels = [idx_to_class[i] for i in pred_indices]

    metrics = {
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "macro_precision": float(
            precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true_labels, y_pred_labels, output_dict=True, zero_division=0
        ),
    }

    metrics_path = Path(f"reports/metrics/{args.model}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = Path(f"reports/figures/{args.model}_confusion_matrix.png")
    save_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        classes,
        cm_path,
        title=f"Confusion Matrix ({args.model})",
    )

    configure_mlflow(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"evaluate_{args.model}"):
        mlflow.log_metrics(
            {
                "test_accuracy": metrics["accuracy"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
                "test_macro_f1": metrics["macro_f1"],
            }
        )
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(cm_path), artifact_path="reports")

    print(
        f"[{args.model}] test_acc={metrics['accuracy']:.4f}, "
        f"test_macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
