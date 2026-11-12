"""Fine-tune pretrained ResNet-18 classifier and log to MLflow."""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score
import yaml
import mlflow

from src.tracking.mlflow_utils import configure_mlflow


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.paths = df["image_path"].tolist()
        self.labels = [class_to_idx[lbl] for lbl in df["location_label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    set_seed(seed)
    res_cfg = cfg["main_resnet"]
    img_size = tuple(cfg["data"]["image_size"])

    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")

    classes = sorted(train_df["location_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_tf = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = FrameDataset(train_df, class_to_idx, train_tf)
    val_ds = FrameDataset(val_df, class_to_idx, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(res_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(res_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(res_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(res_cfg.get("num_workers", 0)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT if res_cfg.get("pretrained", True) else None
    model = models.resnet18(weights=weights)

    if res_cfg.get("freeze_backbone", True):
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(res_cfg["learning_rate"]),
    )

    configure_mlflow(cfg["mlflow"]["experiment_name"])
    best_val_f1 = -1.0
    out_path = Path("models/resnet18_main.pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="main_resnet18"):
        mlflow.log_params(res_cfg)
        mlflow.log_param("seed", seed)
        mlflow.log_param("device", str(device))

        for epoch in range(1, int(res_cfg["epochs"]) + 1):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            train_loss = running_loss / len(train_ds)

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    out = model(x)
                    preds.extend(out.argmax(dim=1).cpu().tolist())
                    targets.extend(y.tolist())

            val_acc = accuracy_score(targets, preds)
            val_f1 = f1_score(targets, preds, average="macro")
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1},
                step=epoch,
            )
            print(
                f"Epoch {epoch}/{res_cfg['epochs']} - "
                f"loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "img_size": img_size,
                    },
                    out_path,
                )

        mlflow.log_artifact(str(out_path), artifact_path="model")


if __name__ == "__main__":
    main()
