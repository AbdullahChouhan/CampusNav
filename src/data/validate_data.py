"""Validate extracted frames, labels, duplicates, and recording counts."""

import hashlib
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import yaml


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["data"]
    with open("configs/locations.yaml", "r", encoding="utf-8") as f:
        valid_locations = set(yaml.safe_load(f)["locations"])

    manifest_path = Path(params["manifest_csv"])
    df = pd.read_csv(manifest_path)

    errors = []
    if df[["image_path", "location_label", "recording_id"]].isnull().any().any():
        errors.append("Manifest contains null values.")

    invalid_labels = set(df["location_label"]) - valid_locations
    if invalid_labels:
        errors.append(f"Invalid labels found: {sorted(invalid_labels)}")

    hashes = {}
    duplicate_count = 0
    for img_path_str in df["image_path"]:
        img_path = Path(img_path_str)
        if not img_path.exists():
            errors.append(f"Missing image file: {img_path}")
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as exc:
            errors.append(f"Corrupt image {img_path}: {exc}")
            continue

        digest = file_md5(img_path)
        if digest in hashes:
            duplicate_count += 1
        else:
            hashes[digest] = img_path_str

    min_rec = int(params.get("min_recordings_per_class", 3))
    rec_per_class = df.groupby("location_label")["recording_id"].nunique().to_dict()
    for loc in valid_locations:
        if rec_per_class.get(loc, 0) < min_rec:
            errors.append(
                f"Class '{loc}' has {rec_per_class.get(loc, 0)} recordings (min required: {min_rec})."
            )

    report = {
        "total_images": int(len(df)),
        "duplicate_images": duplicate_count,
        "class_counts": df["location_label"].value_counts().to_dict(),
        "recordings_per_class": rec_per_class,
        "errors": errors,
        "passed": len(errors) == 0,
    }

    report_path = Path("reports/metrics/validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not report["passed"]:
        raise ValueError(f"Data validation failed: {errors}")

    print(f"Validation passed ({len(df)} images). Report saved to {report_path}")


if __name__ == "__main__":
    main()
