"""Extract frames from raw videos using timestamp intervals, blur faces, and generate dataset_manifest.csv."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage import data as skdata
from skimage.feature import Cascade
import yaml


def load_face_detector() -> Cascade:
    """Load built-in LBP frontal face cascade detector from scikit-image."""
    return Cascade(skdata.lbp_frontal_face_cascade_filename())


def anonymize_faces(frame: np.ndarray, detector: Cascade | None) -> np.ndarray:
    """Detect and apply strong Gaussian blur to faces on the full-resolution frame."""
    if detector is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h_img, w_img = gray.shape[:2]
    max_dim = max(40, min(h_img, w_img) // 2)

    detected = detector.detect_multi_scale(
        img=gray,
        scale_factor=1.2,
        step_ratio=1,
        min_size=(24, 24),
        max_size=(max_dim, max_dim),
    )
    if not detected:
        return frame

    out = frame.copy()
    for patch in detected:
        y = max(0, int(patch["r"]))
        x = max(0, int(patch["c"]))
        h = int(patch["height"])
        w = int(patch["width"])
        roi = out[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        kW = max(15, (w // 3) | 1)
        kH = max(15, (h // 3) | 1)
        out[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (kW, kH), 30)
    return out


def resolve_video_path(raw_dir: Path, video_name: str) -> Path:
    """Resolve video path directly or recursively inside raw_dir subdirectories."""
    direct_path = raw_dir / video_name
    if direct_path.exists():
        return direct_path

    matches = list(raw_dir.rglob(video_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple videos named '{video_name}' found under {raw_dir}: {matches}. "
            "Please specify the relative path in video_intervals.csv."
        )
    raise FileNotFoundError(f"Video file not found under {raw_dir}: {video_name}")


def main() -> None:
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["data"]
    with open("configs/locations.yaml", "r", encoding="utf-8") as f:
        valid_locations = set(yaml.safe_load(f)["locations"])

    annotations_path = Path(params["annotations_csv"])
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations CSV not found at {annotations_path}. "
            "Export your timestamp annotations first."
        )

    df_ann = pd.read_csv(annotations_path)
    frames_dir = Path(params["frames_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(params["raw_videos_dir"])

    interval_sec = float(params["frame_interval_sec"])
    width, height = params["image_size"]
    jpeg_quality = int(params.get("jpeg_quality", 90))
    blur_faces = bool(params.get("blur_faces", True))
    detector = load_face_detector() if blur_faces else None

    manifest_rows = []
    for video_name, group in df_ann.groupby("video_file"):
        video_path = resolve_video_path(raw_dir, str(video_name))
        recording_id = (
            video_path.relative_to(raw_dir).with_suffix("").as_posix().replace("/", "_")
        )

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        t = 0.0
        while t <= duration_sec:
            matched = group[(group["start_time"] <= t) & (group["end_time"] >= t)]
            if not matched.empty:
                label = str(matched.iloc[0]["location_label"])
                if label not in valid_locations:
                    raise ValueError(f"Invalid location label '{label}' in {video_name}")

                frame_idx = int(round(t * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    if blur_faces:
                        frame = anonymize_faces(frame, detector)
                    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    out_dir = frames_dir / recording_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"frame_{int(round(t)):04d}.jpg"
                    cv2.imwrite(
                        str(out_file),
                        resized,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                    manifest_rows.append(
                        {
                            "image_path": out_file.as_posix(),
                            "location_label": label,
                            "recording_id": recording_id,
                            "timestamp_sec": round(t, 2),
                        }
                    )
            t += interval_sec

        cap.release()

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_out = Path(params["manifest_csv"])
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_out, index=False)
    print(f"Extracted {len(manifest_df)} frames (blur_faces={blur_faces}) -> {manifest_out}")


if __name__ == "__main__":
    main()
