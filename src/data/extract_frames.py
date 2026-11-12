"""Extract frames from raw videos using timestamp intervals and generate dataset_manifest.csv."""

from pathlib import Path
import cv2
import pandas as pd
import yaml


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

    manifest_rows = []
    for video_name, group in df_ann.groupby("video_file"):
        video_path = raw_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        recording_id = video_path.stem
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
    print(f"Extracted {len(manifest_df)} frames -> {manifest_out}")


if __name__ == "__main__":
    main()
