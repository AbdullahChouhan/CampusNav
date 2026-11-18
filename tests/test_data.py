"""Tests for configuration, face blurring, and nested video path resolution."""

from pathlib import Path
import numpy as np
from skimage import data as skdata
import yaml

from src.data.extract_frames import anonymize_faces, load_face_detector, resolve_video_path


def test_locations_config():
    with open("configs/locations.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["building"] == "F"
    assert cfg["floor"] == "F3"
    assert len(cfg["locations"]) == 6
    for loc in cfg["locations"]:
        assert loc.startswith("f_f3_")


def test_face_blurring_pipeline():
    detector = load_face_detector()
    sample_img = skdata.astronaut()
    blurred = anonymize_faces(sample_img, detector)
    assert blurred.shape == sample_img.shape
    assert not np.array_equal(blurred, sample_img)


def test_raw_videos_directory_structure():
    evening_dir = Path("data/raw_videos/f_block/3rd_floor/evening")
    assert evening_dir.exists() and evening_dir.is_dir()
    resolved = resolve_video_path(Path("data/raw_videos"), "f_block/3rd_floor/evening/.gitkeep")
    assert resolved == evening_dir / ".gitkeep"
