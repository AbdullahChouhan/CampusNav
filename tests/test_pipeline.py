"""Tests for DVC pipeline and parameter files."""

import yaml


def test_params_and_dvc_yaml():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open("dvc.yaml", "r", encoding="utf-8") as f:
        dvc = yaml.safe_load(f)

    assert "data" in params
    assert "baseline_svm" in params
    assert "main_resnet" in params
    assert "stages" in dvc
    assert "extract_frames" in dvc["stages"]
    assert "register_model" in dvc["stages"]
