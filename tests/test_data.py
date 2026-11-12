"""Tests for configuration and data schema."""

import yaml


def test_locations_config():
    with open("configs/locations.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["building"] == "F"
    assert cfg["floor"] == "F3"
    assert len(cfg["locations"]) == 6
    for loc in cfg["locations"]:
        assert loc.startswith("f_f3_")
