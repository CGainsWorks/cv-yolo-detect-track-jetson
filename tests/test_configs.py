"""The YAML files and the dataclass defaults say the same thing.

Two places to change a threshold is one place too many.  These tests make the
YAML authoritative for a reader and the dataclass authoritative for the code,
and fail if they ever disagree.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from edgetrack.data.synth import CLASSES, config_for_split
from edgetrack.track.bytetrack import ByteTrackConfig
from edgetrack.track.deepsort import DeepSortConfig

ROOT = Path(__file__).resolve().parent.parent


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_bytetrack_yaml_matches_the_dataclass():
    config = load("tracker_bytetrack.yaml")
    defaults = {f.name: f.default for f in fields(ByteTrackConfig)}
    assert set(config) == set(defaults), (
        f"only in yaml: {set(config) - set(defaults)}; "
        f"only in the dataclass: {set(defaults) - set(config)}"
    )
    for key, value in config.items():
        assert defaults[key] == value, f"{key}: yaml {value}, dataclass {defaults[key]}"


def test_deepsort_yaml_matches_the_dataclass():
    config = load("tracker_deepsort.yaml")
    appearance = config.pop("appearance")
    defaults = {f.name: f.default for f in fields(DeepSortConfig)}
    assert set(config) == set(defaults), (
        f"only in yaml: {set(config) - set(defaults)}; "
        f"only in the dataclass: {set(defaults) - set(config)}"
    )
    for key, value in config.items():
        assert defaults[key] == value, f"{key}: yaml {value}, dataclass {defaults[key]}"
    assert appearance["encoder"] in ("histogram", "reid")


def test_tracker_defaults_are_valid():
    ByteTrackConfig().validate()
    DeepSortConfig().validate()


def test_data_yaml_matches_the_generator():
    config = load("data.yaml")
    assert [config["classes"][i] for i in sorted(config["classes"])] == list(CLASSES)
    train = config_for_split("train", config["splits"]["train"]["seed_base"])
    assert train.width == config["image"]["width"]
    assert train.height == config["image"]["height"]
    assert train.palette == config["splits"]["train"]["palette"]
    assert train.min_label_visibility == config["min_label_visibility"]

    test = config_for_split("test", config["splits"]["test"]["seed_base"])
    assert test.palette == config["splits"]["test"]["palette"]
    assert test.noise_sigma == config["domain_gap"]["noise_sigma"]
    assert test.blur_scale == config["domain_gap"]["blur_scale"]
    assert list(test.jpeg_quality) == config["domain_gap"]["jpeg_quality"]
    assert test.n_frames == config["splits"]["test"]["frames_per_sequence"]


def test_model_yaml_matches_the_detector_defaults():
    from edgetrack.detect.base import DetectorConfig

    config = load("model.yaml")
    defaults = DetectorConfig()
    assert list(defaults.imgsz) == config["infer_imgsz"]
    assert defaults.conf_threshold == config["nms"]["conf_threshold"]
    assert defaults.iou_threshold == config["nms"]["iou_threshold"]
    assert defaults.max_detections == config["nms"]["max_detections"]
    assert defaults.agnostic_nms == config["nms"]["class_agnostic"]
    assert config["num_classes"] == len(CLASSES)


def test_export_yaml_agrees_with_model_yaml_on_the_input_size():
    assert load("export.yaml")["onnx"]["imgsz"] == load("model.yaml")["infer_imgsz"]


def test_export_opset_supports_per_channel_quantisation():
    """Per-channel QDQ needs the axis attribute, which arrived in opset 13.

    At opset 12 ``quantize_static(per_channel=True)`` emits a graph ONNX
    Runtime rejects as invalid, and it does so without warning at export time.
    """
    export = load("export.yaml")
    if export["quantisation"]["int8"]["per_channel"]:
        assert export["onnx"]["opset"] >= 13


def test_benchmark_variants_point_at_plausible_artefacts():
    variants = load("benchmark.yaml")["variants"]
    names = [v["name"] for v in variants]
    assert len(names) == len(set(names)), "duplicate variant names"
    for variant in variants:
        assert variant["backend"] in ("onnx", "torch", "tensorrt")
        suffix = Path(variant["model"]).suffix
        expected = {"onnx": ".onnx", "torch": ".pt", "tensorrt": ".engine"}
        assert suffix == expected[variant["backend"]], (
            f"{variant['name']}: backend {variant['backend']} with a {suffix} file"
        )


def test_train_yaml_keeps_colour_augmentation_low():
    """Instance colour is DeepSORT's only cue on this dataset.

    An aggressive HSV augmentation teaches the detector to ignore the signal
    the appearance branch depends on, so the schedule deliberately turns it
    down.  If somebody raises it, this test is where they find out why not.
    """
    train = load("train.yaml")["train"]
    assert train["hsv_s"] <= 0.4
    assert train["hsv_v"] <= 0.4
    assert train["hsv_h"] <= 0.02


@pytest.mark.parametrize("name", [
    "data.yaml", "model.yaml", "train.yaml", "export.yaml", "benchmark.yaml",
    "tracker_bytetrack.yaml", "tracker_deepsort.yaml",
])
def test_every_config_parses(name):
    assert isinstance(load(name), dict)
