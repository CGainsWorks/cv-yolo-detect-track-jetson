"""Tests that need the real exported model.

Marked ``needs_model`` and skipped when the artefacts are absent, so a fresh
checkout still has a green suite before ``make model`` has been run.  CI runs
them in the ``detector`` job, which builds the exports first.

The claim these defend is the one the whole results table rests on: the
precision variants differ in arithmetic and in nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgetrack.data.synth import CLASSES, config_for_split, generate_sequence
from edgetrack.detect import DetectorConfig, build_detector
from edgetrack.track.matching import iou_matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FP32 = MODELS / "yolov8n_edgetrack_fp32.onnx"
FP16 = MODELS / "yolov8n_edgetrack_fp16.onnx"
INT8 = MODELS / "yolov8n_edgetrack_int8.onnx"
CHECKPOINT = MODELS / "yolov8n_edgetrack.pt"

pytestmark = pytest.mark.needs_model


def config(conf: float = 0.25) -> DetectorConfig:
    return DetectorConfig(imgsz=(640, 384), conf_threshold=conf, class_names=CLASSES)


@pytest.fixture(scope="module")
def frames() -> list[np.ndarray]:
    cfg = config_for_split("test", 20000, n_frames=8)
    return [frame.image for frame in generate_sequence(cfg)]


def require(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not built; run `make export`")


def detections_for(path: Path, frames: list[np.ndarray], conf: float = 0.25):
    detector = build_detector(path, "onnx", config(conf), providers=["CPUExecutionProvider"])
    detector.warmup(2)
    out = [detector.detect(frame) for frame in frames]
    info = detector.info
    detector.close()
    return out, info


def test_fp32_detects_the_planted_objects(frames):
    require(FP32)
    detections, info = detections_for(FP32, frames)
    assert info.precision == "fp32"
    counts = [len(frame) for frame in detections]
    assert min(counts) >= 3, f"per-frame detection counts {counts}"
    assert set(d.class_id for frame in detections for d in frame) <= set(range(len(CLASSES)))


def test_precision_is_read_from_the_artefact_not_the_filename(tmp_path):
    """Renaming a file must not change what the results table calls it."""
    require(INT8)
    disguised = tmp_path / "totally_fp32.onnx"
    disguised.write_bytes(INT8.read_bytes())
    detector = build_detector(disguised, "onnx", config(), providers=["CPUExecutionProvider"])
    assert detector.info.precision == "int8"
    detector.close()


def test_fp16_input_dtype_is_half(frames):
    require(FP16)
    _, info = detections_for(FP16, frames)
    assert info.precision == "fp16"
    assert info.as_dict()["input_dtype"] == "float16"


def test_fp16_agrees_with_fp32_on_almost_every_box(frames):
    """Half precision must be a small numeric perturbation, not a new model."""
    require(FP32)
    require(FP16)
    reference, _ = detections_for(FP32, frames)
    half, _ = detections_for(FP16, frames)

    matched = total = 0
    for a, b in zip(reference, half):
        total += len(a)
        if not a or not b:
            continue
        ious = iou_matrix(
            np.array([d.xyxy for d in a]), np.array([d.xyxy for d in b])
        )
        matched += int((ious.max(axis=1) > 0.9).sum())
    assert total > 0
    assert matched / total > 0.9, f"only {matched}/{total} FP32 boxes survived in FP16"


def test_int8_agrees_with_fp32_on_most_boxes(frames):
    """INT8 is allowed to be worse.  It is not allowed to be a different model.

    The bar is deliberately lower than for FP16 and deliberately not zero: an
    INT8 graph whose head is quantised badly produces no usable detections at
    all, which is what the first attempt at this export did.
    """
    require(FP32)
    require(INT8)
    reference, _ = detections_for(FP32, frames)
    quantised, _ = detections_for(INT8, frames)

    matched = total = 0
    for a, b in zip(reference, quantised):
        total += len(a)
        if not a or not b:
            continue
        ious = iou_matrix(np.array([d.xyxy for d in a]), np.array([d.xyxy for d in b]))
        matched += int((ious.max(axis=1) > 0.7).sum())
    assert total > 0
    assert matched / total > 0.7, f"only {matched}/{total} FP32 boxes survived in INT8"


def test_detector_reports_per_stage_timings(frames):
    require(FP32)
    detector = build_detector(FP32, "onnx", config(), providers=["CPUExecutionProvider"])
    detector.warmup(2)
    detector.detect(frames[0])
    assert detector.last_inference_ms > 0
    assert detector.last_preprocess_ms >= 0
    assert detector.last_postprocess_ms >= 0
    detector.close()


def test_a_size_mismatch_is_caught_at_construction():
    require(FP32)
    with pytest.raises(ValueError, match="expects"):
        build_detector(FP32, "onnx", DetectorConfig(imgsz=(320, 320)))


def test_missing_model_gives_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make export"):
        build_detector(tmp_path / "nope.onnx", "onnx", config())


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        build_detector(FP32, "openvino", config())


@pytest.mark.slow
def test_torch_and_onnx_agree(frames):
    """The exported graph is the checkpoint.

    Both backends share the decoder and the NMS in ``edgetrack.detect.base``,
    so any disagreement is in the network itself, which is the thing export can
    silently break.
    """
    require(FP32)
    require(CHECKPOINT)
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    onnx_detections, _ = detections_for(FP32, frames)
    detector = build_detector(CHECKPOINT, "torch", config(), device=device)
    detector.warmup(2)
    torch_detections = [detector.detect(frame) for frame in frames]
    detector.close()

    matched = total = 0
    for a, b in zip(onnx_detections, torch_detections):
        total += len(a)
        if not a or not b:
            continue
        ious = iou_matrix(np.array([d.xyxy for d in a]), np.array([d.xyxy for d in b]))
        matched += int((ious.max(axis=1) > 0.95).sum())
    assert total > 0
    assert matched / total > 0.95, f"{matched}/{total} boxes agreed"


@pytest.mark.slow
def test_map_matches_the_ultralytics_validator():
    """This repository's mAP against the one Ultralytics computes.

    They will not agree exactly.  Ultralytics integrates the precision-recall
    curve with a trapezoid over 101 interpolation points; the implementation
    here takes COCO's mean of the same 101 samples, and the matching rules
    differ in how ties are broken.  The tolerance below is what that
    disagreement is worth on this data, and it is checked rather than assumed
    so that a real regression in the metric cannot hide inside it.
    """
    pytest.importorskip("ultralytics")
    require(CHECKPOINT)
    data_yaml = ROOT / "data" / "processed" / "data.yaml"
    if not data_yaml.exists():
        pytest.skip("dataset not generated; run `make data`")

    import json

    from ultralytics import YOLO

    published = ROOT / "results" / "published" / "detection.json"
    if not published.exists():
        pytest.skip("no published detection results to compare against")
    mine = json.loads(published.read_text(encoding="utf-8"))
    fp32 = next(
        entry["metrics"] for entry in mine["variants"].values()
        if entry["detector"]["precision"] == "fp32"
    )

    metrics = YOLO(str(CHECKPOINT)).val(
        data=str(data_yaml.resolve()), split=mine["split"], imgsz=640, verbose=False
    )
    assert float(metrics.box.map) == pytest.approx(fp32["map50_95"], abs=0.05), (
        f"ultralytics {float(metrics.box.map):.4f} against this repository "
        f"{fp32['map50_95']:.4f}"
    )
    assert float(metrics.box.map50) == pytest.approx(fp32["map50"], abs=0.05)


def test_precision_detection_needs_no_onnx_package():
    """The INT8 label must not depend on an optional dependency.
    ``onnx`` is in requirements-train.txt and deliberately absent from the
    runtime.  An earlier version read the graph through it and silently
    reported INT8 models as FP32 wherever it was missing, which is exactly
    the deployed configuration, and CI caught it only once the weights
    started downloading.  This runs the check in an interpreter where
    importing onnx raises, which is what ``sys.modules[name] = None`` does.
    """
    require(INT8)
    require(FP32)
    import subprocess
    import sys as _sys
    script = (
        "import sys\n"
        "sys.path.insert(0, r'{src}')\n"
        "sys.modules['onnx'] = None\n"
        "try:\n"
        "    import onnx\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('onnx was importable')\n"
        "from edgetrack.detect.onnx_backend import _has_quantisation_nodes\n"
        "print(_has_quantisation_nodes(r'{int8}'), "
        "_has_quantisation_nodes(r'{fp32}'))\n"
    ).format(src=ROOT / "src", int8=INT8, fp32=FP32)
    result = subprocess.run(
        [_sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True False", result.stdout


def test_quantisation_scan_finds_a_marker_across_a_chunk_boundary(tmp_path):
    """The chunked read must not miss an operator name split across blocks."""
    from edgetrack.detect.onnx_backend import _has_quantisation_nodes
    path = tmp_path / "spliced.bin"
    path.write_bytes(b"\x00" * 10 + b"QuantizeLinear" + b"\x00" * 10)
    assert _has_quantisation_nodes(path, chunk=12)
    assert _has_quantisation_nodes(path, chunk=1)
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"\x00" * 4096)
    assert not _has_quantisation_nodes(plain, chunk=7)
