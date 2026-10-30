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
FP32 = MODELS / "yolox_tiny_edgetrack_fp32.onnx"
FP16 = MODELS / "yolox_tiny_edgetrack_fp16.onnx"
INT8 = MODELS / "yolox_tiny_edgetrack_int8.onnx"
CHECKPOINT = MODELS / "yolox_tiny_edgetrack.pt"

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
    pytest.importorskip("yolox")
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
def test_map_matches_pycocotools():
    """This repository's mAP against pycocotools on the same predictions.

    Both take COCO's mean of 101 interpolated precision samples, so on
    identical predictions and ground truth they should agree to within how the
    two break a tie between predictions of equal score.  The predictions are
    the FP32 graph's on the val split at the mAP threshold; the ground truth is
    the YOLO label files.  A real regression in the metric moves the numbers
    by far more than the tolerance.
    """
    pytest.importorskip("pycocotools")
    require(FP32)
    data_root = ROOT / "data" / "processed"
    images = sorted((data_root / "images" / "val").glob("*.jpg"))
    if not images:
        pytest.skip("dataset not generated; run `make data`")

    import sys

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from edgetrack.metrics.detection import evaluate_detections

    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import collect_samples

    detector = build_detector(
        FP32,
        "onnx",
        DetectorConfig(
            imgsz=(640, 384), conf_threshold=0.001, iou_threshold=0.6,
            class_names=CLASSES, max_detections=300,
        ),
        providers=["CPUExecutionProvider"],
    )
    samples, _ = collect_samples(detector, images, data_root / "labels" / "val", False)
    detector.close()
    mine = evaluate_detections(samples, list(range(len(CLASSES))))

    def xywh(box) -> list[float]:
        return [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]

    truth: dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
    }
    predictions = []
    for image_id, sample in enumerate(samples, start=1):
        truth["images"].append({"id": image_id, "width": 640, "height": 384})
        for box, cls in zip(sample.gt_boxes, sample.gt_classes):
            bbox = xywh(box)
            truth["annotations"].append({
                "id": len(truth["annotations"]) + 1, "image_id": image_id,
                "category_id": int(cls), "bbox": bbox, "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
        for box, score, cls in zip(sample.pred_boxes, sample.pred_scores, sample.pred_classes):
            predictions.append({
                "image_id": image_id, "category_id": int(cls), "bbox": xywh(box),
                "score": float(score),
            })
    ground = COCO()
    ground.dataset = truth
    ground.createIndex()
    evaluation = COCOeval(ground, ground.loadRes(predictions), "bbox")
    evaluation.params.maxDets = [1, 10, 300]
    evaluation.evaluate()
    evaluation.accumulate()
    # Read AP from the accumulated array rather than from summarize(), whose
    # first statistic is hard-wired to 100 detections per image and comes back
    # as -1 when 100 is not in maxDets.  Axes: IoU, recall, class, area, maxDets.
    precision = evaluation.eval["precision"][:, :, :, 0, -1]
    coco_map50_95 = float(np.mean(precision[precision > -1]))
    at_50 = precision[0]
    coco_map50 = float(np.mean(at_50[at_50 > -1]))
    assert coco_map50_95 == pytest.approx(mine.map50_95, abs=0.01), (
        f"pycocotools {coco_map50_95:.4f} against this repository {mine.map50_95:.4f}"
    )
    assert coco_map50 == pytest.approx(mine.map50, abs=0.01)


def test_precision_detection_needs_no_onnx_package():
    """The INT8 label must not depend on an optional dependency.
    ``onnx`` is in requirements-train.txt and deliberately absent from the
    runtime.  An earlier version read the graph through it and silently
    reported INT8 models as FP32 wherever it was missing, which is exactly
    the deployed configuration and CI caught it only once the weights
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
