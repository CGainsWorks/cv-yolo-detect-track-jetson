# cv-yolo-detect-track-jetson

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

An end-to-end edge detection and tracking pipeline: fine-tune YOLOv8 on a custom object set, attach ByteTrack and DeepSORT for persistent IDs, compile to TensorRT and benchmark FP32 against FP16 and INT8 on Jetson, publishing results as ROS2 topics.

---

## What this covers

CV entries this repository is the evidence for:

- Real-Time Object Detection and Tracking, YOLOv8 and Jetson

---

## Requirements

### Hardware

- NVIDIA Jetson (Orin Nano, Xavier NX or similar)
- USB or CSI camera
- Optional ROS2 host machine
- Training machine with a CUDA GPU

### Software

- Ultralytics YOLOv8
- ByteTrack and DeepSORT
- TensorRT
- ROS2 publisher nodes
- Annotation tooling (CVAT or Label Studio)
- PyTorch

### Report and documentation deliverables

- mAP on the custom set
- MOTA and IDF1 tracking metrics
- FPS across FP32, FP16 and INT8
- ID-switch analysis
- Dataset card

---

## Inputs

- Labelled custom dataset
- Training config
- Live or recorded video
- Tracker parameters

## Outputs

- Trained weights
- TensorRT engine file
- Per-frame detections with class and confidence
- Persistent track IDs with trajectories
- Annotated video stream
- ROS2 detection topics

---

## Intended structure

```
configs/                data, model, train, tracker, export
src/
  data/                   dataset, augment, download
  train.py
  evaluate.py
  export_tensorrt.py
  track.py
  serve/                  api.py, ros2_node.py
data/                     raw, processed, DATASET_CARD.md
models/                   weights via Releases, not Git
benchmarks/               latency_jetson.md, results.csv
notebooks/
docker/                   Dockerfile.train, Dockerfile.jetson
docs/                     MODEL_CARD.md, results.md
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Dataset collection and labelling, with a dataset card
- [ ] YOLOv8 fine-tuning with reported mAP
- [ ] ByteTrack integration and MOTA/IDF1 evaluation
- [ ] DeepSORT integration for comparison
- [ ] TensorRT export at FP32, FP16 and INT8
- [ ] Jetson latency and throughput benchmark across precisions
- [ ] ROS2 node publishing detections and tracks
- [ ] Error analysis on ID switches
- [ ] Model card and results write-up

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: cv-yolo-detect-track-jetson
An end-to-end edge detection and tracking pipeline: fine-tune YOLOv8 on a custom object set, attach ByteTrack and DeepSORT for persistent IDs, compile to TensorRT and benchmark FP32 against FP16 and INT8 on Jetson, publishing results as ROS2 topics.

TARGET STACK
- Ultralytics YOLOv8
- ByteTrack and DeepSORT
- TensorRT
- ROS2 publisher nodes
- Annotation tooling (CVAT or Label Studio)
- PyTorch

HARDWARE CONTEXT
- NVIDIA Jetson (Orin Nano, Xavier NX or similar)
- USB or CSI camera
- Optional ROS2 host machine
- Training machine with a CUDA GPU

THE SYSTEM MUST TAKE THESE INPUTS
- Labelled custom dataset
- Training config
- Live or recorded video
- Tracker parameters

AND PRODUCE THESE OUTPUTS
- Trained weights
- TensorRT engine file
- Per-frame detections with class and confidence
- Persistent track IDs with trajectories
- Annotated video stream
- ROS2 detection topics

INTENDED REPOSITORY LAYOUT
configs/                data, model, train, tracker, export
src/
  data/                   dataset, augment, download
  train.py
  evaluate.py
  export_tensorrt.py
  track.py
  serve/                  api.py, ros2_node.py
data/                     raw, processed, DATASET_CARD.md
models/                   weights via Releases, not Git
benchmarks/               latency_jetson.md, results.csv
notebooks/
docker/                   Dockerfile.train, Dockerfile.jetson
docs/                     MODEL_CARD.md, results.md

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
