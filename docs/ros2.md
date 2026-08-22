# The ROS2 node

`edgetrack.ros2.node` publishes detections, tracks, trajectories and an
annotated image.  It is a real node running under a real rclpy, and the `ros2`
job in `.github/workflows/ci.yml` proves it on every push by bringing up a
`ros:humble-ros-base` container, launching the node and subscribing to the
topics with `tests/ros2_subscriber_check.py`.

---

## Topics

| Topic | Type | What it carries |
| --- | --- | --- |
| `~/detections` | `vision_msgs/Detection2DArray` | Per-frame detections.  `id` is empty. |
| `~/tracks` | `vision_msgs/Detection2DArray` | Confirmed tracks.  `id` is the persistent identity as a string. |
| `~/trajectories` | `visualization_msgs/MarkerArray` | One `LINE_STRIP` per track, the last thirty centres. |
| `~/image_annotated` | `sensor_msgs/Image`, `bgr8` | The frame with boxes, identities and trails drawn on. |

Detections and tracks are separate topics carrying the same message type on
purpose.  A consumer that wants "what is in front of me right now" subscribes
to the first and never has to reason about identity; anything temporal
subscribes to the second.  Publishing one topic and expecting the consumer to
know whether `id` means anything is how a downstream node ends up treating a
detector as a tracker.

All four are node-relative (`~/`), so two instances on two cameras can run side
by side without a remap.

---

## Running it

With ROS2 sourced and the weights fetched:

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=src
python3 -m edgetrack.ros2.node --ros-args \
  -p source:=synth:7 \
  -p model:=models/yolov8n_edgetrack_fp32.onnx \
  -p tracker:=bytetrack
```

`source` takes the same string every other script does: `synth:SEED` for a
generated sequence, a path to a video or a directory of frames, or a camera
index.  The synthetic source is what makes the node runnable in a container
with no camera and no dataset, which is what CI uses.

Parameters, all with defaults:

| Parameter | Default | Notes |
| --- | --- | --- |
| `source` | `synth:7` | as above |
| `model` | `models/yolov8n_edgetrack_fp32.onnx` | must exist; the node refuses to start otherwise |
| `backend` | `onnx` | `onnx`, `torch` or `tensorrt` |
| `tracker` | `bytetrack` | `bytetrack` or `deepsort` |
| `appearance` | `histogram` | `histogram` or `reid`; only read for DeepSORT |
| `reid_model` | `""` | required when `appearance` is `reid` |
| `conf` | `0.25` | detector confidence threshold |
| `imgsz_w`, `imgsz_h` | `640`, `384` | must match the exported graph |
| `frame_id` | `camera_optical` | stamped on every header |
| `publish_image` | `true` | the annotated stream is the expensive topic |
| `rate_hz` | `15.0` | timer period |
| `max_frames` | `0` | `0` means run the source to exhaustion |
| `shutdown_when_done` | `true` | exit when a finite source runs out |

`shutdown_when_done` is what makes the node usable from a script without a
timeout wrapper: on a video file or a synthetic sequence it stops by itself.

---

## Two things that are easy to get wrong

**`vision_msgs` changed shape between 3.x and 4.x.**  In the 4.x series that
ships with Humble, `ObjectHypothesisWithPose.id` became `hypothesis.class_id`
and turned from an integer into a string, and `BoundingBox2D.center` gained a
nested `position`.  The node handles both by duck typing rather than by pinning
a version, because a Jetson running whatever `vision_msgs` its Jetpack shipped
is the normal case rather than the exception.

**`cv_bridge` is not used.**  The annotated image is built by hand in
`edgetrack.ros2.messages.image_payload`.  On a Jetson image `cv_bridge` pulls a
second OpenCV in beside the one already installed, and the conversion is six
lines.  The one subtlety is that a cropped or sliced frame is not C-contiguous
and `tobytes` on it would serialise the wrong memory, so the payload builder
copies when it has to; `tests/test_ros2_messages.py` pins that.

---

## What is not here

No launch file and no `ament` package.  The node runs from the repository with
`python3 -m`, which is enough for a demo and for CI.  Wrapping it in a proper
`ament_python` package with a `setup.py`, a launch file and a parameter YAML is
maybe an hour of work and nothing about it is interesting; it is left undone
rather than half-done.

No `tf2` frames and no calibration.  The trajectory markers are in image pixels
with `z` fixed at zero, and the frame is named `camera_optical` to make the
pixel convention explicit rather than to imply metres.  Projecting tracks onto
a ground plane needs camera intrinsics and an extrinsic calibration, neither of
which exists for a procedurally generated scene.

No composable node and no intra-process transport.  On a Jetson the annotated
image topic at 15 Hz and 640x384 is about 11 MB/s through the middleware, and
a composed node with intra-process comms would avoid that copy.  Worth doing
when there is a board to measure it on.
