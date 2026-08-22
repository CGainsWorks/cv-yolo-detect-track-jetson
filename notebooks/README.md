# Notebooks

There are none, and that is deliberate.

Everything a notebook would do here is a script with `--help` and a JSON
output: generating the dataset, training, exporting, quantising, scoring,
tracking, benchmarking.  Those run in CI, are covered by tests and produce the
numbers the documentation quotes.  A notebook doing the same work would be a
second implementation that nothing checks, and the first thing to drift.

If you want to poke at the pieces interactively, the library is importable:

```python
import sys; sys.path.insert(0, "src")

from edgetrack.data.synth import config_for_split, generate_sequence
from edgetrack.detect import DetectorConfig, build_detector
from edgetrack.pipeline import Pipeline, annotate
from edgetrack.track import build_tracker

frames = list(generate_sequence(config_for_split("test", 7, n_frames=30)))

pipeline = Pipeline(
    build_detector("models/yolov8n_edgetrack_fp32.onnx", "onnx",
                   DetectorConfig(imgsz=(640, 384))),
    build_tracker("bytetrack"),
)
for index, frame in enumerate(frames):
    result = pipeline.process(frame.image, index)
    print(index, len(result.detections), [t.track_id for t in result.tracks])
```

`docs/architecture.md` describes the seams if you want to replace one of the
pieces.
