"""Frame sources and writers.

Four sources behind one iterator, because every script takes ``--source`` and
the ROS2 node takes the same string.  A synthetic source is a first-class
option rather than a test fixture: it is what makes ``make demo`` work on a
machine with no camera, no dataset and no network.

    --source synth:1234        a generated sequence, seed 1234
    --source path/to/clip.mp4  a video file
    --source path/to/frames/   a directory of images, sorted by name
    --source 0                 camera index 0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .data.synth import SynthConfig, config_for_split, generate_sequence

__all__ = ["FrameSource", "open_source", "VideoWriter", "write_gif", "IMAGE_SUFFIXES"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class FrameSource:
    """An iterable of BGR frames with the metadata the writers need."""

    frames: Iterator[np.ndarray]
    width: int
    height: int
    fps: float
    length: int | None
    description: str

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.frames


def _synthetic(spec: str, max_frames: int | None) -> FrameSource:
    seed = int(spec.split(":", 1)[1]) if ":" in spec else 0
    cfg = config_for_split("test", seed)
    if max_frames is not None:
        cfg = SynthConfig(**{**cfg.__dict__, "n_frames": min(max_frames, cfg.n_frames)})

    def generator() -> Iterator[np.ndarray]:
        for frame in generate_sequence(cfg):
            yield frame.image

    return FrameSource(
        frames=generator(),
        width=cfg.width,
        height=cfg.height,
        fps=15.0,
        length=cfg.n_frames,
        description=f"synthetic sequence, seed {seed}, palette {cfg.palette}",
    )


def _video(path: Path, max_frames: int | None) -> FrameSource:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"OpenCV could not open {path} as a video")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    def generator() -> Iterator[np.ndarray]:
        emitted = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
        finally:
            # Released here rather than by the caller: a generator abandoned
            # part way through still runs its finally block when collected, and
            # leaking capture handles is how a long benchmark run ends up
            # unable to open the next file.
            capture.release()

    length = min(count, max_frames) if count and max_frames else (count or max_frames)
    return FrameSource(
        frames=generator(),
        width=width,
        height=height,
        fps=float(fps),
        length=length,
        description=f"video {path.name}",
    )


def _directory(path: Path, max_frames: int | None) -> FrameSource:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images with suffix {IMAGE_SUFFIXES} under {path}")
    if max_frames is not None:
        files = files[:max_frames]
    first = cv2.imread(str(files[0]))
    if first is None:
        raise ValueError(f"OpenCV could not decode {files[0]}")

    def generator() -> Iterator[np.ndarray]:
        for file in files:
            image = cv2.imread(str(file))
            if image is None:
                raise ValueError(f"OpenCV could not decode {file}")
            yield image

    return FrameSource(
        frames=generator(),
        width=first.shape[1],
        height=first.shape[0],
        fps=15.0,
        length=len(files),
        description=f"image directory {path.name} ({len(files)} frames)",
    )


def _camera(index: int, max_frames: int | None) -> FrameSource:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera index {index}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def generator() -> Iterator[np.ndarray]:
        emitted = 0
        try:
            while max_frames is None or emitted < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
                emitted += 1
        finally:
            capture.release()

    return FrameSource(
        frames=generator(),
        width=width,
        height=height,
        fps=capture.get(cv2.CAP_PROP_FPS) or 30.0,
        length=max_frames,
        description=f"camera {index}",
    )


def open_source(spec: str, max_frames: int | None = None) -> FrameSource:
    """Resolve a ``--source`` string into a :class:`FrameSource`."""
    if spec.startswith("synth"):
        return _synthetic(spec, max_frames)
    path = Path(spec)
    if path.is_dir():
        return _directory(path, max_frames)
    if path.exists():
        return _video(path, max_frames)
    if spec.isdigit():
        return _camera(int(spec), max_frames)
    raise FileNotFoundError(
        f"cannot interpret --source {spec!r}: not a file, a directory, a camera "
        "index or a synth:SEED specifier"
    )


class VideoWriter:
    """MP4 writer that fails loudly if the codec is missing.

    ``cv2.VideoWriter`` returns an object that silently drops every frame when
    the fourcc is unavailable, which on a headless Linux box with the wrong
    OpenCV build produces a zero byte file and no error.  Checking ``isOpened``
    turns that into an exception.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), float(fps), size
        )
        if not self.writer.isOpened():
            raise RuntimeError(
                f"OpenCV could not open {self.path} for writing with fourcc {fourcc!r}.  "
                "On a headless build try --fourcc MJPG and an .avi extension."
            )
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def write_gif(
    path: str | Path,
    frames: list[np.ndarray],
    fps: float = 10.0,
    scale: float = 0.6,
    palette_colours: int = 32,
) -> Path:
    """Write an animated GIF.  Used for the README header image.

    Downscaled by default because a full resolution GIF of a hundred frames is
    several megabytes and this repository keeps large binaries out of the pack
    file.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resized = []
    for frame in frames:
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if palette_colours:
            # GIF is palettised anyway.  Quantising deliberately, rather than
            # letting the encoder do it per frame, keeps the palette stable
            # across frames and roughly halves the file: a noisy synthetic
            # scene otherwise produces a new palette every frame and the
            # inter-frame compression has nothing to work with.
            quantised = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            quantised = quantised // (256 // palette_colours) * (256 // palette_colours)
            rgb = cv2.cvtColor(quantised, cv2.COLOR_BGR2RGB)
        resized.append(rgb)
    imageio.mimsave(str(path), resized, duration=1.0 / max(fps, 1e-3), loop=0)
    return path
