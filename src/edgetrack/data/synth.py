"""Procedural generator for the custom dataset.

Why this exists instead of photographs
--------------------------------------
The brief asks for a custom object set with a dataset card.  Photographing and
hand labelling a multi-object tracking set is weeks of work, and hand labelling
*track identities* through occlusions is the expensive part.  This module
renders a warehouse-floor scene instead, which buys three things that matter
more to this repository than photorealism:

* exact ground truth for both boxes and identities, so MOTA, IDF1 and the
  ID-switch attribution are measured against truth rather than against another
  model's output,
* a controllable occlusion and crossing schedule, so the tracker comparison is
  run on the cases that actually separate ByteTrack from DeepSORT,
* a reproducible dataset that anybody can regenerate from a seed, which is what
  makes every number in the README checkable.

It is not photorealistic and a detector trained here will not transfer to a
real camera.  That is stated in ``data/DATASET_CARD.md`` and in the README
limitations section.  :mod:`edgetrack.data.convert` imports a real CVAT or
Label Studio export into the same on-disk layout, so swapping the synthetic set
for a photographed one changes no code below ``scripts/``.

Determinism
-----------
Everything is driven by ``numpy.random.Generator`` seeded from
:class:`SynthConfig`.  The same config gives byte-identical frames on the same
OpenCV version.  ``tests/test_synth.py`` pins that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterator, Sequence

import cv2
import numpy as np

__all__ = [
    "CLASSES",
    "Annotation",
    "SynthConfig",
    "SequenceFrame",
    "generate_sequence",
    "config_for_split",
]

# Class list.  Order is the label index order and must not change without
# regenerating the dataset; configs/data.yaml carries the same list and
# tests/test_configs.py checks the two agree.
CLASSES: tuple[str, ...] = ("person", "forklift", "pallet", "cone")

_MOVING_CLASSES = (0, 1)  # person, forklift
_STATIC_CLASSES = (2, 3)  # pallet, cone


@dataclass(frozen=True)
class Annotation:
    """One ground-truth object in one frame."""

    class_id: int
    instance_id: int
    xyxy: tuple[float, float, float, float]
    visibility: float

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1, y1, x2 - x1, y2 - y1)


@dataclass(frozen=True)
class SynthConfig:
    """Everything that determines a sequence.

    ``palette`` picks the background and lighting family.  The test split uses
    a different palette from train and val on purpose: it is a deliberate,
    small domain gap so that mAP does not sit at the ceiling and the accuracy
    cost of INT8 quantisation is measurable rather than lost in rounding.
    """

    seed: int
    width: int = 640
    height: int = 384
    n_frames: int = 24
    n_movers: int = 4
    n_static: int = 3
    n_occluders: int = 2
    palette: str = "warehouse"
    noise_sigma: float = 4.0
    blur_scale: float = 0.55
    jpeg_quality: tuple[int, int] = (72, 95)
    brightness_jitter: float = 0.12
    crossing_bias: float = 0.6
    min_label_visibility: float = 0.35

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


@dataclass
class SequenceFrame:
    """A rendered frame with its ground truth."""

    frame_id: int
    image: np.ndarray  # HxWx3 uint8 BGR
    annotations: list[Annotation] = field(default_factory=list)


# --------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------

_PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    # BGR, because OpenCV.
    "warehouse": {
        "floor_near": (118, 122, 126),
        "floor_far": (72, 76, 82),
        "line": (150, 168, 178),
        "wall": (58, 60, 66),
    },
    "depot": {
        "floor_near": (96, 112, 128),
        "floor_far": (48, 58, 70),
        "line": (128, 150, 170),
        "wall": (40, 48, 58),
    },
    "yard": {
        "floor_near": (108, 108, 100),
        "floor_far": (60, 62, 58),
        "line": (170, 172, 160),
        "wall": (52, 54, 50),
    },
}


def config_for_split(split: str, seed: int, **overrides: object) -> SynthConfig:
    """Config for one of ``train``, ``val`` or ``test``.

    Train and val share a palette and a noise level.  Test is rendered in a
    different palette with more sensor noise and stronger blur, which is the
    domain gap referred to in the dataset card.  Test sequences are also longer
    because they are what the tracking metrics run on.
    """
    if split in ("train", "val"):
        base = SynthConfig(seed=seed, palette="warehouse")
    elif split == "test":
        base = SynthConfig(
            seed=seed,
            palette="depot",
            n_frames=120,
            noise_sigma=7.0,
            blur_scale=0.85,
            jpeg_quality=(60, 88),
            brightness_jitter=0.18,
            n_movers=5,
            n_occluders=3,
        )
    else:
        raise ValueError(f"unknown split {split!r}")
    return replace(base, **overrides) if overrides else base


# --------------------------------------------------------------------------
# Background
# --------------------------------------------------------------------------


def _render_background(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    h, w = cfg.height, cfg.width
    colours = _PALETTES[cfg.palette]
    horizon = int(h * rng.uniform(0.20, 0.30))

    image = np.zeros((h, w, 3), dtype=np.float32)
    image[:horizon] = np.asarray(colours["wall"], dtype=np.float32)

    # Floor as a vertical gradient from far to near.
    rows = np.linspace(0.0, 1.0, h - horizon, dtype=np.float32)[:, None, None]
    far = np.asarray(colours["floor_far"], dtype=np.float32)
    near = np.asarray(colours["floor_near"], dtype=np.float32)
    image[horizon:] = far + (near - far) * rows

    # Perspective floor markings converging on a vanishing point.
    vanish_x = int(w * rng.uniform(0.35, 0.65))
    for offset in np.linspace(-2.4, 2.4, 9):
        x_bottom = int(vanish_x + offset * w * 0.55)
        cv2.line(
            image,
            (vanish_x, horizon),
            (x_bottom, h),
            colours["line"],
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    for depth in np.linspace(0.06, 1.0, 7):
        y = int(horizon + (h - horizon) * depth**1.6)
        cv2.line(image, (0, y), (w, y), colours["line"], 1, cv2.LINE_AA)

    # Racking silhouettes on the back wall, so the top of the frame is not flat.
    for _ in range(int(rng.integers(3, 7))):
        x = int(rng.integers(0, w))
        bay_w = int(rng.integers(30, 90))
        bay_h = int(rng.integers(10, horizon))
        shade = float(rng.uniform(0.65, 1.25))
        patch = tuple(min(255.0, float(c) * shade) for c in colours["wall"])
        cv2.rectangle(image, (x, horizon - bay_h), (x + bay_w, horizon), patch, -1)

    # Grime.  Low frequency multiplicative noise keeps the floor from looking
    # like a gradient fill, which would let the detector key on it.
    grime = rng.normal(1.0, 0.05, size=(h // 16 + 2, w // 16 + 2)).astype(np.float32)
    grime = cv2.resize(grime, (w, h), interpolation=cv2.INTER_CUBIC)
    image *= grime[:, :, None]

    return np.clip(image, 0, 255)


def _render_occluders(
    cfg: SynthConfig, rng: np.random.Generator
) -> list[tuple[int, int, int, int, tuple[float, float, float]]]:
    """Static foreground blockers.  Objects pass behind these and vanish."""
    colours = _PALETTES[cfg.palette]
    out = []
    for _ in range(cfg.n_occluders):
        pw = int(rng.integers(cfg.width // 14, cfg.width // 7))
        ph = int(rng.integers(cfg.height // 3, int(cfg.height * 0.8)))
        x = int(rng.integers(0, max(1, cfg.width - pw)))
        y = cfg.height - ph
        shade = float(rng.uniform(0.5, 0.8))
        colour = tuple(float(c) * shade for c in colours["wall"])
        out.append((x, y, pw, ph, colour))
    return out


# --------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------


@dataclass
class _Object:
    instance_id: int
    class_id: int
    x: float  # centre x
    y: float  # bottom y, so objects sit on the floor
    vx: float
    vy: float
    height: float
    colour: tuple[float, float, float]
    phase: float

    @property
    def width(self) -> float:
        if self.class_id == 0:  # person
            return self.height * 0.38
        if self.class_id == 1:  # forklift
            return self.height * 1.25
        if self.class_id == 2:  # pallet
            return self.height * 2.6
        return self.height * 0.85  # cone

    def box(self) -> tuple[float, float, float, float]:
        w = self.width
        return (self.x - w / 2.0, self.y - self.height, self.x + w / 2.0, self.y)


def _instance_colour(rng: np.random.Generator, class_id: int) -> tuple[float, float, float]:
    """Per-instance colour, which is the only appearance cue DeepSORT can use.

    Hue is drawn per instance and saturation kept high, so two instances of the
    same class are usually separable by colour.  ``person`` gets the widest
    spread because people are the class the trackers are compared on.
    """
    if class_id == 3:  # cone, always orange
        hue = float(rng.uniform(8, 20))
        sat, val = 220.0, float(rng.uniform(190, 245))
    elif class_id == 2:  # pallet, always timber
        hue = float(rng.uniform(12, 22))
        sat, val = float(rng.uniform(110, 160)), float(rng.uniform(120, 170))
    else:
        hue = float(rng.uniform(0, 180))
        sat = float(rng.uniform(120, 235))
        val = float(rng.uniform(110, 230))
    hsv = np.uint8([[[int(hue), int(sat), int(val)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (float(bgr[0]), float(bgr[1]), float(bgr[2]))


def _spawn_objects(cfg: SynthConfig, rng: np.random.Generator) -> list[_Object]:
    objects: list[_Object] = []
    next_id = 1

    for _ in range(cfg.n_movers):
        class_id = int(rng.choice(_MOVING_CLASSES, p=[0.7, 0.3]))
        height = (
            float(rng.uniform(46, 118))
            if class_id == 0
            else float(rng.uniform(40, 86))
        )
        obj = _Object(
            instance_id=next_id,
            class_id=class_id,
            x=float(rng.uniform(0.08, 0.92) * cfg.width),
            y=float(rng.uniform(0.45, 0.99) * cfg.height),
            vx=float(rng.uniform(-3.6, 3.6)),
            vy=float(rng.uniform(-0.9, 0.9)),
            height=height,
            colour=_instance_colour(rng, class_id),
            phase=float(rng.uniform(0, 2 * math.pi)),
        )
        objects.append(obj)
        next_id += 1

    for _ in range(cfg.n_static):
        class_id = int(rng.choice(_STATIC_CLASSES, p=[0.5, 0.5]))
        height = (
            float(rng.uniform(14, 26)) if class_id == 2 else float(rng.uniform(16, 30))
        )
        objects.append(
            _Object(
                instance_id=next_id,
                class_id=class_id,
                x=float(rng.uniform(0.05, 0.95) * cfg.width),
                y=float(rng.uniform(0.55, 0.99) * cfg.height),
                vx=0.0,
                vy=0.0,
                height=height,
                colour=_instance_colour(rng, class_id),
                phase=0.0,
            )
        )
        next_id += 1

    # Crossing schedule.  Without this, whether two movers ever meet is left to
    # chance and most sequences would contain no hard association at all.  Pick
    # pairs of movers and aim them at a shared point midway through.
    movers = [o for o in objects if o.class_id in _MOVING_CLASSES]
    rng.shuffle(movers)
    for first, second in zip(movers[0::2], movers[1::2]):
        if rng.random() > cfg.crossing_bias:
            continue
        meet_frame = float(rng.uniform(0.35, 0.7)) * cfg.n_frames
        meet_x = float(rng.uniform(0.25, 0.75) * cfg.width)
        meet_y = float(rng.uniform(0.55, 0.95) * cfg.height)
        for obj in (first, second):
            obj.vx = (meet_x - obj.x) / max(meet_frame, 1.0)
            obj.vy = (meet_y - obj.y) / max(meet_frame, 1.0)

    return objects


def _step(objects: Sequence[_Object], cfg: SynthConfig, rng: np.random.Generator) -> None:
    for obj in objects:
        if obj.vx == 0.0 and obj.vy == 0.0:
            continue
        obj.vx += float(rng.normal(0.0, 0.09))
        obj.vy += float(rng.normal(0.0, 0.035))
        obj.vx = float(np.clip(obj.vx, -5.5, 5.5))
        obj.vy = float(np.clip(obj.vy, -1.6, 1.6))
        obj.x += obj.vx
        obj.y += obj.vy
        # Bounce off the frame with a margin, so objects stay mostly visible.
        if obj.x < 0.02 * cfg.width or obj.x > 0.98 * cfg.width:
            obj.vx = -obj.vx
            obj.x = float(np.clip(obj.x, 0.02 * cfg.width, 0.98 * cfg.width))
        floor_top, floor_bottom = 0.42 * cfg.height, 0.995 * cfg.height
        if obj.y < floor_top or obj.y > floor_bottom:
            obj.vy = -obj.vy
            obj.y = float(np.clip(obj.y, floor_top, floor_bottom))
        # Objects further up the frame are further away, so they are smaller.
        depth = (obj.y - floor_top) / max(floor_bottom - floor_top, 1e-6)
        obj.height *= float(np.clip(1.0 + 0.02 * (depth - 0.5), 0.98, 1.02))
        obj.phase += 0.45


def _draw_object(obj: _Object) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Render one object into a local patch.

    Returns the BGR patch, its 0/255 alpha mask and the top-left corner the
    patch belongs at.  Drawing locally rather than into the full frame keeps
    the cost proportional to object area, which matters because the visibility
    computation composites every object twice.
    """
    x1, y1, x2, y2 = obj.box()
    pad = 2
    ox, oy = int(math.floor(x1)) - pad, int(math.floor(y1)) - pad
    pw = int(math.ceil(x2 - x1)) + 2 * pad
    ph = int(math.ceil(y2 - y1)) + 2 * pad
    pw, ph = max(pw, 3), max(ph, 3)

    patch = np.zeros((ph, pw, 3), dtype=np.float32)
    mask = np.zeros((ph, pw), dtype=np.uint8)
    colour = obj.colour
    dark = tuple(c * 0.45 for c in colour)

    if obj.class_id == 0:  # person
        head_r = max(2, int(pw * 0.30))
        cx = pw // 2
        cv2.circle(patch, (cx, pad + head_r), head_r, dark, -1, cv2.LINE_AA)
        cv2.circle(mask, (cx, pad + head_r), head_r, 255, -1, cv2.LINE_AA)
        torso_top = pad + 2 * head_r
        torso_bottom = pad + int((ph - 2 * pad) * 0.66)
        cv2.rectangle(patch, (pad, torso_top), (pw - pad, torso_bottom), colour, -1)
        cv2.rectangle(mask, (pad, torso_top), (pw - pad, torso_bottom), 255, -1)
        # Legs swing with phase, which gives the motion-blur term something to
        # bite on and stops every crop of a person being identical.
        swing = int(max(1, pw * 0.18) * math.sin(obj.phase))
        for sign in (-1, 1):
            top = (cx + sign * max(1, pw // 5), torso_bottom)
            bottom = (cx + sign * max(1, pw // 5) + swing * sign, ph - pad)
            cv2.line(patch, top, bottom, dark, max(1, pw // 5), cv2.LINE_AA)
            cv2.line(mask, top, bottom, 255, max(1, pw // 5), cv2.LINE_AA)
    elif obj.class_id == 1:  # forklift
        body_top = pad + int((ph - 2 * pad) * 0.30)
        cv2.rectangle(patch, (pad, body_top), (pw - pad, ph - pad - 2), colour, -1)
        cv2.rectangle(mask, (pad, body_top), (pw - pad, ph - pad - 2), 255, -1)
        mast_x = pw - pad - max(2, pw // 12)
        cv2.rectangle(patch, (mast_x, pad), (pw - pad, ph - pad), dark, -1)
        cv2.rectangle(mask, (mast_x, pad), (pw - pad, ph - pad), 255, -1)
        wheel_r = max(2, ph // 8)
        for wx in (pad + wheel_r + 1, pw - pad - wheel_r - 1):
            cv2.circle(patch, (wx, ph - pad - wheel_r), wheel_r, (30, 30, 32), -1, cv2.LINE_AA)
            cv2.circle(mask, (wx, ph - pad - wheel_r), wheel_r, 255, -1, cv2.LINE_AA)
    elif obj.class_id == 2:  # pallet
        cv2.rectangle(patch, (pad, pad), (pw - pad, ph - pad), colour, -1)
        cv2.rectangle(mask, (pad, pad), (pw - pad, ph - pad), 255, -1)
        for i in range(1, 5):
            sx = pad + int((pw - 2 * pad) * i / 5.0)
            cv2.line(patch, (sx, pad), (sx, ph - pad), dark, 1)
    else:  # cone
        apex = (pw // 2, pad)
        left = (pad, ph - pad)
        right = (pw - pad, ph - pad)
        tri = np.array([apex, left, right], dtype=np.int32)
        cv2.fillConvexPoly(patch, tri, colour, cv2.LINE_AA)
        cv2.fillConvexPoly(mask, tri, 255, cv2.LINE_AA)
        band = ph // 2
        cv2.line(patch, (pad, band), (pw - pad, band), (245, 245, 245), max(1, ph // 10))

    return patch, mask, (ox, oy)


def _composite(
    canvas: np.ndarray,
    id_map: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    marker: int,
) -> int:
    """Alpha-composite a patch and stamp ``marker`` into the id map.

    Returns the number of pixels the patch would cover if nothing were drawn
    over it, which is the denominator of the visibility ratio.
    """
    h, w = canvas.shape[:2]
    ox, oy = origin
    ph, pw = mask.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + pw), min(h, oy + ph)
    if x0 >= x1 or y0 >= y1:
        return 0
    sub_mask = mask[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    sub_patch = patch[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    where = sub_mask > 0
    canvas[y0:y1, x0:x1][where] = sub_patch[where]
    id_map[y0:y1, x0:x1][where] = marker
    return int(where.sum())


def _degrade(
    image: np.ndarray, cfg: SynthConfig, rng: np.random.Generator, max_speed: float
) -> np.ndarray:
    """Sensor and codec effects.

    Order matters and follows a real camera: motion blur happens in the optics,
    noise in the sensor, gain in the ISP and compression last.
    """
    out = image
    blur_px = cfg.blur_scale * max_speed
    if blur_px > 1.0:
        k = int(min(15, max(3, round(blur_px))))
        if k % 2 == 0:
            k += 1
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)

    gain = 1.0 + float(rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter))
    bias = float(rng.uniform(-8, 8))
    out = out * gain + bias

    if cfg.noise_sigma > 0:
        out = out + rng.normal(0.0, cfg.noise_sigma, size=out.shape)

    out = np.clip(out, 0, 255).astype(np.uint8)

    quality = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    ok, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - imencode does not fail for valid uint8 BGR
        raise RuntimeError("JPEG encode failed while degrading a synthetic frame")
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def generate_sequence(cfg: SynthConfig) -> Iterator[SequenceFrame]:
    """Yield ``cfg.n_frames`` rendered frames with ground truth.

    Objects are drawn back to front by their bottom edge, so a nearer object
    occludes a further one.  Occluders are drawn last and belong to no
    instance, which is what makes a track disappear completely for a run of
    frames rather than merely shrink.
    """
    rng = cfg.rng()
    background = _render_background(cfg, rng)
    occluders = _render_occluders(cfg, rng)
    objects = _spawn_objects(cfg, rng)

    for frame_id in range(cfg.n_frames):
        if frame_id > 0:
            _step(objects, cfg, rng)

        canvas = background.copy()
        id_map = np.full(canvas.shape[:2], -1, dtype=np.int32)
        order = sorted(range(len(objects)), key=lambda i: objects[i].y)

        own_area: dict[int, int] = {}
        for index in order:
            obj = objects[index]
            patch, mask, origin = _draw_object(obj)
            own_area[index] = _composite(canvas, id_map, patch, mask, origin, index)

        for x, y, pw, ph, colour in occluders:
            cv2.rectangle(canvas, (x, y), (x + pw, y + ph), colour, -1)
            id_map[y : y + ph, x : x + pw] = -2

        counts = np.bincount(
            id_map[id_map >= 0].ravel(), minlength=len(objects)
        ) if (id_map >= 0).any() else np.zeros(len(objects), dtype=np.int64)

        annotations: list[Annotation] = []
        for index, obj in enumerate(objects):
            area = own_area.get(index, 0)
            if area <= 0:
                continue
            visible = int(counts[index]) if index < len(counts) else 0
            x1, y1, x2, y2 = obj.box()
            annotations.append(
                Annotation(
                    class_id=obj.class_id,
                    instance_id=obj.instance_id,
                    xyxy=(
                        float(np.clip(x1, 0, cfg.width)),
                        float(np.clip(y1, 0, cfg.height)),
                        float(np.clip(x2, 0, cfg.width)),
                        float(np.clip(y2, 0, cfg.height)),
                    ),
                    visibility=float(min(1.0, visible / max(area, 1))),
                )
            )

        max_speed = max(
            (math.hypot(o.vx, o.vy) for o in objects if o.vx or o.vy), default=0.0
        )
        image = _degrade(canvas, cfg, rng, max_speed)
        yield SequenceFrame(frame_id=frame_id, image=image, annotations=annotations)
