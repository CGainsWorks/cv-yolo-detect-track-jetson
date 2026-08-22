"""ByteTrack: association by IoU, in two passes, keeping the low scores.

The idea in one sentence.  Most trackers throw away detections below the
confidence threshold; ByteTrack keeps them and gives them a second chance to
match a track that failed to match anything in the first pass, on the argument
that a box which is occluded or motion-blurred still scores low but still marks
where the object is.

Implemented from the paper (Zhang et al., ECCV 2022) rather than vendored, for
three reasons: the reference implementation ships with a YOLOX training stack
this repository does not want, the version in Ultralytics is AGPL and would
contaminate the runtime image, and the ID-switch attribution in
``src/edgetrack/metrics/idswitch.py`` needs hooks into the association stages
that neither exposes.

The pass structure:

1. predict every track,
2. split detections at ``track_thresh`` into high and low,
3. match high detections against tracked *and* lost tracks by IoU,
4. match low detections against the tracks still unmatched from step 3, using
   a looser threshold, and only against tracks that were being tracked last
   frame,
5. unmatched tracks go to lost, and are deleted after ``track_buffer`` frames,
6. unmatched high detections above ``det_thresh`` start new tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import Detection, Track
from .base import BaseTracker, KalmanTrack
from .kalman import KalmanFilter
from .matching import fuse_score, iou_distance, linear_assignment

__all__ = ["ByteTrackConfig", "ByteTrack"]


@dataclass
class ByteTrackConfig:
    """Defaults are what ``scripts/tune_tracker.py`` chose on the val split.

    The sweep is in ``results/published/tuning_bytetrack.json`` and
    ``configs/tracker_bytetrack.yaml`` carries the same numbers;
    ``tests/test_configs.py`` fails if the two drift apart.
    """

    track_thresh: float = 0.5
    det_thresh: float = 0.6
    low_thresh: float = 0.1
    match_thresh: float = 0.7
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    track_buffer: int = 30
    n_init: int = 1
    fuse_score: bool = False
    class_aware: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.low_thresh <= self.track_thresh <= 1.0:
            raise ValueError(
                f"need 0 <= low_thresh <= track_thresh <= 1, got {self.low_thresh} "
                f"and {self.track_thresh}"
            )
        if self.det_thresh < self.track_thresh:
            raise ValueError(
                "det_thresh is the bar for starting a new track and must not be "
                f"below track_thresh, got {self.det_thresh} < {self.track_thresh}"
            )
        if self.track_buffer < 1:
            raise ValueError(f"track_buffer must be at least 1, got {self.track_buffer}")


class ByteTrack(BaseTracker):
    """Motion-only tracker.  Appearance features on detections are ignored."""

    def __init__(
        self, config: ByteTrackConfig | None = None, kf: KalmanFilter | None = None
    ) -> None:
        super().__init__()
        self.config = config or ByteTrackConfig()
        self.config.validate()
        self._kf = kf or KalmanFilter()
        self.lost: list[KalmanTrack] = []
        # Filled in per frame and read by the ID-switch analysis, which wants
        # to know which stage a match came from.
        self.last_association: dict[str, list[tuple[int, int]]] = {}

    def reset(self) -> None:
        super().reset()
        self.lost = []
        self.last_association = {}

    # -- helpers ----------------------------------------------------------

    def _cost(
        self, tracks: Sequence[KalmanTrack], detections: Sequence[Detection], fuse: bool
    ) -> np.ndarray:
        cost = iou_distance(self._boxes(tracks), self._boxes(detections))
        if self.config.class_aware and cost.size:
            # Cross-class association is almost always a mistake in a scene with
            # both people and vehicles: the boxes overlap constantly and the
            # classes are never confusable in practice.  Forbid it rather than
            # penalise it, and record the choice here because it is the reason
            # this tracker beats the class-agnostic variant on IDF1 by a couple
            # of points in docs/tracking.md.
            track_classes = np.array([t.class_id for t in tracks])[:, None]
            det_classes = np.array([d.class_id for d in detections])[None, :]
            cost = np.where(track_classes == det_classes, cost, 1.0)
        if fuse and cost.size:
            cost = fuse_score(cost, np.array([d.score for d in detections]))
        return cost

    # -- main loop --------------------------------------------------------

    def update(self, detections: Sequence[Detection], frame_id: int | None = None) -> list[Track]:
        cfg = self.config
        self.frame_id = self.frame_id + 1 if frame_id is None else int(frame_id)
        self.last_association = {"high": [], "low": [], "unconfirmed": []}

        scores = np.array([d.score for d in detections], dtype=np.float64)
        high_idx = [i for i, s in enumerate(scores) if s >= cfg.track_thresh]
        low_idx = [i for i, s in enumerate(scores) if cfg.low_thresh <= s < cfg.track_thresh]
        high = [detections[i] for i in high_idx]
        low = [detections[i] for i in low_idx]

        confirmed = [t for t in self.tracks if t.is_confirmed]
        unconfirmed = [t for t in self.tracks if not t.is_confirmed]

        for track in self.tracks + self.lost:
            track.predict()

        # Stage 1: high-confidence detections against tracked and lost tracks.
        pool = confirmed + self.lost
        cost = self._cost(pool, high, cfg.fuse_score)
        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost, cfg.match_thresh)

        for ti, di in matches:
            track = pool[ti]
            track.update(high[di], self.frame_id, cfg.n_init)
            if track in self.lost:
                self.lost.remove(track)
                self.tracks.append(track)
            self.last_association["high"].append((track.track_id, high_idx[di]))

        # Stage 2: low-confidence detections against the tracks that were being
        # tracked last frame and did not match above.  Lost tracks are excluded
        # on purpose: a weak detection is not enough evidence to revive an
        # identity that has been missing, and allowing it was worth roughly two
        # extra ID switches per sequence in the ablation.
        remaining = [
            pool[i]
            for i in unmatched_tracks
            if pool[i] in self.tracks and pool[i].time_since_update == 1
        ]
        cost = self._cost(remaining, low, fuse=False)
        matches2, unmatched_remaining, _ = linear_assignment(cost, cfg.second_match_thresh)
        for ti, di in matches2:
            track = remaining[ti]
            track.update(low[di], self.frame_id, cfg.n_init)
            self.last_association["low"].append((track.track_id, low_idx[di]))

        matched_second = {id(remaining[i]) for i, _ in matches2}
        for i in unmatched_tracks:
            track = pool[i]
            if id(track) in matched_second:
                continue
            if track in self.tracks:
                track.mark_missed(cfg.track_buffer, cfg.n_init)
                if not track.is_deleted:
                    self.tracks.remove(track)
                    self.lost.append(track)
            else:
                track.mark_missed(cfg.track_buffer, cfg.n_init)

        # Stage 3: leftover high detections against tentative tracks, so that a
        # brand new track does not immediately spawn a duplicate.
        leftover = [high[i] for i in unmatched_dets]
        leftover_idx = [high_idx[i] for i in unmatched_dets]
        cost = self._cost(unconfirmed, leftover, fuse=False)
        matches3, unmatched_unconfirmed, unmatched_leftover = linear_assignment(
            cost, cfg.unconfirmed_match_thresh
        )
        for ti, di in matches3:
            track = unconfirmed[ti]
            track.update(leftover[di], self.frame_id, cfg.n_init)
            self.last_association["unconfirmed"].append((track.track_id, leftover_idx[di]))
        for i in unmatched_unconfirmed:
            unconfirmed[i].mark_missed(cfg.track_buffer, cfg.n_init)

        # Stage 4: new tracks.
        for i in unmatched_leftover:
            detection = leftover[i]
            if detection.score < cfg.det_thresh:
                continue
            self.tracks.append(
                KalmanTrack(self._kf, self._allocate_id(), detection, self.frame_id)
            )

        self.tracks = [t for t in self.tracks if not t.is_deleted]
        self.lost = [t for t in self.lost if not t.is_deleted]

        return [
            t.to_public()
            for t in self.tracks
            if t.is_confirmed and t.time_since_update == 0
        ]
