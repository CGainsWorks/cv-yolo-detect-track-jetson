"""DeepSORT: appearance-gated association with a matching cascade.

The contrast with ByteTrack is the point of having both.  ByteTrack decides
with geometry alone and recovers occlusions by keeping weak detections.
DeepSORT decides with an appearance descriptor, gated by a Mahalanobis test on
the Kalman state, and prioritises tracks that were seen recently through a
cascade.  Where they differ in practice is exactly the case the synthetic
sequences are built around: two objects of the same class crossing, where
geometry is ambiguous for several frames and colour is not.

Implemented from the paper (Wojke et al., ICIP 2017).  The reference code is
GPL-3.0 and this repository is MIT, so nothing is vendored.

Association order per frame:

1. predict every track,
2. matching cascade over confirmed tracks by ascending ``time_since_update``:
   at each age, cosine distance between detections and the track's feature
   gallery, with entries failing the Mahalanobis gate set to infinity,
3. IoU association between the remaining detections, the unconfirmed tracks
   and the confirmed tracks that were seen last frame,
4. unmatched tracks age out after ``max_age``, unmatched detections start
   tentative tracks that need ``n_init`` hits to confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import Detection, Track
from .base import BaseTracker, KalmanTrack
from .kalman import CHI2_INV95, KalmanFilter
from .matching import cosine_distance, iou_distance, linear_assignment

__all__ = ["DeepSortConfig", "DeepSort"]


@dataclass
class DeepSortConfig:
    """Defaults are what ``scripts/tune_tracker.py`` chose on the val split.

    ``max_cosine_distance`` at 0.5 looks loose next to the 0.2 the paper uses,
    and it agrees with the margin measured on the trained embedding: within an
    identity the cosine distance runs to roughly 0.68 at the 95th percentile
    whilst between identities it averages close to 1.0, so a gate at a quarter
    would reject most of the true matches it exists to recover.  The numbers
    are in ``results/published/reid_train.json``.
    """

    max_cosine_distance: float = 0.5
    max_iou_distance: float = 0.6
    max_age: int = 30
    n_init: int = 2
    det_thresh: float = 0.5
    gallery_size: int = 64
    only_position_gate: bool = False
    class_aware: bool = True
    lambda_motion: float = 0.0

    def validate(self) -> None:
        if not 0.0 < self.max_cosine_distance <= 2.0:
            raise ValueError(
                f"max_cosine_distance must be in (0, 2], got {self.max_cosine_distance}"
            )
        if not 0.0 < self.max_iou_distance <= 1.0:
            raise ValueError(
                f"max_iou_distance must be in (0, 1], got {self.max_iou_distance}"
            )
        if self.n_init < 1:
            raise ValueError(f"n_init must be at least 1, got {self.n_init}")
        if not 0.0 <= self.lambda_motion <= 1.0:
            raise ValueError(
                f"lambda_motion must be in [0, 1], got {self.lambda_motion}"
            )


class DeepSort(BaseTracker):
    """Appearance-based tracker.  Every detection must carry a ``feature``."""

    def __init__(
        self, config: DeepSortConfig | None = None, kf: KalmanFilter | None = None
    ) -> None:
        super().__init__()
        self.config = config or DeepSortConfig()
        self.config.validate()
        self._kf = kf or KalmanFilter()
        self.last_association: dict[str, list[tuple[int, int]]] = {}

    def reset(self) -> None:
        super().reset()
        self.last_association = {}

    # -- cost -------------------------------------------------------------

    def _appearance_cost(
        self, tracks: Sequence[KalmanTrack], detections: Sequence[Detection]
    ) -> np.ndarray:
        """Smallest cosine distance to any descriptor in each track's gallery.

        Taking the minimum over the gallery rather than the mean is what makes
        the tracker robust to a track whose recent crops were all occluded: one
        clean descriptor from twenty frames ago is enough to recognise the
        object when it reappears.
        """
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float64)
        features = np.stack(
            [np.asarray(d.feature, dtype=np.float64).ravel() for d in detections]
        )
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float64)
        for i, track in enumerate(tracks):
            if not track.features:
                cost[i, :] = np.inf
                continue
            gallery = np.stack([f.ravel() for f in track.features])
            cost[i, :] = cosine_distance(features, gallery).min(axis=1)
        return cost

    def _gate(
        self,
        cost: np.ndarray,
        tracks: Sequence[KalmanTrack],
        detections: Sequence[Detection],
    ) -> np.ndarray:
        """Set impossible pairs to infinity.

        Two gates.  The Mahalanobis test rejects a detection that the motion
        model says the object cannot have reached, which is what stops a
        similarly coloured object on the far side of the frame from stealing an
        identity.  The class gate rejects cross-class association for the same
        reason it does in ByteTrack.
        """
        if cost.size == 0:
            return cost
        cfg = self.config
        measurements = np.stack([d.xyah for d in detections])
        dof = 2 if cfg.only_position_gate else 4
        threshold = CHI2_INV95[dof]
        out = cost.copy()
        for i, track in enumerate(tracks):
            distance = self._kf.gating_distance(
                track.mean, track.covariance, measurements, cfg.only_position_gate
            )
            out[i, distance > threshold] = np.inf
            if cfg.lambda_motion > 0:
                # The paper's weighted sum.  Off by default: on this dataset
                # the appearance term alone scored better, and the ablation is
                # in docs/tracking.md.
                blended = (
                    cfg.lambda_motion * (distance / threshold)
                    + (1.0 - cfg.lambda_motion) * out[i, :]
                )
                out[i, :] = np.where(np.isfinite(out[i, :]), blended, np.inf)
        if cfg.class_aware:
            track_classes = np.array([t.class_id for t in tracks])[:, None]
            det_classes = np.array([d.class_id for d in detections])[None, :]
            out = np.where(track_classes == det_classes, out, np.inf)
        return out

    # -- main loop --------------------------------------------------------

    def update(self, detections: Sequence[Detection], frame_id: int | None = None) -> list[Track]:
        cfg = self.config
        self.frame_id = self.frame_id + 1 if frame_id is None else int(frame_id)
        self.last_association = {"cascade": [], "iou": []}

        detections = [d for d in detections if d.score >= min(cfg.det_thresh, 1.0)]
        missing = [i for i, d in enumerate(detections) if d.feature is None]
        if missing:
            raise ValueError(
                "DeepSort needs an appearance descriptor on every detection; "
                f"{len(missing)} of {len(detections)} had none.  Run the detections "
                "through edgetrack.track.appearance.AppearanceEncoder first."
            )

        for track in self.tracks:
            track.predict()

        confirmed = [t for t in self.tracks if t.is_confirmed]
        unconfirmed = [t for t in self.tracks if not t.is_confirmed]

        # Stage 1: the matching cascade.
        unmatched_dets = list(range(len(detections)))
        matched: list[tuple[KalmanTrack, int]] = []
        for age in range(1, cfg.max_age + 1):
            if not unmatched_dets:
                break
            level = [t for t in confirmed if t.time_since_update == age]
            if not level:
                continue
            subset = [detections[i] for i in unmatched_dets]
            cost = self._gate(self._appearance_cost(level, subset), level, subset)
            pairs, _, leftover = linear_assignment(cost, cfg.max_cosine_distance)
            for ti, di in pairs:
                matched.append((level[ti], unmatched_dets[di]))
            unmatched_dets = [unmatched_dets[i] for i in leftover]

        # Anything confirmed that no level matched, including tracks older than
        # max_age which never entered a level at all, still has to be aged.
        matched_ids = {id(track) for track, _ in matched}
        unmatched_confirmed = [t for t in confirmed if id(t) not in matched_ids]

        for track, di in matched:
            track.update(detections[di], self.frame_id, cfg.n_init)
            self.last_association["cascade"].append((track.track_id, di))

        # Stage 2: IoU fallback for unconfirmed tracks and for confirmed tracks
        # that were seen last frame.  A track that has been missing for several
        # frames is deliberately excluded: recovering it is the appearance
        # branch's job and letting IoU do it reintroduces the failure mode the
        # cascade exists to prevent.
        iou_pool = unconfirmed + [t for t in unmatched_confirmed if t.time_since_update == 1]
        stale = [t for t in unmatched_confirmed if t.time_since_update != 1]
        subset = [detections[i] for i in unmatched_dets]
        cost = iou_distance(self._boxes(iou_pool), self._boxes(subset))
        if cfg.class_aware and cost.size:
            track_classes = np.array([t.class_id for t in iou_pool])[:, None]
            det_classes = np.array([d.class_id for d in subset])[None, :]
            cost = np.where(track_classes == det_classes, cost, 1.0)
        pairs, unmatched_pool, unmatched_subset = linear_assignment(cost, cfg.max_iou_distance)
        for ti, di in pairs:
            track = iou_pool[ti]
            track.update(subset[di], self.frame_id, cfg.n_init)
            self.last_association["iou"].append((track.track_id, unmatched_dets[di]))

        for i in unmatched_pool:
            iou_pool[i].mark_missed(cfg.max_age, cfg.n_init)
        for track in stale:
            track.mark_missed(cfg.max_age, cfg.n_init)

        # Stage 3: new tracks.
        for i in unmatched_subset:
            self.tracks.append(
                KalmanTrack(
                    self._kf,
                    self._allocate_id(),
                    subset[i],
                    self.frame_id,
                    feature_gallery=cfg.gallery_size,
                )
            )

        self.tracks = [t for t in self.tracks if not t.is_deleted]

        return [
            t.to_public()
            for t in self.tracks
            if t.is_confirmed and t.time_since_update == 0
        ]
