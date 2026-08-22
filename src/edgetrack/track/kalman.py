"""Constant-velocity Kalman filter on box centre, aspect ratio and height.

State is eight dimensional, ``[cx, cy, a, h, vcx, vcy, va, vh]``, and the
measurement is the first four.  Aspect ratio rather than width is tracked
because height is the quantity that scales with distance in a fixed camera,
so tying the process noise to height gives a filter whose uncertainty grows
with the size of the object rather than uniformly across the frame.  That is
the same parameterisation SORT and DeepSORT use and it is why boxes are
converted to ``xyah`` at the tracker boundary rather than everywhere.

Both trackers in this package share this filter.  Keeping one implementation
means a difference between ByteTrack and DeepSORT results is a difference in
association, not a difference in motion modelling.
"""

from __future__ import annotations

import numpy as np

__all__ = ["KalmanFilter", "CHI2_INV95"]

# Inverse chi-square cumulative distribution at 0.95, indexed by degrees of
# freedom 1 to 9.  Used for the Mahalanobis gate in DeepSORT.  Hard-coded so
# that scipy.stats is not a runtime import.
CHI2_INV95: dict[int, float] = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """Linear Kalman filter with a constant-velocity motion model.

    Parameters
    ----------
    std_position:
        Position process and measurement noise as a fraction of box height.
    std_velocity:
        Velocity process noise as a fraction of box height.
    """

    def __init__(self, std_position: float = 1.0 / 20, std_velocity: float = 1.0 / 160) -> None:
        ndim, dt = 4, 1.0
        self._motion = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion[i, ndim + i] = dt
        self._update = np.eye(ndim, 2 * ndim, dtype=np.float64)
        self.std_position = float(std_position)
        self.std_velocity = float(std_velocity)

    # -- noise ------------------------------------------------------------

    def _process_noise(self, mean: np.ndarray) -> np.ndarray:
        h = max(float(mean[3]), 1.0)
        std = np.array(
            [
                self.std_position * h,
                self.std_position * h,
                1e-2,
                self.std_position * h,
                self.std_velocity * h,
                self.std_velocity * h,
                1e-5,
                self.std_velocity * h,
            ],
            dtype=np.float64,
        )
        return np.diag(np.square(std))

    def _measurement_noise(self, mean: np.ndarray, confidence: float = 1.0) -> np.ndarray:
        h = max(float(mean[3]), 1.0)
        # A low-confidence detection is a noisier measurement.  ByteTrack's
        # second association stage feeds exactly those in, so scaling the
        # measurement noise by confidence keeps a weak detection from yanking
        # the state as hard as a strong one.
        scale = 1.0 / max(float(confidence), 0.1)
        std = np.array(
            [
                self.std_position * h * scale,
                self.std_position * h * scale,
                1e-1,
                self.std_position * h * scale,
            ],
            dtype=np.float64,
        )
        return np.diag(np.square(std))

    # -- filter -----------------------------------------------------------

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Start a track from a first measurement, with zero initial velocity."""
        measurement = np.asarray(measurement, dtype=np.float64).reshape(4)
        mean = np.concatenate([measurement, np.zeros(4, dtype=np.float64)])
        h = max(float(measurement[3]), 1.0)
        std = np.array(
            [
                2 * self.std_position * h,
                2 * self.std_position * h,
                1e-2,
                2 * self.std_position * h,
                10 * self.std_velocity * h,
                10 * self.std_velocity * h,
                1e-5,
                10 * self.std_velocity * h,
            ],
            dtype=np.float64,
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noise = self._process_noise(mean)
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + noise
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray, confidence: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project state into measurement space."""
        projected_mean = self._update @ mean
        projected_cov = self._update @ covariance @ self._update.T
        return projected_mean, projected_cov + self._measurement_noise(mean, confidence)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standard Kalman correction step."""
        measurement = np.asarray(measurement, dtype=np.float64).reshape(4)
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        # Solving rather than inverting: numerically better and no slower at
        # this size.  The transpose dance is because we want K = P H' S^-1 and
        # np.linalg.solve gives us S^-1 X, so solve for the transpose.
        cross = covariance @ self._update.T
        gain = np.linalg.solve(projected_cov.T, cross.T).T
        innovation = measurement - projected_mean
        new_mean = mean + gain @ innovation
        new_cov = covariance - gain @ projected_cov @ gain.T
        # Force symmetry.  Repeated predict/update on float64 drifts by about
        # 1e-12 per step, which is harmless until a Cholesky is attempted on it.
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Squared Mahalanobis distance from the state to each measurement.

        Compare against :data:`CHI2_INV95` at four degrees of freedom, or two
        when ``only_position`` is set.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        measurements = np.asarray(measurements, dtype=np.float64).reshape(-1, 4)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]
        if measurements.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        delta = measurements - projected_mean
        try:
            chol = np.linalg.cholesky(projected_cov)
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by symmetry above
            return np.full(measurements.shape[0], np.inf, dtype=np.float64)
        solved = np.linalg.solve(chol, delta.T)
        return np.sum(solved * solved, axis=0)


def xyah_to_xyxy(mean: np.ndarray) -> tuple[float, float, float, float]:
    """Convert the first four state components back to a pixel box."""
    cx, cy, a, h = (float(v) for v in np.asarray(mean, dtype=np.float64)[:4])
    w = a * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
