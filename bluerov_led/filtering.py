"""Online filtering utilities including Kalman Filters and Rolling Filters."""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from bluerov_led.config import VisionConfig


class KalmanFilter2D:
    """Constant Velocity 2D Kalman Filter for LED Centroid target tracking."""
    
    def __init__(
        self, 
        init_x: float, 
        init_y: float, 
        process_noise_pos: float = 1e-2, 
        process_noise_vel: float = 1e-3, 
        measurement_noise: float = 1.0
    ):
        # State vector: [x, y, vx, vy]^T
        self.X = np.array([init_x, init_y, 0.0, 0.0], dtype=float).reshape(4, 1)
        
        # State Covariance matrix (initially high uncertainty)
        self.P = np.eye(4) * 10.0
        
        # Measurement matrix (we observe x and y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # Process noise params
        self.q_p = process_noise_pos
        self.q_v = process_noise_vel

    def predict(self, dt: float) -> tuple[float, float]:
        """Predicts the next state given a dynamic delta_t."""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        Q = np.array([
            [self.q_p, 0, 0, 0],
            [0, self.q_p, 0, 0],
            [0, 0, self.q_v, 0],
            [0, 0, 0, self.q_v]
        ], dtype=float)
        
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q
        return self.X[0, 0], self.X[1, 0]

    def update(self, meas_x: float, meas_y: float) -> tuple[float, float]:
        """Corrects the state based on actual observations."""
        Z = np.array([meas_x, meas_y], dtype=float).reshape(2, 1)
        
        Y = Z - (self.H @ self.X)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.X = self.X + (K @ Y)
        I = np.eye(4)
        self.P = (I - (K @ self.H)) @ self.P
        
        return self.X[0, 0], self.X[1, 0]


class SignalSmoother1D:
    """1D discrete low-pass filter for measurement jitter (e.g. distance)."""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None
        
    def reset(self):
        self.value = None
        
    def update(self, measurement: float | None) -> float | None:
        if measurement is None:
            return self.value
            
        if self.value is None:
            self.value = measurement
        else:
            self.value = self.alpha * measurement + (1.0 - self.alpha) * self.value
            
        return self.value


def _smoothstep(x: float, t0: float, t1: float) -> float:
    if t1 <= t0:
        return 1.0 if x >= t1 else 0.0
    t = max(0.0, min(1.0, (x - t0) / (t1 - t0)))
    return t * t * (3.0 - 2.0 * t)


def mission_iqr_confidence(
    *,
    pattern_accuracy: float,
    pair_correlation: float,
    geometry_score: float,
    face_id: str | None,
    pattern: str | None,
    config: VisionConfig,
) -> float:
    """Composite confidence C_mission for IQR tiering."""
    acc = max(0.0, min(1.0, pattern_accuracy))
    rho = max(0.0, min(1.0, pair_correlation))
    g = max(0.0, min(1.0, geometry_score))
    c = acc * rho * g

    target = config.target_face_id
    if face_id != target:
        c *= config.iqr_wrong_face_penalty
        return c

    if config.iqr_require_target_pattern:
        expected = config.face_patterns.get(target)
        if expected and pattern != expected:
            c *= config.iqr_wrong_face_penalty

    return c


@dataclass
class IqrDecision:
    """Result of confidence-weighted outlier check."""

    accept: bool
    bypassed: bool
    effective_multiplier: float


class RollingIQRFilter:
    """Real-time sliding window IQR filter with confidence-weighted tiers."""
    
    def __init__(self, window_size: int, iqr_multiplier: float):
        self.window_size = window_size
        self.iqr_multiplier = iqr_multiplier
        self.buffer = collections.deque(maxlen=window_size)
    
    def _effective_multiplier(self, mission_confidence: float, config: VisionConfig) -> float:
        k_base = self.iqr_multiplier
        k_max = config.iqr_confidence_max_multiplier
        blend = _smoothstep(
            mission_confidence,
            config.iqr_confidence_t0,
            config.iqr_confidence_t1,
        )
        return k_base + (k_max - k_base) * blend

    def _should_bypass(
        self,
        mission_confidence: float,
        pattern_accuracy: float,
        face_id: str | None,
        config: VisionConfig,
    ) -> bool:
        if face_id != config.target_face_id:
            return False
        if pattern_accuracy < config.iqr_bypass_min_pattern_accuracy:
            return False
        return mission_confidence >= config.iqr_bypass_confidence_threshold

    def evaluate(
        self,
        val: float,
        *,
        config: VisionConfig,
        pattern_accuracy: float = 0.0,
        pair_correlation: float = 0.0,
        geometry_score: float = 0.0,
        face_id: str | None = None,
        pattern: str | None = None,
    ) -> IqrDecision:
        """
        Tier A: full bypass on high-confidence target-face decode.
        Tier B: widen IQR multiplier via smoothstep on C_mission.
        """
        mission_conf = mission_iqr_confidence(
            pattern_accuracy=pattern_accuracy,
            pair_correlation=pair_correlation,
            geometry_score=geometry_score,
            face_id=face_id,
            pattern=pattern,
            config=config,
        )

        if self._should_bypass(mission_conf, pattern_accuracy, face_id, config):
            return IqrDecision(
                accept=True,
                bypassed=True,
                effective_multiplier=float("inf"),
            )

        k_eff = self._effective_multiplier(mission_conf, config)

        if len(self.buffer) < 5:
            return IqrDecision(
                accept=True,
                bypassed=False,
                effective_multiplier=k_eff,
            )

        q1 = float(np.percentile(self.buffer, 25))
        q3 = float(np.percentile(self.buffer, 75))
        iqr = q3 - q1

        if iqr < 1e-3:
            return IqrDecision(
                accept=True,
                bypassed=False,
                effective_multiplier=k_eff,
            )

        lower_bound = q1 - k_eff * iqr
        upper_bound = q3 + k_eff * iqr
        accept = lower_bound <= val <= upper_bound

        return IqrDecision(
            accept=accept,
            bypassed=False,
            effective_multiplier=k_eff,
        )

    def add_valid(self, val: float):
        """Add verified valid observation into the rolling ring buffer constraint."""
        self.buffer.append(val)
        
    def reset(self):
        self.buffer.clear()
