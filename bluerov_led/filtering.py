"""Online filtering utilities including Kalman Filters and Rolling Filters."""

import collections
import numpy as np


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


class RollingIQRFilter:
    """Real-time sliding window IQR filter for outlier rejection (online mode)."""
    
    def __init__(self, window_size: int, iqr_multiplier: float):
        self.window_size = window_size
        self.iqr_multiplier = iqr_multiplier
        self.buffer = collections.deque(maxlen=window_size)
    
    def is_outlier(self, val: float) -> bool:
        """Determines if a value is an outlier compared to the recent rolling window."""
        if len(self.buffer) < 5:  # Not enough statistics strictly
            return False
            
        # Standard numpy percentile
        q1 = np.percentile(self.buffer, 25)
        q3 = np.percentile(self.buffer, 75)
        iqr = q3 - q1
        
        # We can be slightly loose for initial values
        if iqr < 1e-3:
            return False  # Values are identically flat
            
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        return not (lower_bound <= val <= upper_bound)
        
    def add_valid(self, val: float):
        """Add verified valid observation into the rolling ring buffer constraint."""
        self.buffer.append(val)
        
    def reset(self):
        self.buffer.clear()
