"""Project paths, face patterns, and vision pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Unity RovLeds.cs pattern strings (same phase for both LEDs on a face).
FACE_PATTERNS: dict[str, str] = {
    "FRONT": "11110000",
    "BACK": "11001100",
    "LEFT": "10101010",
    "RIGHT": "10011001",
}


@dataclass
class ProjectPaths:
    """Standard dataset and output directory layout."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    datasets_dir: str = "datasets"
    outputs_dir: str = "outputs"

    def dataset_folder(self, dataset_name: str) -> Path:
        return self.project_root / self.datasets_dir / dataset_name

    def output_folder(self, dataset_name: str) -> Path:
        return self.project_root / self.outputs_dir / dataset_name

    def calibration_folder(self) -> Path:
        return self.project_root / self.outputs_dir / "calibration"

    def pair_csv(self, dataset_name: str) -> Path:
        return self.output_folder(dataset_name) / "back_pair_results.csv"

    def pattern_summary_json(self, dataset_name: str) -> Path:
        return self.output_folder(dataset_name) / "back_pattern_decode_summary.json"

    def filtered_distance_csv(self, dataset_name: str) -> Path:
        return self.output_folder(dataset_name) / "back_pair_distance_filtered.csv"

    def distance_model_json(self) -> Path:
        return self.calibration_folder() / "distance_model_summary.json"

    def observation_packet_json(
        self, dataset_name: str, frame: int | None = None
    ) -> Path:
        folder = self.output_folder(dataset_name)
        if frame is None:
            return folder / "observation_packet_sample.json"
        return folder / f"observation_packet_frame_{frame}.json"


@dataclass
class VisionConfig:
    """LED detection, tracking, and spatio-temporal matching parameters."""

    fps: int = 60
    bit_duration_seconds: float = 0.1

    face_patterns: dict[str, str] = field(default_factory=lambda: dict(FACE_PATTERNS))

    lower_hsv: tuple[int, int, int] = (54, 83, 172)
    upper_hsv: tuple[int, int, int] = (95, 147, 226)

    min_area: float = 15.0
    max_area: float = 10000.0
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 4.50
    on_area_threshold: float = 28.0

    camera_vertical_fov_deg: float = 60.0
    display_scale: float = 0.5

    min_pattern_accuracy: float = 0.95
    min_pixel_distance: float = 20.0

    # Mission target for lock-on and confidence-weighted IQR
    target_face_id: str = "BACK"

    # CONSTANT_ON test mode (MP4 only): spatial pairing without blink decode
    bypass_temporal_decode: bool = False
    constant_on_min_geometry_score: float = 0.85

    # Matcher mode: spatio_temporal (default) or legacy_largest2
    matcher_mode: str = "spatio_temporal"

    # Centroid tracker
    max_match_distance_px: float = 115.0
    max_missed_frames: int = 32
    min_track_age_frames: int = 2

    # Per-track temporal buffers
    signal_buffer_maxlen: int = 60
    geometry_history_maxlen: int = 30
    min_pair_frames: int = 6
    min_decode_frames: int = 48

    # Correlation and face decode
    min_pair_correlation: float = 0.83
    min_pair_score: float = 0.60
    prefer_matched_tracks: bool = True
    high_correlation_fallback: float = 0.89

    # Geometry sanity
    geometry_window_frames: int = 15
    max_pixel_distance_cv: float = 0.20
    min_pixel_distance_px: float = 48.0
    max_pixel_distance_px: float = 300.0
    max_y_alignment_ratio: float = 0.60
    min_area_similarity: float = 0.28
    max_midpoint_jump_px: float = 48.0

    # Dynamic geometry (area-proportional distance gates)
    dynamic_min_distance_k_r: float = 0.88
    min_pixel_distance_far_px: float = 22.0
    near_field_area_threshold: float = 500.0
    dynamic_min_distance_soft_margin_px: float = 8.0
    # Optional cap scaled by blob area; disabled by default (min-only path is production).
    dynamic_max_distance_enabled: bool = False
    dynamic_max_distance_d_ref: float = 168.0
    dynamic_max_distance_r_ref: float = 60.0
    dynamic_max_distance_alpha: float = 0.5

    # Confidence-weighted IQR
    iqr_confidence_max_multiplier: float = 10.0
    iqr_confidence_t0: float = 0.55
    iqr_confidence_t1: float = 0.85
    iqr_bypass_confidence_threshold: float = 0.88
    iqr_bypass_min_pattern_accuracy: float = 0.95
    iqr_wrong_face_penalty: float = 0.5
    iqr_require_target_pattern: bool = True

    # Lock-on track pair state
    lock_acquire_frames: int = 3
    lock_release_miss_frames: int = 8
    lock_score_boost: float = 2.0
    lock_min_pattern_accuracy: float = 0.95
    lock_min_pair_correlation: float = 0.85

    # Legacy largest2 (Phase 1 parity)
    require_exactly_two_candidates: bool = False
    pair_strategy: str = "largest2"

    # Online Stream / Kalman Filter Configurations
    kf_process_noise_pos: float = 0.5
    kf_process_noise_vel: float = 0.05
    kf_measurement_noise_pos: float = 0.5
    rolling_iqr_window: int = 50
    outlier_distance_rejection_iqr_multiplier: float = 3.5
    signal_1d_lpf_alpha: float = 0.3
    max_hold_frames: int = 15

    @property
    def frames_per_bit(self) -> int:
        return int(self.fps * self.bit_duration_seconds)

    @property
    def lower_hsv_array(self) -> np.ndarray:
        return np.array(self.lower_hsv, dtype=np.uint8)

    @property
    def upper_hsv_array(self) -> np.ndarray:
        return np.array(self.upper_hsv, dtype=np.uint8)

    def match_distance_for_image(self, image_width: int, image_height: int) -> float:
        diagonal = (image_width ** 2 + image_height ** 2) ** 0.5
        return max(self.max_match_distance_px, 0.03 * diagonal)


DEFAULT_CALIBRATION_POINTS: list[dict] = [
    {
        "test_name": "BackOnly_Test_01",
        "distance_unit": 1.47,
        "median_px": 168.0,
        "pattern_accuracy": 1.00,
        "filtered_std": 0.61,
        "min_pair_recall_override": 0.65,
        "max_distance_mae_override": 0.22,
        "notes": "Initial static reference",
    },
    {
        "test_name": "BackOnly_Test_02",
        "distance_unit": 2.00,
        "median_px": 118.0,
        "pattern_accuracy": 1.00,
        "filtered_std": 0.002,
        "notes": "Camera moved backward; Y/Z changed slightly",
    },
    {
        "test_name": "BackOnly_Test_03",
        "distance_unit": 2.50,
        "median_px": 92.00543462209176,
        "pattern_accuracy": 1.00,
        "filtered_std": 0.74,
        "notes": "Camera moved backward; Y/Z changed slightly",
    },
    {
        "test_name": "BackOnly_Test_04",
        "distance_unit": 3.00,
        "median_px": 73.06161783043132,
        "pattern_accuracy": 1.00,
        "filtered_std": 0.82,
        "notes": "Camera moved only along X axis; Y/Z fixed",
    },
    {
        "test_name": "BackOnly_Test_05",
        "distance_unit": 4.00,
        "median_px": 53.0,
        "pattern_accuracy": 1.00,
        "filtered_std": 0.67,
        "max_distance_mae_override": 0.32,
        "notes": "Camera moved only along X axis; Y/Z fixed",
    },
    {
        "test_name": "BackOnly_Test_06",
        "distance_unit": 5.00,
        "median_px": 35.0,
        "pattern_accuracy": 0.99,
        "filtered_std": 2.24,
        "min_pair_recall_override": 0.42,
        "max_distance_mae_override": 0.35,
        "notes": "Far-range boundary test",
    },
]
