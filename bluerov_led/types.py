"""Typed records for vision pipeline and controller packets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LedCandidate:
    x: int
    y: int
    w: int
    h: int
    cx: int
    cy: int
    area: float


@dataclass
class PairGeometry:
    led1_x: float
    led1_y: float
    led2_x: float
    led2_y: float
    pixel_distance: float
    mid_x: float
    mid_y: float
    error_x: float
    error_y: float
    ray_x: float
    ray_y: float
    ray_z: float


@dataclass
class FrameRecord:
    frame: int
    file: str
    candidate_count: int
    total_area: float
    bit: int
    pair_found: int
    led1_x: float | None
    led1_y: float | None
    led2_x: float | None
    led2_y: float | None
    pixel_distance: float | None
    mid_x: float | None
    mid_y: float | None
    error_x: float | None
    error_y: float | None
    ray_x: float | None
    ray_y: float | None
    ray_z: float | None
    camera_vertical_fov_deg: float
    image_width: int
    image_height: int
    face_id: str | None = None
    pattern: str | None = None
    pattern_accuracy: float | None = None
    active_track_count: int = 0
    track_id_1: int | None = None
    track_id_2: int | None = None
    pair_correlation: float | None = None
    pair_score: float | None = None
    geometry_score: float | None = None
    matcher_mode: str = "spatio_temporal"
    held: int = 0
    read_ok: int = 1

    def to_csv_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternSummary:
    dataset: str
    fps: int
    bit_duration_seconds: float
    frames_per_bit: int
    expected_pattern: str
    total_frame_count: int
    total_decoded_bit_count: int
    decoded_bits: str
    local_best_score: float
    local_best_start_index: int | None
    local_decoded_window: str | None
    local_matched_expected_shift: str | None
    global_accuracy: float
    bit_error_count: int
    bit_error_rate: float
    best_global_shift: str | None
    error_positions: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternSummary:
        return cls(
            dataset=data["dataset"],
            fps=data["fps"],
            bit_duration_seconds=data["bit_duration_seconds"],
            frames_per_bit=data["frames_per_bit"],
            expected_pattern=data["expected_pattern"],
            total_frame_count=data["total_frame_count"],
            total_decoded_bit_count=data["total_decoded_bit_count"],
            decoded_bits=data["decoded_bits"],
            local_best_score=data["local_best_score"],
            local_best_start_index=data.get("local_best_start_index"),
            local_decoded_window=data.get("local_decoded_window"),
            local_matched_expected_shift=data.get("local_matched_expected_shift"),
            global_accuracy=data["global_accuracy"],
            bit_error_count=data["bit_error_count"],
            bit_error_rate=data["bit_error_rate"],
            best_global_shift=data.get("best_global_shift"),
            error_positions=data.get("error_positions", []),
        )


@dataclass
class PipelineResult:
    dataset: str
    pair_csv: str
    pattern_json: str
    filtered_csv: str
    model_json: str
    packet_json: str
