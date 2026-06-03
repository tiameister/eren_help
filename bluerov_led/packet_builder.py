"""Controller-ready JSON observation packet builder."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bluerov_led.config import VisionConfig
from bluerov_led.distance_model import DistanceModel
from bluerov_led.types import PatternSummary


class ObservationPacketBuilder:
    """Build observation dicts matching the Linux controller schema."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def _row_pattern_accuracy(
        self, row: pd.Series, fallback: float
    ) -> float:
        if "pattern_accuracy" in row.index and pd.notna(row.get("pattern_accuracy")):
            return float(row["pattern_accuracy"])
        return fallback

    def is_valid_row(self, row: pd.Series, fallback_pattern_accuracy: float) -> bool:
        required_columns = [
            "bit",
            "pair_found",
            "pixel_distance",
            "mid_x",
            "mid_y",
            "error_x",
            "error_y",
            "ray_x",
            "ray_y",
            "ray_z",
        ]

        for col in required_columns:
            if col not in row.index:
                return False

        if int(row["bit"]) != 1:
            return False

        if int(row["pair_found"]) != 1:
            return False

        if self.config.matcher_mode == "legacy_largest2":
            if int(row["candidate_count"]) != 2:
                return False
        else:
            face_id = row.get("face_id")
            if face_id is None or (isinstance(face_id, float) and pd.isna(face_id)):
                return False
            if str(face_id).strip() == "":
                return False

        if pd.isna(row["pixel_distance"]):
            return False

        if float(row["pixel_distance"]) < self.config.min_pixel_distance:
            return False

        accuracy = self._row_pattern_accuracy(row, fallback_pattern_accuracy)
        if accuracy < self.config.min_pattern_accuracy:
            return False

        return True

    def select_observation_row(
        self, valid_df: pd.DataFrame, target_frame: int | None
    ) -> tuple[pd.Series | None, int | None]:
        if len(valid_df) == 0:
            return None, None

        if target_frame is None:
            selected = valid_df.iloc[0]
            return selected, 0

        work = valid_df.copy()
        work["frame_delta"] = (work["frame"] - target_frame).abs()
        selected = work.sort_values(["frame_delta", "frame"]).iloc[0]
        return selected, int(selected["frame_delta"])

    def build_packet(
        self,
        dataset: str,
        df: pd.DataFrame,
        pattern_summary: PatternSummary | None,
        distance_model: DistanceModel,
        target_frame: int | None = None,
    ) -> dict[str, Any]:
        if pattern_summary is None:
            fallback_accuracy = 1.0
            fallback_pattern = "11001100"
            bit_error_rate = 0.0
            bit_error_count = 0
        else:
            fallback_accuracy = float(pattern_summary.global_accuracy)
            fallback_pattern = pattern_summary.expected_pattern
            bit_error_rate = float(pattern_summary.bit_error_rate)
            bit_error_count = int(pattern_summary.bit_error_count)

        valid_mask = df.apply(
            lambda row: self.is_valid_row(row, fallback_accuracy),
            axis=1,
        )
        valid_df = df[valid_mask].copy()

        selected_row, selected_frame_delta = self.select_observation_row(
            valid_df, target_frame
        )

        if selected_row is None:
            return {
                "dataset": dataset,
                "requested_frame": target_frame,
                "selected_frame_delta": None,
                "frame": None,
                "valid": False,
                "face_id": None,
                "reason": "No valid observation row found.",
                "pattern": fallback_pattern,
                "pattern_accuracy": fallback_accuracy,
                "bit_error_count": bit_error_count,
                "bit_error_rate": bit_error_rate,
            }

        row = selected_row
        pixel_distance = float(row["pixel_distance"])
        pattern_accuracy = self._row_pattern_accuracy(row, fallback_accuracy)

        face_id = str(row["face_id"]) if pd.notna(row.get("face_id")) else None
        if pd.notna(row.get("pattern")):
            raw_pattern = row["pattern"]
            if isinstance(raw_pattern, (int, float)):
                pattern = str(int(raw_pattern))
            else:
                pattern = str(raw_pattern)
        elif face_id:
            pattern = self.config.face_patterns.get(face_id, fallback_pattern)
        else:
            pattern = fallback_pattern

        estimated_distance = distance_model.estimate(pixel_distance)
        dist_conf = distance_model.confidence(pixel_distance, pattern_accuracy)

        return {
            "dataset": dataset,
            "requested_frame": target_frame,
            "selected_frame_delta": selected_frame_delta,
            "frame": int(row["frame"]),
            "valid": True,
            "face_id": face_id,
            "pattern": pattern,
            "pattern_accuracy": pattern_accuracy,
            "bit_error_count": bit_error_count,
            "bit_error_rate": bit_error_rate,
            "pair_found": bool(row["pair_found"]),
            "candidate_count": int(row["candidate_count"]),
            "led1_px": [float(row["led1_x"]), float(row["led1_y"])],
            "led2_px": [float(row["led2_x"]), float(row["led2_y"])],
            "midpoint_px": [float(row["mid_x"]), float(row["mid_y"])],
            "error_norm": [float(row["error_x"]), float(row["error_y"])],
            "ray_cam": [
                float(row["ray_x"]),
                float(row["ray_y"]),
                float(row["ray_z"]),
            ],
            "pixel_distance": pixel_distance,
            "estimated_distance": estimated_distance,
            "distance_confidence": dist_conf,
            "image_size": [
                int(row["image_width"]),
                int(row["image_height"]),
            ],
        }
