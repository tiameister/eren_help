"""Inverse linear distance calibration model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bluerov_led.config import DEFAULT_CALIBRATION_POINTS


@dataclass
class DistanceModel:
    """estimated_distance = A / pixel_distance + B"""

    A: float
    B: float
    mae: float
    rmse: float
    calibration_points: list[dict]

    def estimate(self, pixel_distance: float | None) -> float | None:
        if pixel_distance is None or pixel_distance <= 0:
            return None
        return self.A * (1.0 / pixel_distance) + self.B

    def confidence(self, pixel_distance: float | None, pattern_accuracy: float) -> float:
        if pixel_distance is None or pixel_distance <= 0:
            return 0.0

        if pixel_distance >= 70:
            pixel_conf = 1.0
        elif pixel_distance >= 50:
            pixel_conf = 0.85
        elif pixel_distance >= 35:
            pixel_conf = 0.65
        else:
            pixel_conf = 0.40

        return round(pixel_conf * pattern_accuracy, 3)

    def to_summary_dict(self) -> dict:
        return {
            "model_type": "inverse_linear",
            "formula": "estimated_distance = A / pixel_distance + B",
            "A": self.A,
            "B": self.B,
            "mae": self.mae,
            "rmse": self.rmse,
            "calibration_points": self.calibration_points,
        }

    @classmethod
    def from_summary_dict(cls, data: dict) -> DistanceModel:
        return cls(
            A=float(data["A"]),
            B=float(data["B"]),
            mae=float(data.get("mae", 0.0)),
            rmse=float(data.get("rmse", 0.0)),
            calibration_points=data.get("calibration_points", []),
        )

    @classmethod
    def fit(
        cls,
        points: list[dict] | None = None,
    ) -> DistanceModel:
        raw = DEFAULT_CALIBRATION_POINTS if points is None else points

        px = np.array([p["median_px"] for p in raw], dtype=float)
        distance = np.array([p["distance_unit"] for p in raw], dtype=float)
        x = 1.0 / px
        A, B = np.polyfit(x, distance, 1)

        eval_rows = []
        errors = []
        for p in raw:
            predicted = float(A) * (1.0 / p["median_px"]) + float(B)
            error = p["distance_unit"] - predicted
            errors.append(error)
            eval_rows.append(
                {
                    "test_name": p["test_name"],
                    "real_distance_unit": p["distance_unit"],
                    "median_px": p["median_px"],
                    "estimated_distance_unit": predicted,
                    "error": error,
                    "abs_error": abs(error),
                    "pattern_accuracy": p["pattern_accuracy"],
                    "filtered_std": p["filtered_std"],
                    "notes": p["notes"],
                }
            )

        df_eval = pd.DataFrame(eval_rows)
        mae = float(df_eval["abs_error"].mean())
        rmse = float(np.sqrt((df_eval["error"] ** 2).mean()))

        return cls(
            A=float(A),
            B=float(B),
            mae=mae,
            rmse=rmse,
            calibration_points=raw,
        )

    def evaluation_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self.calibration_points:
            predicted = self.estimate(p["median_px"])
            error = p["distance_unit"] - predicted
            rows.append(
                {
                    "test_name": p["test_name"],
                    "real_distance_unit": p["distance_unit"],
                    "median_px": p["median_px"],
                    "estimated_distance_unit": predicted,
                    "error": error,
                    "abs_error": abs(error),
                    "pattern_accuracy": p["pattern_accuracy"],
                    "filtered_std": p["filtered_std"],
                    "notes": p["notes"],
                }
            )
        return pd.DataFrame(rows)

    def print_report(self) -> None:
        print("Distance model:")
        print(f"estimated_distance = {self.A:.6f} / pixel_distance + {self.B:.6f}")
        print("\nEvaluation:")
        df = self.evaluation_dataframe()
        print(
            df[
                [
                    "test_name",
                    "real_distance_unit",
                    "median_px",
                    "estimated_distance_unit",
                    "error",
                    "abs_error",
                    "pattern_accuracy",
                    "filtered_std",
                ]
            ]
        )
        print("\nMean absolute error:", self.mae)
        print("RMSE:", self.rmse)
