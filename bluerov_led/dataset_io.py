"""PNG dataset reading and artifact read/write."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from bluerov_led.types import FrameRecord, PatternSummary


def natural_sort_key(path: Path) -> list:
    text = str(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


class DatasetReader:
    """Reads PNG frame sequences from datasets/."""

    def __init__(self, dataset_folder: Path, glob_pattern: str = "*.png") -> None:
        self.dataset_folder = dataset_folder
        self.glob_pattern = glob_pattern

    def list_frame_paths(self) -> list[Path]:
        paths = sorted(
            self.dataset_folder.glob(self.glob_pattern),
            key=natural_sort_key,
        )
        if not paths:
            raise FileNotFoundError(
                f"No PNG files found in {self.dataset_folder}. "
                "Check dataset path and glob pattern."
            )
        return paths


class ArtifactWriter:
    """Writes pipeline CSV/JSON artifacts to outputs/."""

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def write_frame_records_csv(path: Path, records: list[FrameRecord]) -> Path:
        ArtifactWriter.ensure_dir(path.parent)
        df = pd.DataFrame([r.to_csv_row() for r in records])
        df.to_csv(path, index=False)
        return path

    @staticmethod
    def read_frame_records_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def write_json(path: Path, data: dict) -> Path:
        ArtifactWriter.ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return path

    @staticmethod
    def read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def write_pattern_summary(path: Path, summary: PatternSummary) -> Path:
        return ArtifactWriter.write_json(path, summary.to_dict())

    @staticmethod
    def read_pattern_summary(path: Path) -> PatternSummary | None:
        data = ArtifactWriter.read_json(path)
        if data is None:
            return None
        return PatternSummary.from_dict(data)

    @staticmethod
    def write_dataframe_csv(path: Path, df: pd.DataFrame) -> Path:
        ArtifactWriter.ensure_dir(path.parent)
        df.to_csv(path, index=False)
        return path
