"""Offline dataset validation metrics for the LED tracking pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from bluerov_led.config import (
    DEFAULT_CALIBRATION_POINTS,
    FACE_PATTERNS,
    ProjectPaths,
    VisionConfig,
)
from bluerov_led.dataset_io import ArtifactWriter, DatasetReader
from bluerov_led.distance_model import DistanceModel
from bluerov_led.pipeline import BackFacePipeline
from bluerov_led.temporal_decoder import TemporalDecoder


@dataclass
class ValidationThresholds:
    """Minimum acceptable metrics for back-only calibration datasets."""

    min_pair_recall_on_frames: float = 0.75
    min_face_id_accuracy_on_pairs: float = 0.95
    min_mean_pair_pattern_accuracy: float = 0.95
    min_temporal_decode_accuracy: float = 0.95
    max_distance_mae: float = 0.20
    min_median_pixel_distance_px: float = 20.0


@dataclass
class DatasetTestSpec:
    dataset_name: str
    expected_face_id: str
    expected_pattern: str
    ground_truth_distance_unit: float | None = None
    notes: str = ""


@dataclass
class DatasetValidationResult:
    dataset: str
    expected_face_id: str
    expected_pattern: str
    ground_truth_distance_unit: float | None
    frame_count: int
    warmup_frames_excluded: int
    eligible_on_frames: int
    pair_found_frames: int
    true_positive_pair_frames: int
    false_positive_face_frames: int
    pair_recall_on_frames: float
    face_id_accuracy_on_pairs: float
    mean_pair_pattern_accuracy: float
    temporal_decode_accuracy: float
    temporal_bit_error_rate: float
    median_pixel_distance_px: float | None
    estimated_distance_unit: float | None
    distance_mae: float | None
    distance_abs_error: float | None
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def specs_from_calibration_points() -> list[DatasetTestSpec]:
    specs: list[DatasetTestSpec] = []

    for point in DEFAULT_CALIBRATION_POINTS:
        name = point["test_name"]
        specs.append(
            DatasetTestSpec(
                dataset_name=name,
                expected_face_id="BACK",
                expected_pattern=FACE_PATTERNS["BACK"],
                ground_truth_distance_unit=float(point["distance_unit"]),
                notes=str(point.get("notes", "")),
            )
        )

    return specs


def discover_png_datasets(paths: ProjectPaths) -> list[str]:
    root = paths.project_root / paths.datasets_dir

    if not root.exists():
        return []

    names: list[str] = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        if list(folder.glob("*.png")):
            names.append(folder.name)

    return names


def _safe_str(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip()


def validate_dataset_csv(
    df: pd.DataFrame,
    spec: DatasetTestSpec,
    distance_model: DistanceModel,
    config: VisionConfig,
    thresholds: ValidationThresholds,
) -> DatasetValidationResult:
    warmup = config.min_decode_frames
    work = df[df["frame"] >= warmup].copy()

    eligible_on = work[work["bit"] == 1]
    eligible_on_count = len(eligible_on)

    paired = work[work["pair_found"] == 1]
    pair_found_count = len(paired)

    if "face_id" in paired.columns:
        face_ok = paired["face_id"].apply(
            lambda v: _safe_str(v) == spec.expected_face_id
        )
    else:
        face_ok = pd.Series([False] * len(paired))

    if "pattern_accuracy" in paired.columns:
        pattern_acc = paired["pattern_accuracy"].fillna(0.0).astype(float)
        tp_mask = face_ok & (pattern_acc >= config.min_pattern_accuracy)
    else:
        tp_mask = face_ok

    true_positive_count = int(tp_mask.sum())
    false_positive_face_count = int((~face_ok).sum()) if len(paired) else 0

    if eligible_on_count > 0:
        pair_recall = true_positive_count / eligible_on_count
    else:
        pair_recall = 0.0

    if pair_found_count > 0:
        face_id_accuracy = float(face_ok.mean())
        mean_pair_pattern_accuracy = float(
            paired["pattern_accuracy"].fillna(0.0).mean()
        ) if "pattern_accuracy" in paired.columns else 0.0
    else:
        face_id_accuracy = 0.0
        mean_pair_pattern_accuracy = 0.0

    frame_bits = df["bit"].astype(int).tolist()
    decoder = TemporalDecoder(config)
    pattern_summary = decoder.decode_dataset(
        spec.dataset_name,
        frame_bits,
        expected_pattern=spec.expected_pattern,
    )
    temporal_accuracy = float(pattern_summary.global_accuracy)
    temporal_bit_error_rate = float(pattern_summary.bit_error_rate)

    distance_mae = None
    distance_abs_error = None
    median_px = None
    estimated_distance = None

    if true_positive_count > 0:
        tp_rows = paired[tp_mask]
        median_px = float(tp_rows["pixel_distance"].median())
        estimated_distance = distance_model.estimate(median_px)

        if spec.ground_truth_distance_unit is not None and estimated_distance is not None:
            distance_abs_error = abs(estimated_distance - spec.ground_truth_distance_unit)
            distance_mae = distance_abs_error

    failures: list[str] = []

    if eligible_on_count == 0:
        failures.append("No eligible ON frames after warmup.")

    if pair_recall < thresholds.min_pair_recall_on_frames:
        failures.append(
            f"Pair recall on ON frames {pair_recall:.3f} < "
            f"{thresholds.min_pair_recall_on_frames:.3f}"
        )

    if pair_found_count > 0 and face_id_accuracy < thresholds.min_face_id_accuracy_on_pairs:
        failures.append(
            f"Face ID accuracy {face_id_accuracy:.3f} < "
            f"{thresholds.min_face_id_accuracy_on_pairs:.3f}"
        )

    if (
        pair_found_count > 0
        and mean_pair_pattern_accuracy < thresholds.min_mean_pair_pattern_accuracy
    ):
        failures.append(
            f"Mean pair pattern accuracy {mean_pair_pattern_accuracy:.3f} < "
            f"{thresholds.min_mean_pair_pattern_accuracy:.3f}"
        )

    if temporal_accuracy < thresholds.min_temporal_decode_accuracy:
        failures.append(
            f"Temporal decode accuracy {temporal_accuracy:.3f} < "
            f"{thresholds.min_temporal_decode_accuracy:.3f}"
        )

    if distance_mae is not None and distance_mae > thresholds.max_distance_mae:
        failures.append(
            f"Distance MAE {distance_mae:.3f} > {thresholds.max_distance_mae:.3f}"
        )

    if median_px is not None and median_px < thresholds.min_median_pixel_distance_px:
        failures.append(
            f"Median pixel distance {median_px:.1f} < "
            f"{thresholds.min_median_pixel_distance_px:.1f}"
        )

    return DatasetValidationResult(
        dataset=spec.dataset_name,
        expected_face_id=spec.expected_face_id,
        expected_pattern=spec.expected_pattern,
        ground_truth_distance_unit=spec.ground_truth_distance_unit,
        frame_count=len(df),
        warmup_frames_excluded=warmup,
        eligible_on_frames=eligible_on_count,
        pair_found_frames=pair_found_count,
        true_positive_pair_frames=true_positive_count,
        false_positive_face_frames=false_positive_face_count,
        pair_recall_on_frames=pair_recall,
        face_id_accuracy_on_pairs=face_id_accuracy,
        mean_pair_pattern_accuracy=mean_pair_pattern_accuracy,
        temporal_decode_accuracy=temporal_accuracy,
        temporal_bit_error_rate=temporal_bit_error_rate,
        median_pixel_distance_px=median_px,
        estimated_distance_unit=estimated_distance,
        distance_mae=distance_mae,
        distance_abs_error=distance_abs_error,
        passed=len(failures) == 0,
        failure_reasons=failures,
    )


class ValidationRunner:
    """Run extract + metrics across offline PNG datasets."""

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        config: VisionConfig | None = None,
        thresholds: ValidationThresholds | None = None,
    ) -> None:
        self.paths = paths or ProjectPaths()
        self.config = config or VisionConfig(matcher_mode="spatio_temporal")
        self.thresholds = thresholds or ValidationThresholds()
        self.pipeline = BackFacePipeline(config=self.config, paths=self.paths)
        self.distance_model = DistanceModel.fit()

    def resolve_specs(self, datasets: list[str] | None) -> list[DatasetTestSpec]:
        known = {s.dataset_name: s for s in specs_from_calibration_points()}

        if datasets:
            names = datasets
        else:
            names = discover_png_datasets(self.paths)

        specs: list[DatasetTestSpec] = []

        for name in names:
            if name in known:
                specs.append(known[name])
            elif name.startswith("BackOnly_"):
                specs.append(
                    DatasetTestSpec(
                        dataset_name=name,
                        expected_face_id="BACK",
                        expected_pattern=FACE_PATTERNS["BACK"],
                    )
                )

        return specs

    def run_extract(self, spec: DatasetTestSpec, use_cache: bool) -> Path:
        csv_path = self.paths.pair_csv(spec.dataset_name)

        if use_cache and csv_path.exists():
            print(f"  Using cached CSV: {csv_path}")
            return csv_path

        dataset_folder = self.paths.dataset_folder(spec.dataset_name)

        if not dataset_folder.exists() or not list(dataset_folder.glob("*.png")):
            raise FileNotFoundError(
                f"No PNG frames found for dataset: {dataset_folder}"
            )

        print(f"  Running extract on {spec.dataset_name} ...")
        return self.pipeline.extract(spec.dataset_name, preview=False)

    def validate_one(
        self,
        spec: DatasetTestSpec,
        use_cache: bool,
    ) -> DatasetValidationResult:
        csv_path = self.run_extract(spec, use_cache=use_cache)
        df = ArtifactWriter.read_frame_records_csv(csv_path)
        return validate_dataset_csv(
            df=df,
            spec=spec,
            distance_model=self.distance_model,
            config=self.config,
            thresholds=self.thresholds,
        )

    def run_all(
        self,
        datasets: list[str] | None = None,
        use_cache: bool = False,
    ) -> list[DatasetValidationResult]:
        specs = self.resolve_specs(datasets)

        if not specs:
            raise RuntimeError("No datasets found to validate.")

        results: list[DatasetValidationResult] = []

        for spec in specs:
            print(f"\nValidating {spec.dataset_name} ...")
            result = self.validate_one(spec, use_cache=use_cache)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] pair_recall={result.pair_recall_on_frames:.3f} "
                  f"face_acc={result.face_id_accuracy_on_pairs:.3f} "
                  f"temporal={result.temporal_decode_accuracy:.3f} "
                  f"dist_mae={result.distance_mae}")

            if result.failure_reasons:
                for reason in result.failure_reasons:
                    print(f"    - {reason}")

        return results

    def write_report(
        self,
        results: list[DatasetValidationResult],
        report_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        if report_dir is None:
            report_dir = self.paths.project_root / self.paths.outputs_dir / "validation"

        ArtifactWriter.ensure_dir(report_dir)

        json_path = report_dir / "validation_summary.json"
        csv_path = report_dir / "validation_results.csv"

        payload = {
            "matcher_mode": self.config.matcher_mode,
            "thresholds": asdict(self.thresholds),
            "all_passed": all(r.passed for r in results),
            "results": [r.to_dict() for r in results],
        }

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        pd.DataFrame([r.to_dict() for r in results]).to_csv(csv_path, index=False)

        return json_path, csv_path
