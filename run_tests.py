#!/usr/bin/env python3
"""
Auto-validation suite (Phase 3) for the BlueROV2 LED tracking pipeline.

Runs the spatio-temporal pipeline on offline PNG datasets and reports:
  - Pair recall on ON frames (true-positive pair detection proxy)
  - Face ID accuracy on paired frames
  - Temporal decode accuracy (global repeated-pattern match)
  - Distance MAE vs calibration ground truth

Examples:
  python run_tests.py
  python run_tests.py --dataset BackOnly_Test_04
  python run_tests.py --use-cache
  python run_tests.py --matcher legacy_largest2 --dataset BackOnly_Test_04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bluerov_led.config import ProjectPaths, VisionConfig
from bluerov_led.validation import ValidationRunner, ValidationThresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline validation metrics on PNG datasets.",
    )

    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset folder name under datasets/. Repeatable. Default: all PNG datasets.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip extract if outputs/<dataset>/back_pair_results.csv exists.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Always re-run extract before validating (Phase 4 tuning).",
    )
    parser.add_argument(
        "--matcher",
        choices=["spatio_temporal", "legacy_largest2"],
        default="spatio_temporal",
        help="Matcher used during extract.",
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
    )
    parser.add_argument(
        "--min-pair-recall",
        type=float,
        default=0.75,
        help="Minimum TP pair recall on ON frames (after warmup).",
    )
    parser.add_argument(
        "--min-temporal-accuracy",
        type=float,
        default=0.95,
        help="Minimum global temporal decode accuracy.",
    )
    parser.add_argument(
        "--max-distance-mae",
        type=float,
        default=0.22,
        help="Maximum allowed distance MAE in Unity units.",
    )

    return parser


def print_summary_table(results) -> None:
    header = (
        f"{'Dataset':<22} {'Status':<6} {'PairRec':>8} {'FaceAcc':>8} "
        f"{'Temporal':>8} {'DistMAE':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    for result in results:
        dist_mae = (
            f"{result.distance_mae:.3f}"
            if result.distance_mae is not None
            else "n/a"
        )
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.dataset:<22} {status:<6} "
            f"{result.pair_recall_on_frames:8.3f} "
            f"{result.face_id_accuracy_on_pairs:8.3f} "
            f"{result.temporal_decode_accuracy:8.3f} "
            f"{dist_mae:>8}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    paths = ProjectPaths(
        project_root=root,
        datasets_dir=args.datasets_dir,
        outputs_dir=args.outputs_dir,
    )

    config = VisionConfig(matcher_mode=args.matcher)
    if args.matcher == "legacy_largest2":
        config.require_exactly_two_candidates = True

    thresholds = ValidationThresholds(
        min_pair_recall_on_frames=args.min_pair_recall,
        min_temporal_decode_accuracy=args.min_temporal_accuracy,
        max_distance_mae=args.max_distance_mae,
    )

    runner = ValidationRunner(paths=paths, config=config, thresholds=thresholds)

    print("BlueROV2 LED Tracking - Validation Suite")
    print("Matcher:", config.matcher_mode)
    print("Warmup exclusion (frames):", config.min_decode_frames)

    try:
        results = runner.run_all(
            datasets=args.datasets,
            use_cache=args.use_cache,
            force_reextract=args.force_reextract,
        )
    except FileNotFoundError as exc:
        print("ERROR:", exc)
        return 1

    json_path, csv_path = runner.write_report(results)
    print_summary_table(results)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\nSummary: {passed}/{total} datasets passed")
    print("Report JSON:", json_path)
    print("Report CSV:", csv_path)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
