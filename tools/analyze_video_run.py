#!/usr/bin/env python3
"""Quick telemetry pass over datasets/videos/*.mp4 (no re-encode)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bluerov_led.config import VisionConfig
from bluerov_led.dataset_io import ArtifactWriter
from bluerov_led.distance_model import DistanceModel
from bluerov_led.pipeline import StreamingPipeline


def analyze(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cfg = VisionConfig()
    model_path = PROJECT_ROOT / "outputs" / "calibration" / "distance_model_summary.json"
    model_data = ArtifactWriter.read_json(model_path)
    distance_model_dict = (
        model_data if model_data is not None else DistanceModel.fit().to_summary_dict()
    )

    pipe = StreamingPipeline(
        config=cfg,
        distance_model_dict=distance_model_dict,
        udp_ip=None,
    )
    pipe.reset()

    n = pair = valid = held = lock = 0
    dist_samples: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pkt, _, _, rec = pipe.process_frame(
            frame,
            video_path.stem,
            n,
            video_path.name,
            dt=1.0 / fps,
        )
        n += 1
        if rec.pair_found:
            pair += 1
        if pkt and pkt.get("valid"):
            valid += 1
            if pkt.get("estimated_distance") is not None:
                dist_samples.append(float(pkt["estimated_distance"]))
        if pkt and pkt.get("held"):
            held += 1
        if pipe.matcher.lock_on.active:
            lock += 1

    cap.release()

    med_dist = None
    if dist_samples:
        dist_samples.sort()
        med_dist = dist_samples[len(dist_samples) // 2]

    return {
        "frames": n,
        "expected": total,
        "pair_found": pair,
        "valid": valid,
        "held": held,
        "lock_frames": lock,
        "pair_pct": 100.0 * pair / n if n else 0.0,
        "valid_pct": 100.0 * valid / n if n else 0.0,
        "median_est_distance": med_dist,
    }


def verify_output(out_path: Path, expected_frames: int) -> dict:
    cap = cv2.VideoCapture(str(out_path))
    if not cap.isOpened():
        return {"ok": False, "frames": 0, "error": "cannot open output"}
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return {
        "ok": n == expected_frames and n > 0,
        "frames": n,
        "size_mb": out_path.stat().st_size / 1e6,
    }


def main() -> int:
    folder = PROJECT_ROOT / "datasets" / "videos"
    videos = sorted(folder.glob("*.mp4"))
    if not videos:
        print("No MP4 files in datasets/videos/")
        return 1

    print("Video integration report\n")
    for video in videos:
        print(f"--- {video.name} ---")
        stats = analyze(video)
        out = PROJECT_ROOT / "outputs" / f"{video.stem}_annotated.mp4"
        out_check = verify_output(out, stats["frames"]) if out.exists() else {"ok": False, "frames": 0, "error": "missing"}

        print(f"  frames decoded:     {stats['frames']} (container reports {stats['expected']})")
        print(f"  pair_found:         {stats['pair_found']} ({stats['pair_pct']:.1f}%)")
        print(f"  valid observations: {stats['valid']} ({stats['valid_pct']:.1f}%)")
        print(f"  held packets:       {stats['held']}")
        print(f"  lock-on active:     {stats['lock_frames']} frames")
        if stats["median_est_distance"] is not None:
            print(f"  median est. dist:   {stats['median_est_distance']:.3f} (Unity units)")

        if out_check.get("ok"):
            print(f"  annotated output:   OK  {out.name}  {out_check['frames']} frames  {out_check['size_mb']:.1f} MB")
        else:
            err = out_check.get("error", "frame mismatch")
            print(f"  annotated output:   FAIL  {err}  (expected {out.name})")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
