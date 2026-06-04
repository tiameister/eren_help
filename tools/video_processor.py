#!/usr/bin/env python3
"""MP4 video ingestion, adaptive pipeline processing, and HUD annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bluerov_led.config import ProjectPaths, VisionConfig
from bluerov_led.dataset_io import ArtifactWriter
from bluerov_led.distance_model import DistanceModel
from bluerov_led.pipeline import StreamingPipeline
from bluerov_led.types import FrameRecord, LedCandidate


def _float_or_zero(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _led_boxes(
    led1_x: float,
    led1_y: float,
    led2_x: float,
    led2_y: float,
    candidates: list[LedCandidate],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) bounding boxes for each LED."""

    def box_for(cx: float, cy: float) -> tuple[int, int, int, int]:
        best: LedCandidate | None = None
        best_dist = float("inf")
        for c in candidates:
            dist = (c.cx - cx) ** 2 + (c.cy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = c
        if best is not None and best_dist < 60 ** 2:
            return (best.x, best.y, best.x + best.w, best.y + best.h)
        half = 14
        ix, iy = int(cx), int(cy)
        return (ix - half, iy - half, ix + half, iy + half)

    return box_for(led1_x, led1_y), box_for(led2_x, led2_y)


def annotate_frame(
    frame: np.ndarray,
    *,
    record: FrameRecord,
    packet: dict | None,
    lock_active: bool,
    target_face_id: str,
    candidates: list[LedCandidate] | None = None,
) -> np.ndarray:
    """Draw LED boxes, midpoint crosshair, and telemetry HUD."""
    output = frame.copy()
    h, w = output.shape[:2]

    packet = packet or {}
    is_valid = bool(packet.get("valid"))
    is_held = bool(packet.get("held"))

    def _coord(pkt_key: str, rec_attr: str) -> float | None:
        if packet.get(pkt_key) is not None:
            return float(packet[pkt_key])
        rec_val = getattr(record, rec_attr, None)
        if rec_val is not None:
            return float(rec_val)
        return None

    led1_x = _coord("led1_x", "led1_x")
    led1_y = _coord("led1_y", "led1_y")
    led2_x = _coord("led2_x", "led2_x")
    led2_y = _coord("led2_y", "led2_y")
    has_geometry = (
        led1_x is not None
        and led1_y is not None
        and led2_x is not None
        and led2_y is not None
    )
    show_tracking = (
        record.pair_found == 1
        or lock_active
        or (is_held and has_geometry)
    ) and has_geometry

    if show_tracking:
        led1_x = float(led1_x)  # type: ignore[arg-type]
        led1_y = float(led1_y)  # type: ignore[arg-type]
        led2_x = float(led2_x)  # type: ignore[arg-type]
        led2_y = float(led2_y)  # type: ignore[arg-type]
        mid_x = _coord("mid_x", "mid_x")
        mid_y = _coord("mid_y", "mid_y")
        if mid_x is None:
            mid_x = (led1_x + led2_x) / 2.0
        if mid_y is None:
            mid_y = (led1_y + led2_y) / 2.0
        mid_x = float(mid_x)
        mid_y = float(mid_y)

        box1, box2 = _led_boxes(
            led1_x, led1_y, led2_x, led2_y, candidates or []
        )

        box_color = (0, 220, 120) if is_valid else (0, 180, 255)
        for box in (box1, box2):
            cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), box_color, 2)

        cv2.line(
            output,
            (int(led1_x), int(led1_y)),
            (int(led2_x), int(led2_y)),
            (255, 120, 0),
            2,
        )
        cross = 10
        mx, my = int(mid_x), int(mid_y)
        cv2.line(output, (mx - cross, my), (mx + cross, my), (255, 255, 255), 2)
        cv2.line(output, (mx, my - cross), (mx, my + cross), (255, 255, 255), 2)
        cv2.circle(output, (mx, my), 5, (255, 255, 255), -1)

    hud_ok = is_valid or (is_held and has_geometry)
    hud_color = (80, 255, 160) if hud_ok else (80, 80, 255)
    if not hud_ok and not lock_active:
        hud_color = (60, 60, 255)

    face_id = packet.get("face_id") or record.face_id or "—"
    if is_valid or is_held:
        distance = _float_or_zero(packet.get("estimated_distance"))
        yaw_err = _float_or_zero(packet.get("error_x"))
        lock_line = "LOCKED" if lock_active else ("HELD" if is_held else "TRACKING")
        lines = [
            f"Mission Face: {face_id}",
            f"Distance: {distance:.3f}",
            f"Yaw Error: {yaw_err:.4f}",
            f"State: {lock_line}",
        ]
    else:
        lines = [
            "TARGET LOST",
            f"Mission Face: {target_face_id}",
            "Distance: —",
            "Yaw Error: —",
        ]
        if lock_active:
            lines.append("State: LOCK SEARCH")

    panel_w = 420
    panel_h = 28 + len(lines) * 26
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)
    cv2.rectangle(output, (12, 12), (12 + panel_w, 12 + panel_h), hud_color, 2)

    for idx, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (24, 38 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            hud_color,
            2,
            cv2.LINE_AA,
        )

    status = "VALID" if is_valid else ("HELD" if is_held else "LOST")
    cv2.putText(
        output,
        f"F:{record.frame}  {status}  Cands:{record.candidate_count}",
        (12, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return output




def resolve_output_path(input_video: Path, outputs_dir: Path) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir / f"{input_video.stem}_annotated.mp4"


def process_video(
    input_video: Path,
    output_video: Path | None = None,
    *,
    project_root: Path | None = None,
    outputs_dir: str = "outputs",
    no_udp: bool = True,
) -> Path:
    """
    Read MP4 frames, run StreamingPipeline.process_frame, write annotated MP4.
    Emits PROGRESS: i/n lines for GUI integration.
    """
    root = project_root or PROJECT_ROOT
    paths = ProjectPaths(project_root=root, outputs_dir=outputs_dir)
    input_video = input_video.resolve()

    if not input_video.is_file():
        raise FileNotFoundError(f"Video not found: {input_video}")

    if input_video.stat().st_size < 1:
        raise RuntimeError(f"Video file is empty: {input_video}")

    out_path = output_video or resolve_output_path(
        input_video, paths.project_root / paths.outputs_dir
    )
    out_path = out_path.resolve()

    config = VisionConfig(matcher_mode="spatio_temporal")
    model_path = paths.distance_model_json()
    model_data = ArtifactWriter.read_json(model_path)
    distance_model_dict = (
        model_data if model_data is not None else DistanceModel.fit().to_summary_dict()
    )

    pipeline = StreamingPipeline(
        config=config,
        distance_model_dict=distance_model_dict,
        udp_ip=None if no_udp else "127.0.0.1",
        udp_port=5005,
    )
    pipeline.reset()

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = float(config.fps)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0

    if width < 1 or height < 1:
        capture.release()
        raise RuntimeError(
            f"Invalid video dimensions {width}x{height}: {input_video}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {out_path}")

    dataset_name = input_video.stem
    frame_index = 0
    processed = 0
    write_failed = False

    print(f"Input: {input_video}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Resolution: {width}x{height} @ {fps:.2f} fps", flush=True)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if total > 0:
                print(f"PROGRESS: {frame_index + 1}/{total}", flush=True)
                print(
                    f"Processing frame {frame_index + 1}/{total}...",
                    flush=True,
                )
            else:
                print(f"PROGRESS: {frame_index + 1}/?", flush=True)
                print(
                    f"Processing frame {frame_index + 1}...",
                    flush=True,
                )

            packet, candidates, _mask, record = pipeline.process_frame(
                frame,
                dataset_name,
                frame_index,
                file_name=f"{input_video.name}:{frame_index}",
                dt=1.0 / fps,
            )

            lock_active = pipeline.matcher.lock_on.active
            annotated = annotate_frame(
                frame,
                record=record,
                packet=packet,
                lock_active=lock_active,
                target_face_id=config.target_face_id,
                candidates=candidates,
            )
            writer.write(annotated)
            frame_index += 1
            processed += 1

    except Exception:
        write_failed = True
        raise
    finally:
        capture.release()
        writer.release()
        if write_failed and out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass

    if processed < 1:
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"No frames decoded from video (file may be corrupt): {input_video}"
        )

    if total > 0:
        print(f"PROGRESS: {total}/{total}", flush=True)
    print(f"Finished. Wrote {processed} frames to {out_path}", flush=True)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate MP4 with LED tracking HUD.")
    parser.add_argument("--video", required=True, help="Path to input .mp4")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .mp4 path (default: outputs/<stem>_annotated.mp4)",
    )
    parser.add_argument("--outputs-dir", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.video)
    output_path = Path(args.output) if args.output else None
    try:
        process_video(
            input_path,
            output_path,
            outputs_dir=args.outputs_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
