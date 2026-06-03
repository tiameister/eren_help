#!/usr/bin/env python3
"""BlueROV2 LED tracking — single CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bluerov_led.config import ProjectPaths, VisionConfig
from bluerov_led.pipeline import BackFacePipeline
from bluerov_led.preview import preview_dataset
from bluerov_led.udp_transport import UdpReceiver, UdpSender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BlueROV2 LED-based tracking - offline PNG pipeline and UDP tools.",
    )
    parser.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Root folder for PNG datasets.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root folder for generated artifacts.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_preview = sub.add_parser("preview", help="Play PNG sequence.")
    p_preview.add_argument("--dataset", default="BackOnly_Test_01")

    p_tune = sub.add_parser("tune-hsv", help="Interactive HSV tuner.")
    p_tune.add_argument("--dataset", default="BackOnly_Test_02")
    p_tune.add_argument("--frame-index", type=int, default=80)

    p_extract = sub.add_parser("extract", help="Extract LED pair CSV from PNGs.")
    p_extract.add_argument("--dataset", required=True)
    p_extract.add_argument(
        "--preview",
        action="store_true",
        help="Show live detection windows (press q to quit).",
    )
    p_extract.add_argument(
        "--matcher",
        choices=["spatio_temporal", "legacy_largest2"],
        default="spatio_temporal",
        help="Pair selection algorithm.",
    )

    p_decode = sub.add_parser("decode", help="Decode temporal blink pattern.")
    p_decode.add_argument("--dataset", required=True)

    p_filter = sub.add_parser("filter", help="IQR-filter valid distance frames.")
    p_filter.add_argument("--dataset", required=True)

    p_cal = sub.add_parser("calibrate", help="Fit distance model.")
    p_cal.add_argument(
        "--dataset",
        default=None,
        help="Unused; calibrates from built-in points.",
    )

    p_packet = sub.add_parser("packet", help="Build observation JSON packet.")
    p_packet.add_argument("--dataset", required=True)
    p_packet.add_argument("--frame", type=int, default=None)

    p_run = sub.add_parser(
        "run",
        help="Full pipeline: extract, decode, filter, calibrate, packet.",
    )
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--preview", action="store_true")
    p_run.add_argument(
        "--matcher",
        choices=["spatio_temporal", "legacy_largest2"],
        default="spatio_temporal",
        help="Pair selection algorithm.",
    )

    p_send = sub.add_parser("send-udp", help="Send observation packet over UDP.")
    p_send.add_argument("--dataset", default="BackOnly_Test_04")
    p_send.add_argument("--frame", type=int, default=None)
    p_send.add_argument("--ip", default="127.0.0.1")
    p_send.add_argument("--port", type=int, default=5005)
    p_send.add_argument("--count", type=int, default=10)
    p_send.add_argument("--rate", type=float, default=10.0)

    p_recv = sub.add_parser("recv-udp", help="Receive observation packets.")
    p_recv.add_argument("--host", default="0.0.0.0")
    p_recv.add_argument("--port", type=int, default=5005)
    p_recv.add_argument("--timeout", type=float, default=None)

    return parser


def make_pipeline(args: argparse.Namespace) -> BackFacePipeline:
    root = Path(__file__).resolve().parent
    paths = ProjectPaths(
        project_root=root,
        datasets_dir=args.datasets_dir,
        outputs_dir=args.outputs_dir,
    )
    matcher_mode = getattr(args, "matcher", "spatio_temporal")
    config = VisionConfig(matcher_mode=matcher_mode)
    if matcher_mode == "legacy_largest2":
        config.require_exactly_two_candidates = True
    return BackFacePipeline(config=config, paths=paths)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pipeline = make_pipeline(args)
    paths = pipeline.paths

    if args.command == "preview":
        preview_dataset(paths.dataset_folder(args.dataset))
        return 0

    if args.command == "tune-hsv":
        from tools.hsv_tuner import run_hsv_tuner

        run_hsv_tuner(
            paths.dataset_folder(args.dataset),
            frame_index=args.frame_index,
        )
        return 0

    if args.command == "extract":
        pipeline.extract(args.dataset, preview=args.preview)
        return 0

    if args.command == "decode":
        pipeline.decode_pattern(args.dataset)
        return 0

    if args.command == "filter":
        pipeline.filter_distances(args.dataset)
        return 0

    if args.command == "calibrate":
        pipeline.calibrate()
        return 0

    if args.command == "packet":
        pipeline.build_packet(args.dataset, frame=args.frame)
        return 0

    if args.command == "run":
        result = pipeline.run_all(args.dataset, preview=args.preview)
        print("\nPipeline complete.")
        print("  pair_csv:", result.pair_csv)
        print("  pattern_json:", result.pattern_json)
        print("  filtered_csv:", result.filtered_csv)
        print("  model_json:", result.model_json)
        print("  packet_json:", result.packet_json)
        return 0

    if args.command == "send-udp":
        packet_path = paths.observation_packet_json(args.dataset, args.frame)
        sender = UdpSender(args.ip, args.port)
        try:
            sender.send_from_file(packet_path, count=args.count, rate_hz=args.rate)
        finally:
            sender.close()
        return 0

    if args.command == "recv-udp":
        UdpReceiver(host=args.host, port=args.port, timeout=args.timeout).run()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
