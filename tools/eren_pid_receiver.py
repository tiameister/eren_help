#!/usr/bin/env python3
"""
Mock PID entry point — receive lean vision UDP packets and print control fields.

Zero third-party dependencies (stdlib only).

Usage:
  python tools/eren_pid_receiver.py
  python tools/eren_pid_receiver.py --host 0.0.0.0 --port 5005
"""

from __future__ import annotations

import argparse
import json
import socket
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive PID UDP JSON packets from BlueROV LED stream-udp.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=5005, help="UDP port.")
    return parser


def format_pid_line(packet: dict) -> str:
    valid = packet.get("valid", False)
    error_yaw = packet.get("error_yaw", 0.0)
    error_heave = packet.get("error_heave", 0.0)
    distance_surge = packet.get("distance_surge", 0.0)
    timestamp = packet.get("timestamp", 0.0)

    return (
        f"valid={valid}  "
        f"error_yaw={error_yaw:+.4f}  "
        f"error_heave={error_heave:+.4f}  "
        f"distance_surge={distance_surge:.3f}  "
        f"timestamp={timestamp:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print("PID receiver started.")
    print("Listening on:", sock.getsockname())
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            data, address = sock.recvfrom(8192)
            try:
                packet = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                print(f"Invalid JSON from {address}: {exc}")
                continue

            print(format_pid_line(packet))

    except KeyboardInterrupt:
        print("\nReceiver stopped.")
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
