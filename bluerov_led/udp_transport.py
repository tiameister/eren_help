"""UDP JSON observation packet send/receive."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

from bluerov_led.pid_packet import PID_PACKET_KEYS, build_pid_udp_packet


class UdpSender:
    """Send observation packets as UTF-8 JSON over UDP."""

    def __init__(self, ip: str, port: int) -> None:
        self.destination = (ip, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_packet(self, packet: dict[str, Any], seq: int) -> int:
        packet_to_send = dict(packet)
        packet_to_send["udp_seq"] = seq
        packet_to_send["sent_time_unix"] = time.time()
        payload = json.dumps(packet_to_send).encode("utf-8")
        self._sock.sendto(payload, self.destination)
        return len(payload)

    def send_pid_packet(
        self,
        *,
        valid: bool,
        error_yaw: float,
        error_heave: float,
        distance_surge: float,
    ) -> int:
        """Send lean 5-field PID JSON (no udp_seq or extra metadata)."""
        packet = build_pid_udp_packet(
            valid=valid,
            error_yaw=error_yaw,
            error_heave=error_heave,
            distance_surge=distance_surge,
        )
        payload = json.dumps(
            {key: packet[key] for key in PID_PACKET_KEYS}
        ).encode("utf-8")
        self._sock.sendto(payload, self.destination)
        return len(payload)

    def send_from_file(
        self,
        packet_path: Path,
        count: int = 10,
        rate_hz: float = 10.0,
    ) -> None:
        if not packet_path.exists():
            raise FileNotFoundError(
                f"Packet file not found: {packet_path}\n"
                "Run `python main.py packet` first."
            )

        with packet_path.open("r", encoding="utf-8") as f:
            packet = json.load(f)

        interval = 0.0 if rate_hz <= 0 else 1.0 / rate_hz

        print("UDP sender started.")
        print("Packet file:", packet_path)
        print("Destination:", self.destination)
        print("Count:", count)
        print("Rate:", rate_hz, "Hz")

        for seq in range(count):
            nbytes = self.send_packet(packet, seq)
            print(
                f"Sent seq={seq}, bytes={nbytes}, "
                f"valid={packet.get('valid')}, "
                f"frame={packet.get('frame')}, "
                f"error={packet.get('error_norm')}, "
                f"distance={packet.get('estimated_distance')}"
            )
            if interval > 0 and seq < count - 1:
                time.sleep(interval)

        print("UDP sender finished.")

    def close(self) -> None:
        self._sock.close()


class UdpReceiver:
    """Listen for observation packets and print summary fields."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        timeout: float | None = None,
    ) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        if timeout is not None:
            self._sock.settimeout(timeout)

    def run(self) -> None:
        print("UDP receiver started.")
        print("Listening on:", self._sock.getsockname())
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                data, address = self._sock.recvfrom(8192)
                receive_time = time.time()

                try:
                    packet = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    print("Received invalid JSON from:", address)
                    print("Raw data:", data)
                    continue

                sent_time = packet.get("sent_time_unix")
                latency_ms = None
                if sent_time is not None:
                    latency_ms = (receive_time - float(sent_time)) * 1000.0

                print("-" * 70)
                print("From:", address)
                print("udp_seq:", packet.get("udp_seq"))
                print("valid:", packet.get("valid"))
                print("dataset:", packet.get("dataset"))
                print("frame:", packet.get("frame"))
                print("face_id:", packet.get("face_id"))
                print("pattern_accuracy:", packet.get("pattern_accuracy"))
                print("bit_error_rate:", packet.get("bit_error_rate"))
                print("error_norm:", packet.get("error_norm"))
                print("ray_cam:", packet.get("ray_cam"))
                print("pixel_distance:", packet.get("pixel_distance"))
                print("estimated_distance:", packet.get("estimated_distance"))
                print("distance_confidence:", packet.get("distance_confidence"))
                if latency_ms is not None:
                    print(f"latency_ms: {latency_ms:.3f}")

        except KeyboardInterrupt:
            print("\nReceiver stopped by user.")
        finally:
            self._sock.close()
