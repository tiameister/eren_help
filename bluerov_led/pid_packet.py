"""Lean PID controller UDP payload builder."""

from __future__ import annotations

import time
from typing import Any

PID_PACKET_KEYS = (
    "valid",
    "error_yaw",
    "error_heave",
    "distance_surge",
    "timestamp",
)


def build_pid_udp_packet(
    *,
    valid: bool,
    error_yaw: float,
    error_heave: float,
    distance_surge: float,
) -> dict[str, Any]:
    """Build the 5-field JSON contract for the Linux PID controller."""
    return {
        "valid": valid,
        "error_yaw": float(error_yaw),
        "error_heave": float(error_heave),
        "distance_surge": float(distance_surge),
        "timestamp": time.time(),
    }
