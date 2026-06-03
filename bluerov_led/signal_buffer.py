"""Per-track ON/OFF signal and geometry history buffers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from bluerov_led.config import VisionConfig


@dataclass
class TrackSample:
    on_bit: int
    cx: float
    cy: float
    area: float


class TrackSignalBuffer:
    """Maintains temporal deques for each tracked LED ID."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._signals: dict[int, deque[int]] = {}
        self._geometry: dict[int, deque[TrackSample]] = {}

    def reset(self) -> None:
        self._signals.clear()
        self._geometry.clear()

    def append(
        self,
        track_id: int,
        on_bit: int,
        cx: float,
        cy: float,
        area: float,
    ) -> None:
        if track_id not in self._signals:
            self._signals[track_id] = deque(maxlen=self.config.signal_buffer_maxlen)
            self._geometry[track_id] = deque(
                maxlen=self.config.geometry_history_maxlen
            )

        self._signals[track_id].append(on_bit)
        self._geometry[track_id].append(
            TrackSample(on_bit=on_bit, cx=cx, cy=cy, area=area)
        )

    def get_signal(self, track_id: int) -> list[int]:
        buf = self._signals.get(track_id)
        if buf is None:
            return []
        return list(buf)

    def get_geometry(self, track_id: int) -> list[TrackSample]:
        buf = self._geometry.get(track_id)
        if buf is None:
            return []
        return list(buf)

    def signal_length(self, track_id: int) -> int:
        buf = self._signals.get(track_id)
        return 0 if buf is None else len(buf)

    def ready_for_pairing(self, track_id: int) -> bool:
        return self.signal_length(track_id) >= self.config.min_pair_frames

    def prune_stale(self, active_track_ids: set[int]) -> None:
        stale = [tid for tid in self._signals if tid not in active_track_ids]
        for tid in stale:
            del self._signals[tid]
            self._geometry.pop(tid, None)
