"""Frame-level ON/OFF bit from candidate total area."""

from __future__ import annotations

from bluerov_led.config import BackFaceConfig
from bluerov_led.types import LedCandidate


class BitExtractor:
    """Derive a single blink bit per frame from aggregate LED area."""

    def __init__(self, config: BackFaceConfig) -> None:
        self.config = config

    def bit_from_candidates(self, candidates: list[LedCandidate]) -> int:
        total_area = sum(c.area for c in candidates)
        return 1 if total_area > self.config.on_area_threshold else 0

    def total_area(self, candidates: list[LedCandidate]) -> float:
        return sum(c.area for c in candidates)
