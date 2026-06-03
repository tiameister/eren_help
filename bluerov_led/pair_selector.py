"""LED pair selection (Phase 1: largest-two-by-area, exactly-two gate)."""

from __future__ import annotations

from bluerov_led.config import VisionConfig
from bluerov_led.types import LedCandidate


class PairSelector:
    """Select the two LEDs belonging to the same face."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    def select(
        self, candidates: list[LedCandidate]
    ) -> tuple[LedCandidate, LedCandidate] | None:
        if len(candidates) < 2:
            return None

        if (
            self.config.require_exactly_two_candidates
            and len(candidates) != 2
        ):
            return None

        if self.config.pair_strategy == "largest2":
            return candidates[0], candidates[1]

        raise ValueError(f"Unknown pair strategy: {self.config.pair_strategy}")
