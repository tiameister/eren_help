"""Decode temporal ON/OFF signals against known face patterns."""

from __future__ import annotations

from dataclasses import dataclass

from bluerov_led.config import VisionConfig
from bluerov_led.temporal_decoder import TemporalDecoder


@dataclass
class FaceDecodeResult:
    face_id: str
    pattern: str
    global_accuracy: float
    bit_error_rate: float
    bit_error_count: int
    decoded_bits: str


class FacePatternDecoder:
    """Match a binary frame signal to the best face pattern in the registry."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._decoder = TemporalDecoder(config)

    def decode_signal(self, frame_bits: list[int]) -> FaceDecodeResult | None:
        if len(frame_bits) < self.config.min_decode_frames:
            return None

        decoded_bits = self._decoder.decode_frame_bits(frame_bits)
        if len(decoded_bits) < 8:
            return None

        decoded_string = "".join(str(b) for b in decoded_bits)

        best_face: str | None = None
        best_pattern: str | None = None
        best_accuracy = -1.0
        best_error_rate = 1.0
        best_error_count = 0

        for face_id, pattern in self.config.face_patterns.items():
            match = self._decoder.global_repeated_pattern_match(
                decoded_string, pattern
            )
            accuracy = match["global_accuracy"]

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_face = face_id
                best_pattern = pattern
                best_error_rate = match["bit_error_rate"]
                best_error_count = match["bit_error_count"]

        if best_face is None or best_pattern is None:
            return None

        return FaceDecodeResult(
            face_id=best_face,
            pattern=best_pattern,
            global_accuracy=best_accuracy,
            bit_error_rate=best_error_rate,
            bit_error_count=best_error_count,
            decoded_bits=decoded_string,
        )
