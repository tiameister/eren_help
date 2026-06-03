"""Temporal blink pattern decoding from per-frame bits."""

from __future__ import annotations

from bluerov_led.config import VisionConfig
from bluerov_led.types import PatternSummary


class TemporalDecoder:
    """Decode frame bits into a pattern string and global accuracy metrics."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config

    @staticmethod
    def cyclic_shifts(pattern: str) -> list[str]:
        return [pattern[i:] + pattern[:i] for i in range(len(pattern))]

    def decode_frame_bits(self, frame_bits: list[int]) -> list[int]:
        decoded: list[int] = []
        step = self.config.frames_per_bit

        for start in range(0, len(frame_bits), step):
            group = frame_bits[start : start + step]
            if len(group) < step:
                break
            ones = sum(group)
            zeros = len(group) - ones
            decoded.append(1 if ones >= zeros else 0)

        return decoded

    def best_pattern_window_match(
        self, decoded: str, expected: str
    ) -> tuple[float, str | None, int | None, str | None]:
        expected_shifts = self.cyclic_shifts(expected)
        pattern_len = len(expected)

        best_score = -1.0
        best_shift: str | None = None
        best_start: int | None = None
        best_window: str | None = None

        if len(decoded) < pattern_len:
            return best_score, best_shift, best_start, best_window

        for start in range(0, len(decoded) - pattern_len + 1):
            window = decoded[start : start + pattern_len]

            for shift in expected_shifts:
                matches = sum(1 for a, b in zip(window, shift) if a == b)
                score = matches / pattern_len

                if score > best_score:
                    best_score = score
                    best_shift = shift
                    best_start = start
                    best_window = window

        return best_score, best_shift, best_start, best_window

    @staticmethod
    def repeated_pattern_for_length(pattern: str, length: int) -> str:
        if len(pattern) == 0:
            raise ValueError("Pattern length cannot be zero.")
        return (pattern * ((length // len(pattern)) + 1))[:length]

    def global_repeated_pattern_match(
        self, decoded: str, expected: str
    ) -> dict:
        expected_shifts = self.cyclic_shifts(expected)

        best_accuracy = -1.0
        best_shift: str | None = None
        best_error_positions: list[int] = []

        for shift in expected_shifts:
            expected_repeated = self.repeated_pattern_for_length(
                shift, len(decoded)
            )

            error_positions = [
                i
                for i, (a, b) in enumerate(zip(decoded, expected_repeated))
                if a != b
            ]

            bit_error_count = len(error_positions)
            accuracy = (
                0.0
                if len(decoded) == 0
                else 1 - (bit_error_count / len(decoded))
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_shift = shift
                best_error_positions = error_positions

        bit_error_count = len(best_error_positions)
        bit_error_rate = (
            1.0 if len(decoded) == 0 else bit_error_count / len(decoded)
        )

        return {
            "global_accuracy": best_accuracy,
            "bit_error_count": bit_error_count,
            "bit_error_rate": bit_error_rate,
            "best_global_shift": best_shift,
            "error_positions": best_error_positions,
        }

    def decode_dataset(
        self,
        dataset: str,
        frame_bits: list[int],
        expected_pattern: str | None = None,
    ) -> PatternSummary:
        decoded_bits = self.decode_frame_bits(frame_bits)
        decoded_string = "".join(str(b) for b in decoded_bits)

        if expected_pattern is None:
            expected_pattern = self.config.face_patterns.get(
                "BACK", "11001100"
            )

        window_score, window_shift, window_start, window = (
            self.best_pattern_window_match(decoded_string, expected_pattern)
        )
        global_match = self.global_repeated_pattern_match(
            decoded_string, expected_pattern
        )

        return PatternSummary(
            dataset=dataset,
            fps=self.config.fps,
            bit_duration_seconds=self.config.bit_duration_seconds,
            frames_per_bit=self.config.frames_per_bit,
            expected_pattern=expected_pattern,
            total_frame_count=len(frame_bits),
            total_decoded_bit_count=len(decoded_bits),
            decoded_bits=decoded_string,
            local_best_score=window_score,
            local_best_start_index=window_start,
            local_decoded_window=window,
            local_matched_expected_shift=window_shift,
            global_accuracy=global_match["global_accuracy"],
            bit_error_count=global_match["bit_error_count"],
            bit_error_rate=global_match["bit_error_rate"],
            best_global_shift=global_match["best_global_shift"],
            error_positions=global_match["error_positions"],
        )

    def print_summary(self, summary: PatternSummary) -> None:
        print("Dataset:", summary.dataset)
        print("FPS:", summary.fps)
        print("Bit duration:", summary.bit_duration_seconds)
        print("Frames per bit:", summary.frames_per_bit)
        print("Expected pattern:", summary.expected_pattern)
        print("Total frame count:", summary.total_frame_count)
        print("Total decoded bit count:", summary.total_decoded_bit_count)
        print("\nDecoded bits:")
        print(summary.decoded_bits)
        print("\nBest local 8-bit pattern match:")
        print("Local score:", summary.local_best_score)
        print("Best start index:", summary.local_best_start_index)
        print("Decoded window:", summary.local_decoded_window)
        print("Matched expected shift:", summary.local_matched_expected_shift)
        print("\nGlobal repeated-pattern match:")
        print("Global accuracy:", summary.global_accuracy)
        print("Bit error count:", summary.bit_error_count)
        print("Bit error rate:", summary.bit_error_rate)
        print("Best global shift:", summary.best_global_shift)

        if summary.bit_error_count > 0:
            print(
                "First error positions:",
                summary.error_positions[:20],
            )
        else:
            print("First error positions: none")

        if summary.global_accuracy >= 0.95:
            print("\nResult: BACK pattern is globally reliable.")
        elif summary.global_accuracy >= 0.80:
            print(
                "\nResult: BACK pattern is detected, "
                "but there are noticeable bit errors."
            )
        else:
            print("\nResult: BACK pattern is NOT globally reliable.")
