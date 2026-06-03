"""Spatio-temporal LED pair matching and dynamic face decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from bluerov_led.centroid_tracker import CentroidTracker, TrackedBlob
from bluerov_led.config import VisionConfig
from bluerov_led.face_decoder import FacePatternDecoder
from bluerov_led.signal_buffer import TrackSignalBuffer, TrackSample
from bluerov_led.types import LedCandidate


@dataclass
class PairMatchResult:
    face_id: str
    pattern: str
    pattern_accuracy: float
    bit_error_rate: float
    bit_error_count: int
    pair_correlation: float
    pair_score: float
    geometry_score: float
    led1: LedCandidate
    led2: LedCandidate
    track_id_1: int
    track_id_2: int
    active_track_count: int
    fused_bit: int


def signal_correlation(signal_a: list[int], signal_b: list[int]) -> float:
    """Binary agreement rate over the overlapping tail (same-phase LEDs)."""
    length = min(len(signal_a), len(signal_b))
    if length == 0:
        return 0.0

    a = signal_a[-length:]
    b = signal_b[-length:]
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / length


def fuse_binary_signals(signal_a: list[int], signal_b: list[int]) -> list[int]:
    length = min(len(signal_a), len(signal_b))
    if length == 0:
        return []

    a = signal_a[-length:]
    b = signal_b[-length:]
    fused: list[int] = []

    for x, y in zip(a, b):
        fused.append(1 if (x + y) >= 1 else 0)

    return fused


class SpatioTemporalMatcher:
    """Find same-face LED pair via correlation, pattern decode, and geometry."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.tracker = CentroidTracker(config)
        self.buffers = TrackSignalBuffer(config)
        self.face_decoder = FacePatternDecoder(config)

    def reset(self) -> None:
        self.tracker.reset()
        self.buffers.reset()

    def _per_track_on(self, area: float) -> int:
        return 1 if area >= self.config.on_area_threshold else 0

    def update_tracks(
        self,
        candidates: list[LedCandidate],
        image_width: int,
        image_height: int,
    ) -> list[TrackedBlob]:
        tracks = self.tracker.update(candidates, image_width, image_height)

        for track in tracks:
            if track.candidate is not None:
                on_bit = self._per_track_on(track.area)
                area = track.area
            else:
                # Track coasting without a detection: treat as OFF for temporal decode.
                on_bit = 0
                area = 0.0

            self.buffers.append(
                track_id=track.track_id,
                on_bit=on_bit,
                cx=track.cx,
                cy=track.cy,
                area=area,
            )

        active_ids = {t.track_id for t in tracks}
        self.buffers.prune_stale(active_ids)
        return tracks

    def _geometry_score(
        self,
        track_id_a: int,
        track_id_b: int,
    ) -> tuple[float, LedCandidate | None, LedCandidate | None]:
        track_a = self.tracker.get_track(track_id_a)
        track_b = self.tracker.get_track(track_id_b)

        if (
            track_a is not None
            and track_b is not None
            and track_a.candidate is not None
            and track_b.candidate is not None
        ):
            led1 = track_a.candidate
            led2 = track_b.candidate
            dist = math.hypot(led1.cx - led2.cx, led1.cy - led2.cy)

            if dist < self.config.min_pixel_distance_px:
                return 0.0, None, None
            if dist > self.config.max_pixel_distance_px:
                return 0.0, None, None

            dy = abs(led1.cy - led2.cy)
            if dy / max(dist, 1.0) > self.config.max_y_alignment_ratio:
                return 0.0, None, None

            area_min = min(led1.area, led2.area)
            area_max = max(led1.area, led2.area)
            if area_max <= 0 or (area_min / area_max) < self.config.min_area_similarity:
                return 0.0, None, None

            return 1.0, led1, led2

        geom_a = self.buffers.get_geometry(track_id_a)
        geom_b = self.buffers.get_geometry(track_id_b)

        window = self.config.geometry_window_frames
        samples_a = geom_a[-window:]
        samples_b = geom_b[-window:]

        if len(samples_a) < 3 or len(samples_b) < 3:
            return 0.0, None, None

        length = min(len(samples_a), len(samples_b))
        samples_a = samples_a[-length:]
        samples_b = samples_b[-length:]

        distances: list[float] = []
        midpoints: list[tuple[float, float]] = []

        for sa, sb in zip(samples_a, samples_b):
            if sa.on_bit != 1 or sb.on_bit != 1:
                continue

            dist = math.hypot(sa.cx - sb.cx, sa.cy - sb.cy)
            distances.append(dist)
            midpoints.append(((sa.cx + sb.cx) / 2.0, (sa.cy + sb.cy) / 2.0))

        if len(distances) < 3:
            return 0.0, None, None

        mean_dist = sum(distances) / len(distances)

        if mean_dist <= 0:
            return 0.0, None, None

        dist_std = (sum((d - mean_dist) ** 2 for d in distances) / len(distances)) ** 0.5
        dist_cv = dist_std / mean_dist

        if dist_cv > self.config.max_pixel_distance_cv:
            return 0.0, None, None

        if mean_dist < self.config.min_pixel_distance_px:
            return 0.0, None, None

        if mean_dist > self.config.max_pixel_distance_px:
            return 0.0, None, None

        last_a = samples_a[-1]
        last_b = samples_b[-1]
        dy = abs(last_a.cy - last_b.cy)

        if dy / max(mean_dist, 1.0) > self.config.max_y_alignment_ratio:
            return 0.0, None, None

        area_min = min(last_a.area, last_b.area)
        area_max = max(last_a.area, last_b.area)

        if area_max <= 0 or (area_min / area_max) < self.config.min_area_similarity:
            return 0.0, None, None

        max_jump = 0.0
        for i in range(1, len(midpoints)):
            mx0, my0 = midpoints[i - 1]
            mx1, my1 = midpoints[i]
            max_jump = max(max_jump, math.hypot(mx1 - mx0, my1 - my0))

        if max_jump > self.config.max_midpoint_jump_px:
            return 0.0, None, None

        dist_score = 1.0 - min(1.0, dist_cv / self.config.max_pixel_distance_cv)
        geometry_score = dist_score

        led1 = LedCandidate(
            x=int(last_a.cx),
            y=int(last_a.cy),
            w=1,
            h=1,
            cx=int(last_a.cx),
            cy=int(last_a.cy),
            area=last_a.area,
        )
        led2 = LedCandidate(
            x=int(last_b.cx),
            y=int(last_b.cy),
            w=1,
            h=1,
            cx=int(last_b.cx),
            cy=int(last_b.cy),
            area=last_b.area,
        )

        track_a = self.tracker.get_track(track_id_a)
        track_b = self.tracker.get_track(track_id_b)

        if track_a is not None and track_a.candidate is not None:
            led1 = track_a.candidate
        if track_b is not None and track_b.candidate is not None:
            led2 = track_b.candidate

        return geometry_score, led1, led2

    def find_best_pair(self) -> PairMatchResult | None:
        tracks = self.tracker.list_tracks()
        eligible = [
            t
            for t in tracks
            if t.age_frames >= self.config.min_track_age_frames
            and self.buffers.ready_for_pairing(t.track_id)
        ]

        if len(eligible) < 2:
            return None

        best: PairMatchResult | None = None
        best_combined = -1.0

        for track_a, track_b in combinations(eligible, 2):
            id_a = track_a.track_id
            id_b = track_b.track_id

            signal_a = self.buffers.get_signal(id_a)
            signal_b = self.buffers.get_signal(id_b)

            correlation = signal_correlation(signal_a, signal_b)

            if correlation < self.config.min_pair_correlation:
                continue

            fused = fuse_binary_signals(signal_a, signal_b)
            decode = self.face_decoder.decode_signal(fused)

            if decode is None:
                continue

            if decode.global_accuracy < self.config.min_pattern_accuracy:
                continue

            geometry_score, led1, led2 = self._geometry_score(id_a, id_b)

            if led1 is None or led2 is None or geometry_score <= 0:
                continue

            pair_score = correlation * decode.global_accuracy
            combined = pair_score * geometry_score

            if combined < self.config.min_pair_score:
                continue

            if combined > best_combined:
                best_combined = combined
                fused_bit = 1 if fused and fused[-1] == 1 else 0

                best = PairMatchResult(
                    face_id=decode.face_id,
                    pattern=decode.pattern,
                    pattern_accuracy=decode.global_accuracy,
                    bit_error_rate=decode.bit_error_rate,
                    bit_error_count=decode.bit_error_count,
                    pair_correlation=correlation,
                    pair_score=pair_score,
                    geometry_score=geometry_score,
                    led1=led1,
                    led2=led2,
                    track_id_1=id_a,
                    track_id_2=id_b,
                    active_track_count=self.tracker.active_track_count,
                    fused_bit=fused_bit,
                )

        return best
