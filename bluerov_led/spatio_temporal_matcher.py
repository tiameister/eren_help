"""Spatio-temporal LED pair matching and dynamic face decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from bluerov_led.centroid_tracker import CentroidTracker, TrackedBlob
from bluerov_led.config import VisionConfig
from bluerov_led.face_decoder import FaceDecodeResult, FacePatternDecoder
from bluerov_led.signal_buffer import TrackSignalBuffer
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
        self._last_candidate_count = 0
        self._last_candidates: list[LedCandidate] = []
        self._last_face_decode: FaceDecodeResult | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self.buffers.reset()
        self._last_candidate_count = 0
        self._last_candidates = []
        self._last_face_decode = None

    def _per_track_on(self, area: float) -> int:
        return 1 if area >= self.config.on_area_threshold else 0

    def update_tracks(
        self,
        candidates: list[LedCandidate],
        image_width: int,
        image_height: int,
        dt: float | None = None
    ) -> list[TrackedBlob]:
        tracks = self.tracker.update(candidates, image_width, image_height, dt)

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
        self._last_candidate_count = len(candidates)
        self._last_candidates = list(candidates)
        return tracks

    def _min_track_age_frames(self, track: TrackedBlob) -> int:
        if (
            self._last_candidate_count <= 2
            and track.candidate is not None
        ):
            return 3
        return self.config.min_track_age_frames

    def _min_pair_frames_for_track(self, track: TrackedBlob) -> int:
        if (
            self.config.prefer_matched_tracks
            and self._last_candidate_count == 2
            and track.candidate is not None
        ):
            return max(6, self.config.min_pair_frames // 2)
        return self.config.min_pair_frames

    def _eligible_tracks(self) -> list[TrackedBlob]:
        tracks = self.tracker.list_tracks()

        matched_live = [
            t
            for t in tracks
            if t.candidate is not None
            and t.age_frames >= self._min_track_age_frames(t)
            and self.buffers.signal_length(t.track_id)
            >= self._min_pair_frames_for_track(t)
        ]

        if self.config.prefer_matched_tracks and len(matched_live) >= 2:
            return matched_live

        return [
            t
            for t in tracks
            if t.age_frames >= self._min_track_age_frames(t)
            and self.buffers.ready_for_pairing(t.track_id)
        ]

    def _nearest_track(
        self,
        candidate: LedCandidate,
        tracks: list[TrackedBlob],
    ) -> TrackedBlob | None:
        best: TrackedBlob | None = None
        best_dist = float("inf")

        for track in tracks:
            dist = math.hypot(track.cx - candidate.cx, track.cy - candidate.cy)
            if dist < best_dist:
                best_dist = dist
                best = track

        return best

    def _priority_pair_candidates(
        self,
        eligible: list[TrackedBlob],
    ) -> list[tuple[TrackedBlob, TrackedBlob]]:
        pairs: list[tuple[TrackedBlob, TrackedBlob]] = []

        if len(self._last_candidates) >= 2:
            top_two = sorted(
                self._last_candidates,
                key=lambda c: c.area,
                reverse=True,
            )[:2]
            track_a = self._nearest_track(top_two[0], eligible)
            track_b = self._nearest_track(top_two[1], eligible)

            if (
                track_a is not None
                and track_b is not None
                and track_a.track_id != track_b.track_id
            ):
                pairs.append((track_a, track_b))

        if self._last_candidate_count == 2:
            matched = [t for t in eligible if t.candidate is not None]
            if len(matched) == 2:
                pairs.append((matched[0], matched[1]))

        return pairs

    def _is_far_range_scene(self) -> bool:
        if not self._last_candidates:
            return False
        return max(c.area for c in self._last_candidates) < 90.0

    def _combination_pairs(
        self,
        eligible: list[TrackedBlob],
    ) -> list[tuple[TrackedBlob, TrackedBlob]]:
        if (
            self._is_far_range_scene()
            and self._last_candidate_count >= 3
            and len(self._last_candidates) >= 2
        ):
            top_two = sorted(
                self._last_candidates,
                key=lambda c: c.area,
                reverse=True,
            )[:2]
            track_a = self._nearest_track(top_two[0], eligible)
            track_b = self._nearest_track(top_two[1], eligible)

            if (
                track_a is not None
                and track_b is not None
                and track_a.track_id != track_b.track_id
            ):
                return [(track_a, track_b)]

        return list(combinations(eligible, 2))

    def _fallback_largest_two_pair(self) -> PairMatchResult | None:
        if (
            len(self._last_candidates) < 2
            or self._last_face_decode is None
            or not self._is_far_range_scene()
        ):
            return None

        led1, led2 = sorted(
            self._last_candidates,
            key=lambda c: c.area,
            reverse=True,
        )[:2]

        dist = math.hypot(led1.cx - led2.cx, led1.cy - led2.cy)
        min_dist_px = self.config.min_pixel_distance_px
        if max(led1.area, led2.area) < 100.0:
            min_dist_px = min(min_dist_px, 22.0)

        if dist < min_dist_px or dist > self.config.max_pixel_distance_px:
            return None

        dy = abs(led1.cy - led2.cy)
        if dy / max(dist, 1.0) > self.config.max_y_alignment_ratio:
            return None

        area_min = min(led1.area, led2.area)
        area_max = max(led1.area, led2.area)
        if area_max <= 0 or (area_min / area_max) < self.config.min_area_similarity:
            return None

        decode = self._last_face_decode
        track_a = self._nearest_track(led1, self.tracker.list_tracks())
        track_b = self._nearest_track(led2, self.tracker.list_tracks())

        return PairMatchResult(
            face_id=decode.face_id,
            pattern=decode.pattern,
            pattern_accuracy=decode.global_accuracy,
            bit_error_rate=decode.bit_error_rate,
            bit_error_count=decode.bit_error_count,
            pair_correlation=1.0,
            pair_score=decode.global_accuracy,
            geometry_score=1.0,
            led1=led1,
            led2=led2,
            track_id_1=track_a.track_id if track_a else -1,
            track_id_2=track_b.track_id if track_b else -1,
            active_track_count=self.tracker.active_track_count,
            fused_bit=1,
        )

    def _resolve_face_decode(
        self,
        fused: list[int],
        correlation: float,
        relaxed_decode: bool = False,
    ) -> FaceDecodeResult | None:
        min_decoded_bits = 4 if relaxed_decode else 8
        min_signal_frames = 24 if relaxed_decode else None
        decode = self.face_decoder.decode_signal(
            fused,
            min_decoded_bits=min_decoded_bits,
            min_signal_frames=min_signal_frames,
        )

        if decode is not None:
            self._last_face_decode = decode
            return decode

        fallback_threshold = (
            0.82
            if relaxed_decode
            else self.config.high_correlation_fallback
        )

        if (
            correlation >= fallback_threshold
            and self._last_face_decode is not None
        ):
            return self._last_face_decode

        return None

    def _evaluate_track_pair(
        self,
        track_a: TrackedBlob,
        track_b: TrackedBlob,
        relaxed_decode: bool = False,
        relaxed_correlation: bool = False,
    ) -> PairMatchResult | None:
        id_a = track_a.track_id
        id_b = track_b.track_id

        signal_a = self.buffers.get_signal(id_a)
        signal_b = self.buffers.get_signal(id_b)

        correlation = signal_correlation(signal_a, signal_b)

        min_correlation = self.config.min_pair_correlation
        if relaxed_correlation:
            min_correlation = min(0.82, min_correlation - 0.06)

        if correlation < min_correlation:
            return None

        fused = fuse_binary_signals(signal_a, signal_b)
        decode = self._resolve_face_decode(
            fused,
            correlation,
            relaxed_decode=relaxed_decode,
        )

        if decode is None:
            return None

        if decode.global_accuracy < self.config.min_pattern_accuracy:
            return None

        geometry_score, led1, led2 = self._geometry_score(id_a, id_b)

        if led1 is None or led2 is None or geometry_score <= 0:
            return None

        pair_score = correlation * decode.global_accuracy
        combined = pair_score * geometry_score

        min_pair_score = self.config.min_pair_score
        if relaxed_decode or relaxed_correlation:
            min_pair_score *= 0.85

        if combined < min_pair_score:
            return None

        fused_bit = 1 if fused and fused[-1] == 1 else 0

        return PairMatchResult(
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

            min_dist_px = self.config.min_pixel_distance_px
            if max(led1.area, led2.area) < 100.0:
                min_dist_px = min(min_dist_px, 22.0)

            if dist < min_dist_px:
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
        eligible = self._eligible_tracks()

        if len(eligible) < 2:
            return None

        best: PairMatchResult | None = None
        best_combined = -1.0

        pair_lists: list[tuple[TrackedBlob, TrackedBlob, bool, bool]] = []

        far_range = self._is_far_range_scene()

        for track_a, track_b in self._priority_pair_candidates(eligible):
            pair_lists.append((track_a, track_b, far_range, far_range))

        for track_a, track_b in self._combination_pairs(eligible):
            pair_lists.append((track_a, track_b, False, False))

        seen_ids: set[tuple[int, int]] = set()

        for track_a, track_b, relaxed_decode, relaxed_correlation in pair_lists:
            key = tuple(sorted((track_a.track_id, track_b.track_id)))
            if key in seen_ids:
                continue
            seen_ids.add(key)

            candidate = self._evaluate_track_pair(
                track_a,
                track_b,
                relaxed_decode=relaxed_decode,
                relaxed_correlation=relaxed_correlation,
            )

            if candidate is None:
                continue

            combined = candidate.pair_score * candidate.geometry_score

            if combined > best_combined:
                best_combined = combined
                best = candidate

        if best is not None:
            return best

        return self._fallback_largest_two_pair()
