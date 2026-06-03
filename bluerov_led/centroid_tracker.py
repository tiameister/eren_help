"""Lightweight Euclidean centroid tracker for LED blobs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from bluerov_led.config import VisionConfig
from bluerov_led.types import LedCandidate


@dataclass
class TrackedBlob:
    track_id: int
    cx: float
    cy: float
    area: float
    missed_frames: int = 0
    age_frames: int = 1
    candidate: LedCandidate | None = None


class CentroidTracker:
    """Greedy nearest-neighbor centroid tracker with EMA smoothing."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._tracks: dict[int, TrackedBlob] = {}
        self._next_id = 1
        self._match_distance: float | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._match_distance = None

    def _ensure_match_distance(self, image_width: int, image_height: int) -> float:
        if self._match_distance is None:
            self._match_distance = self.config.match_distance_for_image(
                image_width, image_height
            )
        return self._match_distance

    def update(
        self,
        candidates: list[LedCandidate],
        image_width: int,
        image_height: int,
    ) -> list[TrackedBlob]:
        max_dist = self._ensure_match_distance(image_width, image_height)
        alpha = self.config.centroid_ema_alpha

        if not self._tracks:
            for cand in candidates:
                self._spawn_track(cand)
            return list(self._tracks.values())

        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_detections = list(range(len(candidates)))

        pairs: list[tuple[float, int, int]] = []

        for t_idx, tid in enumerate(track_ids):
            track = self._tracks[tid]
            for d_idx in unmatched_detections:
                cand = candidates[d_idx]
                dist = math.hypot(cand.cx - track.cx, cand.cy - track.cy)
                if dist <= max_dist:
                    pairs.append((dist, t_idx, d_idx))

        pairs.sort(key=lambda item: item[0])

        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        for dist, t_idx, d_idx in pairs:
            if t_idx in used_tracks or d_idx in used_detections:
                continue

            tid = track_ids[t_idx]
            cand = candidates[d_idx]
            track = self._tracks[tid]

            track.cx = alpha * cand.cx + (1.0 - alpha) * track.cx
            track.cy = alpha * cand.cy + (1.0 - alpha) * track.cy
            track.area = cand.area
            track.candidate = cand
            track.missed_frames = 0
            track.age_frames += 1

            unmatched_tracks.discard(tid)
            used_tracks.add(t_idx)
            used_detections.add(d_idx)

        for t_idx, tid in enumerate(track_ids):
            if t_idx not in used_tracks and tid in unmatched_tracks:
                track = self._tracks[tid]
                track.missed_frames += 1
                track.candidate = None

        for d_idx in unmatched_detections:
            if d_idx not in used_detections:
                self._spawn_track(candidates[d_idx])

        stale = [
            tid
            for tid, track in self._tracks.items()
            if track.missed_frames > self.config.max_missed_frames
        ]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    def _spawn_track(self, candidate: LedCandidate) -> TrackedBlob:
        track = TrackedBlob(
            track_id=self._next_id,
            cx=float(candidate.cx),
            cy=float(candidate.cy),
            area=candidate.area,
            candidate=candidate,
        )
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def get_track(self, track_id: int) -> TrackedBlob | None:
        return self._tracks.get(track_id)

    def list_tracks(self) -> list[TrackedBlob]:
        return list(self._tracks.values())
