"""HSV-based LED candidate extraction."""

from __future__ import annotations

import cv2
import numpy as np

from bluerov_led.config import BackFaceConfig
from bluerov_led.types import LedCandidate


class LedCandidateExtractor:
    """Extract LED blob candidates from a BGR frame."""

    def __init__(self, config: BackFaceConfig) -> None:
        self.config = config

    def extract_from_hsv(
        self, hsv_frame: np.ndarray
    ) -> tuple[list[LedCandidate], np.ndarray]:
        mask = cv2.inRange(
            hsv_frame,
            self.config.lower_hsv_array,
            self.config.upper_hsv_array,
        )

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(
            mask_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[LedCandidate] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.config.min_area or area > self.config.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if h == 0:
                continue

            aspect_ratio = w / h

            if (
                aspect_ratio < self.config.min_aspect_ratio
                or aspect_ratio > self.config.max_aspect_ratio
            ):
                continue

            cx = x + w // 2
            cy = y + h // 2

            candidates.append(
                LedCandidate(
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                    cx=int(cx),
                    cy=int(cy),
                    area=float(area),
                )
            )

        candidates.sort(key=lambda c: c.area, reverse=True)
        return candidates, mask_clean

    def extract(self, frame: np.ndarray) -> tuple[list[LedCandidate], np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return self.extract_from_hsv(hsv)
