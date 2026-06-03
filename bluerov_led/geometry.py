"""Midpoint, image error, camera ray, and pixel distance."""

from __future__ import annotations

import math

from bluerov_led.config import BackFaceConfig
from bluerov_led.types import LedCandidate, PairGeometry


class GeometryCalculator:
    """Compute pair geometry and camera-frame ray toward midpoint."""

    def __init__(self, config: BackFaceConfig) -> None:
        self.config = config

    def compute_pair_geometry(
        self,
        c1: LedCandidate,
        c2: LedCandidate,
        image_width: int,
        image_height: int,
    ) -> PairGeometry:
        image_center_x = image_width / 2.0
        image_center_y = image_height / 2.0

        led1_x = float(c1.cx)
        led1_y = float(c1.cy)
        led2_x = float(c2.cx)
        led2_y = float(c2.cy)

        pixel_distance = math.sqrt(
            (led1_x - led2_x) ** 2 + (led1_y - led2_y) ** 2
        )

        mid_x = (led1_x + led2_x) / 2.0
        mid_y = (led1_y + led2_y) / 2.0

        error_x = (mid_x - image_center_x) / image_center_x
        error_y = (image_center_y - mid_y) / image_center_y

        vertical_fov_rad = math.radians(self.config.camera_vertical_fov_deg)
        fy = (image_height / 2.0) / math.tan(vertical_fov_rad / 2.0)
        fx = fy

        # Normalized camera coordinates; z=1 defines the image plane ray.
        x_cam = (mid_x - image_center_x) / fx
        y_cam = (image_center_y - mid_y) / fy
        z_cam = 1.0

        norm = math.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)

        return PairGeometry(
            led1_x=led1_x,
            led1_y=led1_y,
            led2_x=led2_x,
            led2_y=led2_y,
            pixel_distance=pixel_distance,
            mid_x=mid_x,
            mid_y=mid_y,
            error_x=error_x,
            error_y=error_y,
            ray_x=x_cam / norm,
            ray_y=y_cam / norm,
            ray_z=z_cam / norm,
        )
