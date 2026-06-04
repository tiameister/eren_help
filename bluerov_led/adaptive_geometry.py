"""Area-proportional geometry gates for LED pair distance validation."""

from __future__ import annotations

import math

from bluerov_led.config import VisionConfig


def effective_blob_radius_sum(a1: float, a2: float) -> float:
    """Proxy for combined LED image scale: sqrt(A1) + sqrt(A2)."""
    return math.sqrt(max(a1, 0.0)) + math.sqrt(max(a2, 0.0))


def _distance_floor(a1: float, a2: float, config: VisionConfig) -> float:
    if max(a1, a2) < config.near_field_area_threshold:
        return config.min_pixel_distance_far_px
    return config.min_pixel_distance_px


def dynamic_min_pixel_distance(a1: float, a2: float, config: VisionConfig) -> float:
    """
    Minimum required inter-LED pixel distance from blob areas.

    d_min = max(d_floor, k_r * (sqrt(A1) + sqrt(A2)))
    """
    r_eff = effective_blob_radius_sum(a1, a2)
    d_floor = _distance_floor(a1, a2, config)
    return max(d_floor, config.dynamic_min_distance_k_r * r_eff)


def dynamic_max_pixel_distance(a1: float, a2: float, config: VisionConfig) -> float:
    """Cap absurd pairs; mild scaling with blob scale (min-only path uses static cap)."""
    if not config.dynamic_max_distance_enabled:
        return config.max_pixel_distance_px

    r_eff = effective_blob_radius_sum(a1, a2)
    r_ref = max(config.dynamic_max_distance_r_ref, 1e-3)
    scale = (r_eff / r_ref) ** config.dynamic_max_distance_alpha
    scaled = config.dynamic_max_distance_d_ref * scale
    return min(config.max_pixel_distance_px, scaled)


def distance_gate_score(
    d: float,
    a1: float,
    a2: float,
    config: VisionConfig,
) -> float:
    """
    Hard reject below d_min; soft ramp in [d_min, d_min + margin]; 1.0 above.
    """
    d_min = dynamic_min_pixel_distance(a1, a2, config)
    if d < d_min:
        return 0.0

    margin = config.dynamic_min_distance_soft_margin_px
    if margin <= 0 or d >= d_min + margin:
        return 1.0

    return (d - d_min) / margin


def passes_distance_gate(
    d: float,
    a1: float,
    a2: float,
    config: VisionConfig,
) -> bool:
    """True if distance is within dynamic min/max bounds."""
    if d <= 0:
        return False
    d_min = dynamic_min_pixel_distance(a1, a2, config)
    d_max = dynamic_max_pixel_distance(a1, a2, config)
    return d_min <= d <= d_max


def _anchor_self_check() -> None:
    """Sanity checks for calibration anchor scenes."""
    cfg = VisionConfig()

    # Test_01 glare sub-blobs (~37 px, large split blobs) — reject
    assert not passes_distance_gate(37.0, 600.0, 600.0, cfg)

    # Test_01 true pair (~168 px) — accept
    assert passes_distance_gate(168.0, 900.0, 900.0, cfg)

    # Test_05 mid-range (~53 px, moderate areas) — accept
    assert passes_distance_gate(53.0, 130.0, 130.0, cfg)

    # Test_06 far (~35 px) — accept
    assert passes_distance_gate(35.0, 40.0, 40.0, cfg)


if __name__ == "__main__":
    _anchor_self_check()
    print("adaptive_geometry anchor checks passed")
