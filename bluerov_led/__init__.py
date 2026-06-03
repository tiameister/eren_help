"""BlueROV2 LED-based visual tracking — offline pipeline and observation packets."""

from bluerov_led.config import FACE_PATTERNS, ProjectPaths, VisionConfig
from bluerov_led.pipeline import BackFacePipeline, PipelineResult
from bluerov_led.validation import ValidationRunner, ValidationThresholds

__all__ = [
    "FACE_PATTERNS",
    "VisionConfig",
    "ProjectPaths",
    "BackFacePipeline",
    "PipelineResult",
    "ValidationRunner",
    "ValidationThresholds",
]
