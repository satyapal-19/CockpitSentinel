"""Object-detection components for driver distraction."""

from cockpit_sentinel.detection.distraction import (
    DetectionBox,
    DistractionAnalysis,
    DistractionConfig,
    DistractionDetector,
    load_distraction_config,
)

__all__ = [
    "DetectionBox",
    "DistractionAnalysis",
    "DistractionConfig",
    "DistractionDetector",
    "load_distraction_config",
]
