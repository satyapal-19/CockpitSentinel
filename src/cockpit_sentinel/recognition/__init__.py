"""Face detection, recognition, and auto-cabin profile identification."""

from cockpit_sentinel.recognition.face_identifier import (
    FaceIdentifier,
    compute_similarity,
    get_face_identifier,
)

__all__ = ["FaceIdentifier", "compute_similarity", "get_face_identifier"]
