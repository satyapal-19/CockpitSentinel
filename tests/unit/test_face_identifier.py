"""Unit tests for FaceIdentifier biometric face recognition module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from cockpit_sentinel.database.models import DriverProfile, DriverThresholds
from cockpit_sentinel.recognition import FaceIdentifier, compute_similarity, get_face_identifier

SAMPLE_FACE_PATH = Path("data/external/nthu/train/notdrowsy/001_glasses_nonsleepyCombination_0_notdrowsy.jpg")


def test_compute_similarity_identical_vectors():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(compute_similarity(v1, v2), 1e-5) == 1.0


def test_compute_similarity_orthogonal_vectors():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert pytest.approx(compute_similarity(v1, v2), 1e-5) == 0.0


def test_compute_similarity_opposite_vectors():
    v1 = [1.0, 0.0, 0.0]
    v2 = [-1.0, 0.0, 0.0]
    assert pytest.approx(compute_similarity(v1, v2), 1e-5) == -1.0


def test_compute_similarity_empty_or_mismatched():
    assert compute_similarity([], [1.0, 2.0]) == 0.0
    assert compute_similarity([1.0], []) == 0.0
    assert compute_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_face_identifier_singleton():
    f1 = get_face_identifier()
    f2 = get_face_identifier()
    assert f1 is f2
    assert isinstance(f1, FaceIdentifier)


def test_identify_driver_with_mocked_extractor():
    identifier = FaceIdentifier()
    # Mock extract_embedding to return a fixed 512-D unit vector
    target_vec = [1.0] + [0.0] * 511
    identifier.extract_embedding = MagicMock(return_value=target_vec)

    driver1 = DriverProfile(
        id="drv-1",
        name="Driver One",
        license_number="DL-01",
        face_embedding=target_vec,
        face_registered=True,
    )
    driver2 = DriverProfile(
        id="drv-2",
        name="Driver Two",
        license_number="DL-02",
        face_embedding=[0.0, 1.0] + [0.0] * 510,
        face_registered=True,
    )
    driver3 = DriverProfile(
        id="drv-3",
        name="Driver Three",
        license_number="DL-03",
        face_embedding=None,
        face_registered=False,
    )

    candidates = [driver1, driver2, driver3]
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # Should match driver 1 with similarity ~ 1.0
    matched, score = identifier.identify_driver(dummy_frame, candidates, threshold=0.55)
    assert matched is not None
    assert matched.id == "drv-1"
    assert pytest.approx(score, 1e-4) == 1.0


def test_identify_driver_below_threshold():
    identifier = FaceIdentifier()
    query_vec = [1.0] + [0.0] * 511
    identifier.extract_embedding = MagicMock(return_value=query_vec)

    # Only candidate has orthogonal vector (similarity = 0.0)
    candidate = DriverProfile(
        id="drv-unmatched",
        name="Other Person",
        license_number="DL-99",
        face_embedding=[0.0, 1.0] + [0.0] * 510,
        face_registered=True,
    )

    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    matched, score = identifier.identify_driver(dummy_frame, [candidate], threshold=0.55)
    assert matched is None
    assert score == 0.0


def test_identify_driver_no_candidates():
    identifier = FaceIdentifier()
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    matched, score = identifier.identify_driver(dummy_frame, [], threshold=0.55)
    assert matched is None
    assert score == 0.0


def test_enroll_driver_face():
    identifier = FaceIdentifier()
    mock_vec = [0.1] * 512
    identifier.extract_embedding = MagicMock(return_value=mock_vec)

    driver = DriverProfile(
        id="drv-enroll-test",
        name="Enroll Candidate",
        license_number="DL-TEST-01",
        face_embedding=None,
        face_registered=False,
    )

    dummy_frame = np.zeros((160, 160, 3), dtype=np.uint8)
    updated = identifier.enroll_driver_face(driver, dummy_frame)

    assert updated.face_registered is True
    assert updated.face_embedding is not None
    assert len(updated.face_embedding) == 512
    # Enrollment normalizes vectors to unit length
    assert pytest.approx(np.linalg.norm(updated.face_embedding), 1e-4) == 1.0
    assert pytest.approx(compute_similarity(updated.face_embedding, mock_vec), 1e-4) == 1.0


@pytest.mark.skipif(not SAMPLE_FACE_PATH.exists(), reason="Sample face dataset image not available")
def test_real_image_embedding_extraction():
    img = cv2.imread(str(SAMPLE_FACE_PATH))
    assert img is not None

    identifier = get_face_identifier()
    emb = identifier.extract_embedding(img)
    assert emb is not None
    assert len(emb) == 512
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, 1e-4) == 1.0
