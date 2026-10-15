"""Face recognition and automatic in-cabin driver identification engine."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from cockpit_sentinel.database.models import DriverProfile

logger = logging.getLogger(__name__)


class FaceIdentifier:
    """Biometric facial identification engine for in-cabin driver recognition."""

    def __init__(
        self,
        device: str | None = None,
        similarity_threshold: float = 0.65,
    ) -> None:
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.similarity_threshold = similarity_threshold

        logger.info("Initializing FaceIdentifier on device: %s", self.device)
        self._mtcnn = MTCNN(
            image_size=160,
            margin=14,
            keep_all=False,
            device=self.device,
            post_process=True,
        )
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # OpenCV Haar fallback face detector
        self._haar_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._haar_cascade = None

    def extract_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """Extract a 512-dimensional L2-normalized face embedding from a frame.

        Args:
            image: BGR or RGB image numpy array (H, W, 3).

        Returns:
            Normalized 1D numpy array of shape (512,) or None if no face is detected.
        """
        if image is None or image.size == 0:
            return None

        # Convert to RGB if 3 channels
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        try:
            # 1. Primary: Try MTCNN aligned crop
            face_tensor = self._mtcnn(rgb_image)
            if face_tensor is not None:
                if len(face_tensor.shape) == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                face_tensor = face_tensor.to(self.device)
                with torch.no_grad():
                    embedding = self._model(face_tensor).cpu().numpy().flatten()
                norm = np.linalg.norm(embedding)
                if norm > 1e-6:
                    return embedding / norm
        except Exception as exc:
            logger.debug("MTCNN detection exception: %s", exc)

        # 2. Secondary Fallback: OpenCV Haar Cascade + aligned crop
        if self._haar_cascade is not None:
            try:
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                faces = self._haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
                )
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    # Add 25% margin around Haar box and try MTCNN alignment on crop
                    pad_x = int(w * 0.25)
                    pad_y = int(h * 0.25)
                    y1 = max(0, y - pad_y)
                    y2 = min(rgb_image.shape[0], y + h + pad_y)
                    x1 = max(0, x - pad_x)
                    x2 = min(rgb_image.shape[1], x + w + pad_x)
                    face_crop = rgb_image[y1:y2, x1:x2]

                    try:
                        face_tensor = self._mtcnn(face_crop)
                        if face_tensor is not None:
                            if len(face_tensor.shape) == 3:
                                face_tensor = face_tensor.unsqueeze(0)
                            face_tensor = face_tensor.to(self.device)
                            with torch.no_grad():
                                embedding = self._model(face_tensor).cpu().numpy().flatten()
                            norm = np.linalg.norm(embedding)
                            if norm > 1e-6:
                                return embedding / norm
                    except Exception:
                        pass

                    # Fallback to direct resize if MTCNN on crop also failed
                    crop = rgb_image[y : y + h, x : x + w]
                    resized = cv2.resize(crop, (160, 160))
                    tensor = (
                        torch.from_numpy(resized)
                        .permute(2, 0, 1)
                        .float()
                        .sub(127.5)
                        .div(128.0)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    with torch.no_grad():
                        embedding = self._model(tensor).cpu().numpy().flatten()
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-6:
                        return embedding / norm
            except Exception as exc:
                logger.debug("Haar cascade face extraction exception: %s", exc)

        return None

    @staticmethod
    def compute_similarity(
        emb1: np.ndarray | list[float], emb2: np.ndarray | list[float]
    ) -> float:
        """Calculate cosine similarity between two face embeddings [-1.0, 1.0]."""
        v1 = np.asarray(emb1, dtype=np.float32)
        v2 = np.asarray(emb2, dtype=np.float32)
        if v1.shape != v2.shape:
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        dot = float(np.dot(v1, v2) / (norm1 * norm2))
        return max(-1.0, min(1.0, dot))

    def identify_driver(
        self,
        frame: np.ndarray,
        candidates: list[DriverProfile],
        threshold: float | None = None,
    ) -> tuple[DriverProfile | None, float]:
        """Match current cabin video frame against enrolled driver face profiles.

        Args:
            frame: Video frame (OpenCV BGR).
            candidates: List of registered driver profiles from database.
            threshold: Minimum cosine similarity to accept match (default: 0.60).

        Returns:
            (best_matched_driver, confidence_score) or (None, best_score).
        """
        thresh = threshold if threshold is not None else self.similarity_threshold
        current_emb = self.extract_embedding(frame)
        if current_emb is None:
            return None, 0.0

        best_driver: DriverProfile | None = None
        best_score: float = -1.0

        for driver in candidates:
            if not driver.face_registered or not driver.face_embedding:
                continue
            sim = self.compute_similarity(current_emb, driver.face_embedding)
            if sim > best_score:
                best_score = sim
                best_driver = driver

        if best_driver is not None and best_score >= thresh:
            return best_driver, best_score

        return None, max(0.0, best_score)

    def enroll_driver_face(
        self,
        driver: DriverProfile,
        frames_or_images: np.ndarray | list[np.ndarray],
    ) -> DriverProfile:
        """Enroll or update driver biometric face embedding in their profile."""
        if isinstance(frames_or_images, np.ndarray):
            frames_list = [frames_or_images]
        else:
            frames_list = list(frames_or_images)

        embeddings: list[np.ndarray] = []
        for img in frames_list:
            emb = self.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            raise ValueError("No valid face could be detected in the provided image(s).")

        # Average embeddings across all captured frames for enhanced robustness
        avg_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 1e-6:
            avg_emb = avg_emb / norm

        driver.face_embedding = [float(x) for x in avg_emb]
        driver.face_registered = True
        return driver


# Singleton helper
_face_identifier_instance: FaceIdentifier | None = None


def get_face_identifier() -> FaceIdentifier:
    """Return global FaceIdentifier instance."""
    global _face_identifier_instance
    if _face_identifier_instance is None:
        _face_identifier_instance = FaceIdentifier()
    return _face_identifier_instance


compute_similarity = FaceIdentifier.compute_similarity
