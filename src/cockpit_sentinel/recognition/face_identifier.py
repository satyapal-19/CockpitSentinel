"""Face recognition and automatic in-cabin driver identification engine."""

from __future__ import annotations

import logging
from pathlib import Path

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

        # Check for lightweight ONNX FaceNet models (quantized INT8 or FP32)
        self._ort_session = None
        self._ort_input_name = None
        self._dnn_model = None

        int8_candidates = [
            Path("weights/pretrained/facenet_int8.onnx"),
            Path(__file__).resolve().parents[3] / "weights" / "pretrained" / "facenet_int8.onnx",
        ]
        fp32_candidates = [
            Path("weights/pretrained/facenet.onnx"),
            Path(__file__).resolve().parents[3] / "weights" / "pretrained" / "facenet.onnx",
        ]

        # 1. Try ONNX Runtime with quantized INT8 model (~22.8MB, ~89MB peak RAM, sub-20ms)
        try:
            import onnxruntime as ort

            for opath in int8_candidates:
                if opath.exists():
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = 2
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self._ort_session = ort.InferenceSession(
                        str(opath), sess_options=sess_options, providers=["CPUExecutionProvider"]
                    )
                    self._ort_input_name = self._ort_session.get_inputs()[0].name
                    logger.info("Loaded quantized FaceNet INT8 model via ONNX Runtime: %s", opath)
                    break
        except Exception as e:
            logger.debug("ONNX Runtime INT8 load notice: %s", e)

        # 2. Try OpenCV cv2.dnn with FP32 ONNX model if INT8 session not loaded
        if self._ort_session is None:
            for opath in fp32_candidates:
                if opath.exists():
                    try:
                        self._dnn_model = cv2.dnn.readNetFromONNX(str(opath))
                        logger.info("Loaded FaceNet ONNX model via cv2.dnn: %s", opath)
                        break
                    except Exception as e:
                        logger.warning("Failed to load FaceNet ONNX model: %s", e)

        # Lazy-loaded PyTorch models to conserve memory on 512MB cloud instances
        self._model: InceptionResnetV1 | None = None
        self._mtcnn: MTCNN | None = None

        if self._ort_session is None and self._dnn_model is None:
            # If neither ONNX is available, initialize PyTorch InceptionResnetV1
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # OpenCV Haar face detector (fast ~5ms, zero extra memory)
        self._haar_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._haar_cascade = None

    @property
    def model(self) -> InceptionResnetV1:
        if self._model is None:
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        return self._model

    @property
    def mtcnn(self) -> MTCNN:
        if self._mtcnn is None:
            self._mtcnn = MTCNN(
                image_size=160,
                margin=14,
                keep_all=False,
                device=self.device,
                post_process=True,
            )
        return self._mtcnn

    def _forward_face_tensor(self, face_tensor: torch.Tensor | np.ndarray) -> np.ndarray | None:
        """Forward a [3, 160, 160] or [1, 3, 160, 160] face tensor through FaceNet."""
        try:
            if isinstance(face_tensor, torch.Tensor):
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                face_np = face_tensor.detach().cpu().numpy().astype(np.float32)
            else:
                face_np = np.asarray(face_tensor, dtype=np.float32)
                if face_np.ndim == 3:
                    if face_np.shape[2] == 3:  # H, W, C -> C, H, W
                        face_np = np.transpose(face_np, (2, 0, 1))
                    face_np = np.expand_dims(face_np, axis=0)

            if self._ort_session is not None:
                embedding = self._ort_session.run(None, {self._ort_input_name: face_np})[0].flatten()
            elif self._dnn_model is not None:
                self._dnn_model.setInput(face_np)
                embedding = self._dnn_model.forward().flatten()
            else:
                t = torch.from_numpy(face_np).to(self.device)
                with torch.inference_mode():
                    embedding = self.model(t).cpu().numpy().flatten()

            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                return embedding / norm
        except Exception as exc:
            logger.debug("Forward face embedding exception: %s", exc)
        return None

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

        h, w = rgb_image.shape[:2]

        # 1. Fast Path: If input is already an explicit cropped thumbnail (e.g. 160x160 from frontend/MediaPipe)
        if w <= 220 and h <= 220 and 0.7 <= (w / max(1, h)) <= 1.4 and min(h, w) >= 40:
            try:
                resized = cv2.resize(rgb_image, (160, 160))
                normalized = (resized.astype(np.float32) - 127.5) / 128.0
                emb = self._forward_face_tensor(normalized)
                if emb is not None:
                    return emb
            except Exception as exc:
                logger.debug("Cropped thumbnail extraction exception: %s", exc)

        # Downscale high-resolution frames to max 720px for fast face detection
        if max(h, w) > 720:
            scale = 720.0 / max(h, w)
            rgb_image = cv2.resize(rgb_image, (int(w * scale), int(h * scale)))
            h, w = rgb_image.shape[:2]

        # 2. Primary for full images: OpenCV Haar Cascade detector (Fast ~5ms, zero extra memory)
        if self._haar_cascade is not None:
            try:
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                faces = self._haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
                )
                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    # Add 25% margin around Haar box
                    pad_x = int(fw * 0.25)
                    pad_y = int(fh * 0.25)
                    y1 = max(0, y - pad_y)
                    y2 = min(rgb_image.shape[0], y + fh + pad_y)
                    x1 = max(0, x - pad_x)
                    x2 = min(rgb_image.shape[1], x + fw + pad_x)
                    face_crop = rgb_image[y1:y2, x1:x2]

                    resized = cv2.resize(face_crop, (160, 160))
                    normalized = (resized.astype(np.float32) - 127.5) / 128.0
                    emb = self._forward_face_tensor(normalized)
                    if emb is not None:
                        return emb
            except Exception as exc:
                logger.debug("Haar cascade face extraction exception: %s", exc)

        # 3. Fallback: MTCNN face detection (only if Haar didn't detect face and MTCNN available)
        try:
            face_tensor = self.mtcnn(rgb_image)
            if face_tensor is not None:
                embedding = self._forward_face_tensor(face_tensor)
                if embedding is not None:
                    return embedding
        except Exception as exc:
            logger.debug("MTCNN fallback detection exception: %s", exc)

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
            if not getattr(driver, "face_registered", False) or not getattr(driver, "face_embedding", None):
                continue
            sim = self.compute_similarity(current_emb, driver.face_embedding)
            if sim > best_score:
                best_score = sim
                best_driver = driver

        if best_driver is not None and best_score >= thresh:
            return best_driver, best_score

        return None, max(0.0, best_score)

    def identify_from_frames(
        self,
        frames: list[np.ndarray],
        candidates: list[DriverProfile],
        threshold: float | None = None,
    ) -> tuple[DriverProfile | None, float]:
        """Match multiple cabin video frames against enrolled driver profiles using averaged embeddings.

        Args:
            frames: Sequence of video frames (OpenCV BGR).
            candidates: List of registered driver profiles from database.
            threshold: Minimum cosine similarity to accept match.

        Returns:
            (best_matched_driver, confidence_score) or (None, best_score).
        """
        if not frames or not candidates:
            return None, 0.0

        embeddings: list[np.ndarray] = []
        for f in frames:
            emb = self.extract_embedding(f)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None, 0.0

        avg_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 1e-6:
            avg_emb = avg_emb / norm

        thresh = threshold if threshold is not None else self.similarity_threshold
        best_driver: DriverProfile | None = None
        best_score: float = -1.0

        for driver in candidates:
            if not getattr(driver, "face_registered", False) or not getattr(driver, "face_embedding", None):
                continue
            sim = self.compute_similarity(avg_emb, driver.face_embedding)
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
