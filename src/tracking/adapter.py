from __future__ import annotations
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

from src.models import TrackingFrame, Point2D
from src.app_info import tracking_model_sha256

# Путь к модели относительно корня проекта
_HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_HERE, "..", "..", "hand_landmarker.task")

# Индексы ключевых точек MediaPipe Hands
WRIST = 0
THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

# Соединения для рисования скелета
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class TrackingAdapter:
    source_name = "mediapipe"
    model_name = "MediaPipe Hand Landmarker"
    requires_video = True

    """Обёртка над MediaPipe HandLandmarker (Tasks API >= 0.10)."""

    def __init__(self, model_path: str = MODEL_PATH,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                "Скачайте: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self.model_path = os.path.abspath(model_path)
        self.model_sha256 = tracking_model_sha256()
        self._frame_ts_ms: int = 0   # монотонная временная метка в мс

    def process(self, bgr_frame: np.ndarray | None = None) -> TrackingFrame:
        if bgr_frame is None:
            return TrackingFrame(
                timestamp=time.monotonic(),
                landmarks=[],
                is_valid=False,
                source=self.source_name,
            )

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 33   # ~30 fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        ts = time.monotonic()

        if not result.hand_landmarks:
            return TrackingFrame(
                timestamp=ts,
                landmarks=[],
                is_valid=False,
                source=self.source_name,
            )

        lms = result.hand_landmarks[0]
        label = "Unknown"
        if result.handedness:
            label = result.handedness[0][0].category_name  # "Left" / "Right"

        landmarks = [Point2D(x=lm.x, y=lm.y, z=lm.z) for lm in lms]

        return TrackingFrame(
            timestamp=ts,
            landmarks=landmarks,
            is_valid=True,
            hand_label=label,
            source=self.source_name,
            coordinate_system="image_normalized",
        )

    def draw_landmarks(self, bgr_frame: np.ndarray,
                       frame: TrackingFrame) -> np.ndarray:
        if not frame.is_valid:
            return bgr_frame
        h, w = bgr_frame.shape[:2]
        img = bgr_frame.copy()
        pts = [(int(p.x * w), int(p.y * h)) for p in frame.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (80, 220, 80), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, (255, 255, 255), -1)
            cv2.circle(img, p, 4, (80, 180, 80), 1)
        return img

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
