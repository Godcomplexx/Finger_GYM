from __future__ import annotations

import os
import time

import cv2
import numpy as np


class SessionVideoRecorder:
    def __init__(self, path: str, width: int, height: int, fps: float = 30.0):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.started_at = time.time()
        self.frames_written = 0
        self._last_frame: np.ndarray | None = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        self._last_frame = frame_bgr.copy()
        elapsed = max(0.0, time.time() - self.started_at)
        target_frames = max(1, int(elapsed * self.fps))
        while self.frames_written < target_frames:
            self._write_frame(frame_bgr)

    def _write_frame(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)
        self.frames_written += 1

    def close(self) -> dict:
        duration = max(0.0, time.time() - self.started_at)
        if self._last_frame is not None:
            target_frames = max(self.frames_written, int(duration * self.fps))
            while self.frames_written < target_frames:
                self._write_frame(self._last_frame)
        self._writer.release()
        return {
            "path": self.path,
            "fps": self.fps,
            "frames": self.frames_written,
            "durationSec": round(duration, 2),
        }
