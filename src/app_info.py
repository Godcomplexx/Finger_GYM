from __future__ import annotations

import hashlib
import os
import sys


MODULE_NAME = "Finger GYM"
MODULE_VERSION = "0.2.0"
ALGORITHM_VERSION = "ruleset-2026-05-05"
MODEL_NAME = "MediaPipe Hand Landmarker"
TRACKING_MODEL_FILE = "hand_landmarker.task"


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resource_root() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return project_root()


def runtime_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return project_root()


def tracking_model_path() -> str:
    return os.path.join(resource_root(), TRACKING_MODEL_FILE)


def sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracking_model_sha256() -> str | None:
    return sha256_file(tracking_model_path())
