from __future__ import annotations

from typing import Protocol

import numpy as np

from src.models import TrackingFrame


class HandTracker(Protocol):
    source_name: str
    model_name: str
    model_sha256: str | None
    requires_video: bool

    def process(self, bgr_frame: np.ndarray | None = None) -> TrackingFrame:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> "HandTracker":
        ...

    def __exit__(self, *args) -> None:
        ...
