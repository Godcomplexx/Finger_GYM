from __future__ import annotations

from src.tracking.base import HandTracker


def create_tracker(name: str) -> HandTracker:
    tracker_name = name.strip().lower()
    if tracker_name == "mediapipe":
        from src.tracking.adapter import TrackingAdapter

        return TrackingAdapter()
    if tracker_name == "ultraleap":
        from src.tracking.ultraleap_adapter import UltraleapTrackingAdapter

        return UltraleapTrackingAdapter()
    raise ValueError(f"Unknown tracker: {name}")
