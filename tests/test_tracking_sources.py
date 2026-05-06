import pytest

from src.models import TrackingFrame
from src.tracking.adapter import TrackingAdapter
from src.tracking.factory import create_tracker


def test_tracking_frame_source_defaults():
    frame = TrackingFrame(timestamp=1.0, landmarks=[], is_valid=False)
    assert frame.source == "unknown"
    assert frame.coordinate_system == "image_normalized"


def test_tracker_source_names_are_declared():
    assert TrackingAdapter.source_name == "mediapipe"
    assert TrackingAdapter.requires_video is True


def test_unknown_tracker_rejected():
    with pytest.raises(ValueError):
        create_tracker("unknown")
