from types import SimpleNamespace

import pytest

from src.models import TrackingFrame
from src.tracking.adapter import TrackingAdapter
from src.tracking.factory import create_tracker
from src.tracking.ultraleap_adapter import (
    UltraleapTrackingAdapter,
    UltraleapWorkspace,
    normalize_ultraleap_vector,
)


def test_tracking_frame_source_defaults():
    frame = TrackingFrame(timestamp=1.0, landmarks=[], is_valid=False)
    assert frame.source == "unknown"
    assert frame.coordinate_system == "image_normalized"


def test_tracker_source_names_are_declared():
    assert TrackingAdapter.source_name == "mediapipe"
    assert UltraleapTrackingAdapter.source_name == "ultraleap"
    assert TrackingAdapter.requires_video is True
    assert UltraleapTrackingAdapter.requires_video is False


def test_unknown_tracker_rejected():
    with pytest.raises(ValueError):
        create_tracker("unknown")


def test_ultraleap_vector_normalization():
    workspace = UltraleapWorkspace(width_mm=300, height_mm=300, center_y_mm=200)
    point = normalize_ultraleap_vector(
        SimpleNamespace(x=0, y=200, z=30),
        workspace,
    )
    assert point.x == pytest.approx(0.5)
    assert point.y == pytest.approx(0.5)
    assert point.z == pytest.approx(0.1)
