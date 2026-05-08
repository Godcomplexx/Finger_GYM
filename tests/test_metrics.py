"""Тесты модуля src/processing/metrics.py"""
import math
import pytest
from src.models import TrackingFrame, Point2D
from src.processing.metrics import (
    compute_palm_width,
    compute_palm_center,
    normalized_distance,
    thumb_index_distance,
    thumb_index_angle_deg,
    avg_tip_to_palm_distance,
    finger_curl,
    all_finger_curls,
    index_finger_curl,
    hand_jitter,
    palm_facing_camera,
    palm_facing_quality,
    back_facing_quality,
    valid_tracking_ratio,
)

# ── Фикстуры ──────────────────────────────────────────────────────────────────

def _make_frame(landmarks: list[tuple[float, float]],
                valid: bool = True,
                hand_label: str = "Right") -> TrackingFrame:
    """Создаёт TrackingFrame из списка (x, y) координат."""
    lms = [Point2D(x=p[0], y=p[1], z=p[2] if len(p) > 2 else 0.0) for p in landmarks]
    return TrackingFrame(timestamp=0.0, landmarks=lms,
                         is_valid=valid, hand_label=hand_label)


def _open_palm_landmarks() -> list[tuple[float, float]]:
    """21 точка, имитирующая открытую ладонь правой руки."""
    # Запястье (0), основания пальцев (5,9,13,17), кончики далеко от ладони
    pts = [(0.5, 0.8)] * 21
    pts[0]  = (0.50, 0.80)   # WRIST
    pts[1]  = (0.40, 0.73)   # THUMB_CMC
    pts[2]  = (0.34, 0.68)   # THUMB_MCP
    pts[3]  = (0.28, 0.62)   # THUMB_IP
    pts[4]  = (0.22, 0.56)   # THUMB_TIP  — далеко
    pts[5]  = (0.42, 0.60)   # INDEX_MCP
    pts[6]  = (0.42, 0.48)
    pts[7]  = (0.42, 0.38)
    pts[8]  = (0.42, 0.28)   # INDEX_TIP  — далеко от ладони
    pts[9]  = (0.50, 0.58)   # MIDDLE_MCP
    pts[10] = (0.50, 0.46)
    pts[11] = (0.50, 0.36)
    pts[12] = (0.50, 0.26)   # MIDDLE_TIP
    pts[13] = (0.58, 0.60)   # RING_MCP
    pts[14] = (0.58, 0.48)
    pts[15] = (0.58, 0.38)
    pts[16] = (0.58, 0.28)   # RING_TIP
    pts[17] = (0.64, 0.64)   # PINKY_MCP
    pts[18] = (0.64, 0.53)
    pts[19] = (0.64, 0.44)
    pts[20] = (0.64, 0.34)   # PINKY_TIP
    return pts


def _fist_landmarks() -> list[tuple[float, float]]:
    """21 точка, имитирующая кулак — кончики пальцев прижаты к ладони."""
    pts = list(_open_palm_landmarks())
    # Кончики пальцев приближаем к основаниям (MCP)
    pts[8]  = (0.44, 0.65)   # INDEX_TIP  ≈ INDEX_MCP
    pts[12] = (0.51, 0.64)   # MIDDLE_TIP ≈ MIDDLE_MCP
    pts[16] = (0.59, 0.65)   # RING_TIP   ≈ RING_MCP
    pts[20] = (0.65, 0.68)   # PINKY_TIP  ≈ PINKY_MCP
    return pts


# ── Тесты palm_width ──────────────────────────────────────────────────────────

class TestPalmWidth:
    def test_invalid_frame_returns_zero(self):
        frame = _make_frame([(0, 0)] * 21, valid=False)
        assert compute_palm_width(frame) == 0.0

    def test_not_enough_landmarks(self):
        frame = TrackingFrame(timestamp=0, landmarks=[], is_valid=True)
        assert compute_palm_width(frame) == 0.0

    def test_positive_for_valid_frame(self):
        frame = _make_frame(_open_palm_landmarks())
        w = compute_palm_width(frame)
        assert w > 0.0

    def test_wrist_pinky_distance(self):
        # WRIST=(0,0), PINKY_MCP=(0.15,0) → width=0.15
        pts = [(0, 0)] * 21
        pts[0]  = (0.0, 0.0)
        pts[17] = (0.15, 0.0)
        frame = _make_frame(pts)
        assert abs(compute_palm_width(frame) - 0.15) < 1e-6


# ── Тесты normalized_distance ─────────────────────────────────────────────────

class TestNormalizedDistance:
    def test_zero_palm_width_returns_zero(self):
        a, b = Point2D(0, 0), Point2D(1, 0)
        assert normalized_distance(a, b, 0.0) == 0.0

    def test_distance_normalized_correctly(self):
        a = Point2D(0.0, 0.0)
        b = Point2D(0.3, 0.0)
        # distance=0.3, palm_width=0.15 → normalized=2.0
        result = normalized_distance(a, b, 0.15)
        assert abs(result - 2.0) < 1e-6


# ── Тесты thumb_index_distance ────────────────────────────────────────────────

class TestThumbIndexDistance:
    def test_invalid_frame(self):
        frame = _make_frame([(0, 0)] * 21, valid=False)
        assert thumb_index_distance(frame, 0.15) == 1.0

    def test_open_palm_large_distance(self):
        frame = _make_frame(_open_palm_landmarks())
        pw = compute_palm_width(frame)
        d = thumb_index_distance(frame, pw)
        # При открытой ладони расстояние большое — не щипок
        assert d > 0.25

    def test_pinch_small_distance(self):
        pts = list(_open_palm_landmarks())
        # Сближаем THUMB_TIP (4) и INDEX_TIP (8)
        pts[4] = (0.43, 0.29)
        pts[8] = (0.43, 0.29)
        frame = _make_frame(pts)
        pw = compute_palm_width(frame)
        d = thumb_index_distance(frame, pw)
        assert d < 0.25


class TestThumbIndexAngle:
    def test_invalid_frame(self):
        frame = _make_frame([(0, 0)] * 21, valid=False)
        assert thumb_index_angle_deg(frame) == 180.0

    def test_pinch_small_angle(self):
        pts = list(_open_palm_landmarks())
        pts[4] = (0.43, 0.29)
        pts[8] = (0.43, 0.29)
        frame = _make_frame(pts)
        assert thumb_index_angle_deg(frame) < 1.0

    def test_open_palm_larger_angle(self):
        frame = _make_frame(_open_palm_landmarks())
        assert thumb_index_angle_deg(frame) > 35.0


# ── Тесты finger_curl ─────────────────────────────────────────────────────────

class TestFingerCurl:
    def test_extended_finger_low_curl(self):
        # Кончик пальца далеко от основания → curl низкий
        tip = Point2D(0.5, 0.2)
        mcp = Point2D(0.5, 0.6)
        pw  = 0.15
        curl = finger_curl(tip, mcp, pw)
        assert curl < 0.4, f"Ожидали curl < 0.4, получили {curl}"

    def test_bent_finger_high_curl(self):
        # Кончик пальца близко к основанию → curl высокий
        tip = Point2D(0.50, 0.62)
        mcp = Point2D(0.50, 0.60)
        pw  = 0.15
        curl = finger_curl(tip, mcp, pw)
        assert curl > 0.5, f"Ожидали curl > 0.5, получили {curl}"

    def test_zero_palm_width(self):
        assert finger_curl(Point2D(0, 0), Point2D(1, 1), 0.0) == 0.0

    def test_curl_in_range(self):
        for tip_y in [0.1, 0.3, 0.5, 0.6, 0.7]:
            c = finger_curl(Point2D(0.5, tip_y), Point2D(0.5, 0.6), 0.15)
            assert 0.0 <= c <= 1.0, f"curl={c} вне диапазона [0,1]"


# ── Тесты all_finger_curls ────────────────────────────────────────────────────

class TestAllFingerCurls:
    def test_returns_four_values(self):
        frame = _make_frame(_open_palm_landmarks())
        pw = compute_palm_width(frame)
        curls = all_finger_curls(frame, pw)
        assert len(curls) == 4

    def test_open_palm_low_curls(self):
        frame = _make_frame(_open_palm_landmarks())
        pw = compute_palm_width(frame)
        curls = all_finger_curls(frame, pw)
        assert all(c < 0.5 for c in curls), f"open palm curls: {curls}"

    def test_fist_high_curls(self):
        frame = _make_frame(_fist_landmarks())
        pw = compute_palm_width(frame)
        curls = all_finger_curls(frame, pw)
        bent = sum(1 for c in curls if c > 0.5)
        assert bent >= 3, f"fist curls: {curls}"

    def test_invalid_frame_returns_zeros(self):
        frame = _make_frame([(0, 0)] * 21, valid=False)
        curls = all_finger_curls(frame, 0.15)
        assert curls == [0.0, 0.0, 0.0, 0.0]


# ── Тесты hand_jitter ─────────────────────────────────────────────────────────

class TestHandJitter:
    def test_empty_list(self):
        assert hand_jitter([]) == 0.0

    def test_single_point_no_jitter(self):
        assert hand_jitter([Point2D(0.5, 0.5)]) == 0.0

    def test_stable_hand_low_jitter(self):
        centers = [Point2D(0.5, 0.5)] * 30
        assert hand_jitter(centers) == 0.0

    def test_unstable_hand_high_jitter(self):
        import random
        random.seed(42)
        centers = [Point2D(random.uniform(0.3, 0.7), random.uniform(0.3, 0.7))
                   for _ in range(30)]
        assert hand_jitter(centers) > 0.05


# ── Тесты valid_tracking_ratio ────────────────────────────────────────────────

class TestValidTrackingRatio:
    def test_empty_list(self):
        assert valid_tracking_ratio([]) == 0.0

    def test_all_valid(self):
        frames = [TrackingFrame(0, [], True) for _ in range(10)]
        assert valid_tracking_ratio(frames) == 1.0

    def test_all_invalid(self):
        frames = [TrackingFrame(0, [], False) for _ in range(10)]
        assert valid_tracking_ratio(frames) == 0.0

    def test_half_valid(self):
        frames = ([TrackingFrame(0, [], True)] * 5 +
                  [TrackingFrame(0, [], False)] * 5)
        assert abs(valid_tracking_ratio(frames) - 0.5) < 1e-6


# ── Тесты palm_facing_camera ──────────────────────────────────────────────────

class TestPalmFacingCamera:
    def test_invalid_frame(self):
        frame = _make_frame([(0, 0)] * 21, valid=False)
        assert palm_facing_camera(frame) is False

    def test_right_hand_palm_facing(self):
        # Правая рука, ладонь к камере: после flip указательный MCP правее мизинца.
        # vi=(+0.1,−0.2,0), vp=(−0.1,−0.2,0) → nz = (+0.1)(−0.2)−(−0.2)(−0.1) = −0.04
        # Right + nz < 0 → True ✓
        pts = [(0.5, 0.7)] * 21
        pts[0]  = (0.50, 0.70)  # WRIST
        pts[5]  = (0.60, 0.50)  # INDEX_MCP (правее)
        pts[17] = (0.40, 0.50)  # PINKY_MCP (левее)
        frame = _make_frame(pts, hand_label="Right")
        assert palm_facing_camera(frame) is True

    def test_right_hand_back_facing(self):
        # Правая рука, тыл к камере: указательный MCP левее мизинца.
        # vi=(−0.1,−0.2,0), vp=(+0.1,−0.2,0) → nz = +0.04
        # Right + nz > 0 → False ✓
        pts = [(0.5, 0.7)] * 21
        pts[0]  = (0.50, 0.70)  # WRIST
        pts[5]  = (0.40, 0.50)  # INDEX_MCP (левее)
        pts[17] = (0.60, 0.50)  # PINKY_MCP (правее)
        frame = _make_frame(pts, hand_label="Right")
        assert palm_facing_camera(frame) is False

    def test_palm_facing_quality_full(self):
        pts = [(0.5, 0.7, 0.0)] * 21
        pts[0] = (0.50, 0.70, 0.0)
        pts[5] = (0.60, 0.50, 0.0)
        pts[17] = (0.40, 0.50, 0.0)
        frame = _make_frame(pts, hand_label="Right")
        assert palm_facing_quality(frame) == pytest.approx(1.0)
        assert back_facing_quality(frame) == 0.0

    def test_back_facing_quality_full(self):
        pts = [(0.5, 0.7, 0.0)] * 21
        pts[0] = (0.50, 0.70, 0.0)
        pts[5] = (0.40, 0.50, 0.0)
        pts[17] = (0.60, 0.50, 0.0)
        frame = _make_frame(pts, hand_label="Right")
        assert back_facing_quality(frame) == pytest.approx(1.0)
        assert palm_facing_quality(frame) == 0.0

    def test_edge_on_quality_is_zero(self):
        pts = [(0.5, 0.7, 0.0)] * 21
        pts[0] = (0.50, 0.70, 0.0)
        pts[5] = (0.50, 0.50, 0.10)
        pts[17] = (0.50, 0.50, -0.10)
        frame = _make_frame(pts, hand_label="Right")
        assert palm_facing_quality(frame) == pytest.approx(0.0)
        assert back_facing_quality(frame) == pytest.approx(0.0)
