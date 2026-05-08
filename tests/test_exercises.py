"""Тесты модуля src/exercises/ — логика распознавания поз и оценки."""
import time
import pytest
from src.models import (
    TrackingFrame, Point2D, CalibrationProfile, ExerciseStatus,
)
from src.exercises.exercises import (
    OpenPalmExercise, FistExercise, PinchExercise,
    PointGestureExercise, PalmFacingExercise, BackFacingExercise,
    ZoneMovementExercise,
    create_exercises,
)


# ── Фикстуры ──────────────────────────────────────────────────────────────────

def _calib(palm_width: float = 0.15) -> CalibrationProfile:
    return CalibrationProfile(
        palm_width=palm_width,
        palm_center=Point2D(0.5, 0.5),
        base_tip_to_palm=1.2,
        base_thumb_index=0.8,
        is_ready=True,
    )


def _frame(landmarks: list[tuple[float, float]],
           valid: bool = True,
           hand_label: str = "Right") -> TrackingFrame:
    lms = [Point2D(x=p[0], y=p[1], z=p[2] if len(p) > 2 else 0.0) for p in landmarks]
    return TrackingFrame(timestamp=time.monotonic(),
                         landmarks=lms, is_valid=valid, hand_label=hand_label)


def _open_palm_pts() -> list[tuple[float, float]]:
    # Полностью разогнутая ладонь: MCPs → PIPs → DIPs → Tips выстроены вертикально вверх
    pts = [(0.5, 0.8)] * 21
    pts[0]  = (0.50, 0.80)  # WRIST
    pts[4]  = (0.22, 0.56)  # THUMB_TIP
    # Указательный: MCP=5, PIP=6, DIP=7, TIP=8
    pts[5]  = (0.42, 0.60); pts[6]  = (0.42, 0.48); pts[7]  = (0.42, 0.38); pts[8]  = (0.42, 0.28)
    # Средний: MCP=9, PIP=10, DIP=11, TIP=12
    pts[9]  = (0.50, 0.58); pts[10] = (0.50, 0.46); pts[11] = (0.50, 0.36); pts[12] = (0.50, 0.26)
    # Безымянный: MCP=13, PIP=14, DIP=15, TIP=16
    pts[13] = (0.58, 0.60); pts[14] = (0.58, 0.48); pts[15] = (0.58, 0.38); pts[16] = (0.58, 0.28)
    # Мизинец: MCP=17, PIP=18, DIP=19, TIP=20
    pts[17] = (0.64, 0.64); pts[18] = (0.64, 0.53); pts[19] = (0.64, 0.44); pts[20] = (0.64, 0.34)
    return pts


def _fist_pts() -> list[tuple[float, float]]:
    # Кулак: кончики и PIP согнуты к ладони
    pts = list(_open_palm_pts())
    pts[4] = (0.56, 0.75)
    for mcp_idx, pip_idx, dip_idx, tip_idx in [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]:
        mcp_x, mcp_y = pts[mcp_idx]
        pts[pip_idx] = (mcp_x, mcp_y + 0.10)
        pts[dip_idx] = (mcp_x + 0.10, mcp_y + 0.10)
        pts[tip_idx] = (mcp_x + 0.151, mcp_y + 0.161)
    return pts


def _pinch_pts() -> list[tuple[float, float]]:
    pts = list(_open_palm_pts())
    pts[4] = (0.43, 0.29)   # THUMB_TIP
    pts[8] = (0.43, 0.29)   # INDEX_TIP (вплотную)
    return pts


def _point_pts() -> list[tuple[float, float]]:
    pts = list(_fist_pts())
    pts[1] = (0.40, 0.60)
    pts[2] = (0.32, 0.60)
    pts[3] = (0.28, 0.62)
    pts[4] = (0.24, 0.64)
    # Указательный полностью разгибаем (MCP, PIP, DIP, TIP в линию вверх)
    pts[6] = (0.42, 0.48); pts[7] = (0.42, 0.38); pts[8] = (0.42, 0.28)
    return pts


def _shift_pts(points: list[tuple[float, float]], dx: float) -> list[tuple[float, float]]:
    return [(x + dx, y) for x, y in points]


def _orientation_pts(kind: str) -> list[tuple[float, float, float]]:
    pts = [(0.50, 0.70, 0.0)] * 21
    pts[0] = (0.50, 0.70, 0.0)
    if kind == "palm":
        pts[5] = (0.60, 0.50, 0.0)
        pts[17] = (0.40, 0.50, 0.0)
    elif kind == "back":
        pts[5] = (0.40, 0.50, 0.0)
        pts[17] = (0.60, 0.50, 0.0)
    elif kind == "edge":
        pts[5] = (0.50, 0.50, 0.10)
        pts[17] = (0.50, 0.50, -0.10)
    else:
        raise ValueError(kind)
    pts[8] = (pts[5][0], 0.25, pts[5][2])
    pts[9] = (0.50, 0.50, 0.0)
    pts[12] = (0.50, 0.25, 0.0)
    pts[13] = (0.52, 0.50, 0.0)
    pts[16] = (0.52, 0.25, 0.0)
    pts[20] = (pts[17][0], 0.25, pts[17][2])
    return pts


# ── Тесты create_exercises ────────────────────────────────────────────────────

def test_create_exercises_count():
    exs = create_exercises(_calib())
    assert len(exs) == 8


def test_create_exercises_ids():
    ids = [e.exercise_id for e in create_exercises(_calib())]
    assert "open_palm" in ids
    assert "fist" in ids
    assert "pinch" in ids
    assert "zone_movement" in ids
    assert "hold_still" in ids


# ── Тесты OpenPalmExercise ────────────────────────────────────────────────────

class TestOpenPalmExercise:
    def test_pose_detected_open_hand(self):
        ex = OpenPalmExercise(_calib())
        f  = _frame(_open_palm_pts())
        assert ex._pose_detected(f) is True

    def test_pose_not_detected_fist(self):
        ex = OpenPalmExercise(_calib())
        f  = _frame(_fist_pts())
        assert ex._pose_detected(f) is False

    def test_evaluate_unreliable_on_no_tracking(self):
        ex = OpenPalmExercise(_calib())
        ex._prepare_start -= 10
        ex._prepare_confirmed = True
        for _ in range(20):
            ex.feed(TrackingFrame(time.monotonic(), [], False))
        result = ex.evaluate()
        assert result.status == ExerciseStatus.UNRELIABLE
        assert result.score == 0

    def test_evaluate_done_after_hold(self):
        ex = OpenPalmExercise(_calib())
        ex.min_hold_sec = 0.0
        ex.required_hold_sec = 0.0
        ex._prepare_start -= 10   # пропускаем фазу подготовки
        ex._prepare_confirmed = True
        f = _frame(_open_palm_pts())
        for _ in range(30):
            ex.feed(f)
        result = ex.evaluate()
        assert result.valid_tracking_ratio > 0.6

    def test_entry_motion_is_not_recorded_before_stable_arm(self):
        ex = OpenPalmExercise(_calib())
        ex._prepare_start -= 10
        ex._prepare_confirmed = True
        points = _open_palm_pts()

        for dx in [0.00, 0.06, -0.05, 0.04, -0.04, 0.03, -0.03]:
            ex.feed(_frame(_shift_pts(points, dx)))
        assert len(ex._frames) == 0

        for _ in range(8):
            ex.feed(_frame(points))
        assert len(ex._frames) == 1


# ── Тесты FistExercise ────────────────────────────────────────────────────────

class TestFistExercise:
    def test_pose_detected_fist(self):
        ex = FistExercise(_calib())
        f  = _frame(_fist_pts())
        assert ex._pose_detected(f) is True

    def test_pose_not_detected_open_palm(self):
        ex = FistExercise(_calib())
        f  = _frame(_open_palm_pts())
        assert ex._pose_detected(f) is False
        assert ex._pose_quality(f) == 0.0

    def test_full_fist_quality_is_perfect(self):
        ex = FistExercise(_calib())
        f = _frame(_fist_pts())
        assert ex._pose_quality(f) == 1.0

    def test_max_score_is_15(self):
        ex = FistExercise(_calib())
        assert ex.max_score == 15

    def test_partial_fist_has_continuous_quality(self):
        ex = FistExercise(_calib())
        pts = _fist_pts()
        pts[6] = (0.42, 0.48); pts[7] = (0.42, 0.38); pts[8] = (0.42, 0.28)
        pts[10] = (0.50, 0.46); pts[11] = (0.50, 0.36); pts[12] = (0.50, 0.26)
        f = _frame(pts)

        assert ex._pose_detected(f) is False
        assert 0.0 < ex._pose_quality(f) < 1.0


# ── Тесты PinchExercise ───────────────────────────────────────────────────────

class TestPinchExercise:
    def test_pose_detected_pinch(self):
        ex = PinchExercise(_calib())
        f  = _frame(_pinch_pts())
        assert ex._pose_detected(f) is True

    def test_pinch_with_small_gap_scores_by_angle(self):
        ex = PinchExercise(_calib())
        pts = list(_open_palm_pts())
        pts[4] = (0.45, 0.32)
        pts[8] = (0.46, 0.30)
        f = _frame(pts)

        assert ex._pose_detected(f) is True
        assert ex._pose_quality(f) > 0.95

    def test_pose_not_detected_open_palm(self):
        ex = PinchExercise(_calib())
        f  = _frame(_open_palm_pts())
        assert ex._pose_detected(f) is False

    def test_max_score_is_15(self):
        ex = PinchExercise(_calib())
        assert ex.max_score == 15


# ── Тесты PointGestureExercise ────────────────────────────────────────────────

class TestPointGestureExercise:
    def test_pose_detected_point(self):
        ex = PointGestureExercise(_calib())
        f  = _frame(_point_pts())
        assert ex._pose_detected(f) is True
        assert ex._pose_quality(f) == 1.0

    def test_slightly_bent_index_is_still_high_quality(self):
        ex = PointGestureExercise(_calib())
        pts = _point_pts()
        pts[8] = (0.42, 0.36)
        f = _frame(pts)
        assert ex._pose_quality(f) > 0.9

    def test_pose_not_detected_fist(self):
        ex = PointGestureExercise(_calib())
        f  = _frame(_fist_pts())
        assert ex._pose_detected(f) is False
        assert ex._pose_quality(f) == 0.0


class TestPalmFacingExercise:
    def test_full_palm_facing_quality_is_perfect(self):
        ex = PalmFacingExercise(_calib())
        f = _frame(_orientation_pts("palm"), hand_label="Right")
        assert ex._pose_detected(f) is True
        assert ex._pose_quality(f) == pytest.approx(1.0)

    def test_edge_on_palm_facing_quality_is_zero(self):
        ex = PalmFacingExercise(_calib())
        f = _frame(_orientation_pts("edge"), hand_label="Right")
        assert ex._pose_detected(f) is False
        assert ex._pose_quality(f) == pytest.approx(0.0)


class TestBackFacingExercise:
    def test_full_back_facing_quality_is_perfect(self):
        ex = BackFacingExercise(_calib())
        f = _frame(_orientation_pts("back"), hand_label="Right")
        assert ex._pose_detected(f) is True
        assert ex._pose_quality(f) == pytest.approx(1.0)

    def test_edge_on_back_facing_quality_is_zero(self):
        ex = BackFacingExercise(_calib())
        f = _frame(_orientation_pts("edge"), hand_label="Right")
        assert ex._pose_detected(f) is False
        assert ex._pose_quality(f) == pytest.approx(0.0)



# ── Тесты ZoneMovementExercise ────────────────────────────────────────────────

class TestZoneMovementExercise:
    def test_initial_zone_is_zero(self):
        ex = ZoneMovementExercise(_calib())
        assert ex.current_zone() == 0

    def test_zones_hit_initially_zero(self):
        ex = ZoneMovementExercise(_calib())
        assert ex.zones_hit() == 0

    def test_evaluate_unreliable_no_tracking(self):
        ex = ZoneMovementExercise(_calib())
        ex._prepare_start -= 10
        ex._prepare_confirmed = True
        for _ in range(15):
            ex.feed(TrackingFrame(time.monotonic(), [], False))
        result = ex.evaluate()
        assert result.status == ExerciseStatus.UNRELIABLE
        assert result.score == 0

    def test_max_score_is_15(self):
        ex = ZoneMovementExercise(_calib())
        assert ex.max_score == 15


# ── Тесты is_timeout ──────────────────────────────────────────────────────────

def test_exercise_not_timeout_immediately():
    ex = OpenPalmExercise(_calib())
    assert ex.is_timeout() is False


def test_exercise_completes_after_observation_window():
    ex = OpenPalmExercise(_calib())
    ex._active_start = time.monotonic() - ex.max_duration_sec - 0.1
    assert ex.is_timeout() is True
    assert ex.is_complete() is True
