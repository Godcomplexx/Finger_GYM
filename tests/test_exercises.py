"""Тесты модуля src/exercises/ — логика распознавания поз и оценки."""
import time
import pytest
from src.models import (
    TrackingFrame, Point2D, CalibrationProfile, ExerciseStatus,
)
from src.exercises.exercises import (
    OpenPalmExercise, FistExercise, PinchExercise,
    PointGestureExercise, HoldStillExercise, ZoneMovementExercise,
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
           valid: bool = True) -> TrackingFrame:
    lms = [Point2D(x=x, y=y) for x, y in landmarks]
    return TrackingFrame(timestamp=time.monotonic(),
                         landmarks=lms, is_valid=valid)


def _open_palm_pts() -> list[tuple[float, float]]:
    pts = [(0.5, 0.8)] * 21
    pts[0]  = (0.50, 0.80); pts[17] = (0.65, 0.70)   # WRIST, PINKY_MCP
    pts[5]  = (0.42, 0.60); pts[9]  = (0.50, 0.58)
    pts[13] = (0.58, 0.60); pts[17] = (0.64, 0.64)
    # Кончики далеко
    pts[8]  = (0.42, 0.28); pts[12] = (0.50, 0.26)
    pts[16] = (0.58, 0.28); pts[20] = (0.64, 0.34)
    pts[4]  = (0.22, 0.56)   # THUMB_TIP
    return pts


def _fist_pts() -> list[tuple[float, float]]:
    pts = list(_open_palm_pts())
    # Кончики прижимаем к основаниям
    pts[8]  = (0.44, 0.65)
    pts[12] = (0.51, 0.64)
    pts[16] = (0.59, 0.65)
    pts[20] = (0.65, 0.68)
    return pts


def _pinch_pts() -> list[tuple[float, float]]:
    pts = list(_open_palm_pts())
    pts[4] = (0.43, 0.29)   # THUMB_TIP
    pts[8] = (0.43, 0.29)   # INDEX_TIP (вплотную)
    return pts


def _point_pts() -> list[tuple[float, float]]:
    pts = list(_fist_pts())
    # Указательный разгибаем
    pts[8] = (0.42, 0.28)
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

    def test_max_score_is_15(self):
        ex = FistExercise(_calib())
        assert ex.max_score == 15


# ── Тесты PinchExercise ───────────────────────────────────────────────────────

class TestPinchExercise:
    def test_pose_detected_pinch(self):
        ex = PinchExercise(_calib())
        f  = _frame(_pinch_pts())
        assert ex._pose_detected(f) is True

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

    def test_pose_not_detected_fist(self):
        ex = PointGestureExercise(_calib())
        f  = _frame(_fist_pts())
        assert ex._pose_detected(f) is False


# ── Тесты HoldStillExercise ───────────────────────────────────────────────────

class TestHoldStillExercise:
    def test_evaluate_unreliable_no_tracking(self):
        ex = HoldStillExercise(_calib())
        ex._prepare_start -= 10
        ex._prepare_confirmed = True
        for _ in range(15):
            ex.feed(TrackingFrame(time.monotonic(), [], False))
        result = ex.evaluate()
        assert result.status == ExerciseStatus.UNRELIABLE

    def test_evaluate_stable_hand_no_jitter(self):
        ex = HoldStillExercise(_calib())
        ex.required_hold_sec = 0.0
        ex.min_hold_sec = 0.0
        ex._prepare_start -= 10   # пропускаем фазу подготовки
        ex._prepare_confirmed = True
        pts = _open_palm_pts()
        for _ in range(40):
            ex.feed(_frame(pts))
        result = ex.evaluate()
        assert result.valid_tracking_ratio > 0.9
        # Стабильная рука — штраф за jitter должен быть нулём
        assert result.metrics["jitter"] < 0.01


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
