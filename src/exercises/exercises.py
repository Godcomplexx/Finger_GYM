from __future__ import annotations
import time
import math
from src.models import TrackingFrame, CalibrationProfile
from src.exercises.base import BaseExercise
from src.processing.metrics import (
    avg_tip_to_palm_distance,
    thumb_index_distance,
    all_finger_curls,
    index_finger_curl,
    palm_facing_camera,
    compute_palm_center,
    finger_spread,
    fingers_pointing_up,
)


# ── 1. Открытая ладонь ────────────────────────────────────────────────────────

class OpenPalmExercise(BaseExercise):
    exercise_id = "open_palm"
    instruction = "Покажите открытую ладонь — разогните все пальцы и держите"
    max_score = 10
    required_hold_sec = 3.0
    min_hold_sec = 1.5
    max_duration_sec = 40.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        tip_dist = avg_tip_to_palm_distance(frame, pw)
        curls = all_finger_curls(frame, pw)
        extended = sum(1 for c in curls if c < 0.40)
        spread = finger_spread(frame, pw)
        return extended >= 3 and tip_dist > 0.55 and spread > 0.25


# ── 2. Кулак ──────────────────────────────────────────────────────────────────

class FistExercise(BaseExercise):
    exercise_id = "fist"
    instruction = "Сожмите кулак и держите"
    max_score = 15
    required_hold_sec = 3.0
    min_hold_sec = 1.5
    max_duration_sec = 40.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        bent = sum(1 for c in curls if c > 0.50)
        return bent >= 3


# ── 3. Щипковый захват ────────────────────────────────────────────────────────

class PinchExercise(BaseExercise):
    exercise_id = "pinch"
    instruction = "Сведите большой и указательный пальцы вместе, остальные разогните"
    max_score = 15
    required_hold_sec = 3.0
    min_hold_sec = 1.5
    max_duration_sec = 40.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        d  = thumb_index_distance(frame, pw)
        if d >= 0.30:
            return False
        curls = all_finger_curls(frame, pw)
        other_all_bent = sum(1 for c in curls[1:] if c > 0.65) == 3
        return not other_all_bent


# ── 4. Указательный жест ─────────────────────────────────────────────────────

class PointGestureExercise(BaseExercise):
    exercise_id = "point_gesture"
    instruction = "Вытяните указательный палец, остальные согните"
    max_score = 10
    required_hold_sec = 3.0
    min_hold_sec = 1.5
    max_duration_sec = 40.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        index_curl = curls[0]
        others_bent = sum(1 for c in curls[1:] if c > 0.55)
        return index_curl < 0.40 and others_bent >= 2


# ── 5. Ладонь к камере ────────────────────────────────────────────────────────

class PalmFacingExercise(BaseExercise):
    exercise_id = "palm_facing"
    instruction = "Держите руку вертикально — пальцы вверх, ладонью к камере"
    max_score = 10
    required_hold_sec = 3.0
    min_hold_sec = 1.5
    max_duration_sec = 40.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return palm_facing_camera(frame) and fingers_pointing_up(frame)


# ── 6. Тыльная сторона к камере ───────────────────────────────────────────────

class BackFacingExercise(BaseExercise):
    exercise_id = "back_facing"
    instruction = "Держите руку вертикально — пальцы вверх, тыльной стороной к камере"
    max_score = 10
    required_hold_sec = 3.0
    min_hold_sec = 1.5
    max_duration_sec = 40.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return not palm_facing_camera(frame) and fingers_pointing_up(frame)


# ── 7. Перемещение по зонам экрана ────────────────────────────────────────────

ZONES = [
    (0.25, 0.35),  # левая зона
    (0.50, 0.35),  # центральная зона
    (0.75, 0.35),  # правая зона
]
ZONE_RADIUS = 0.18   # увеличен для пожилых
ZONE_HOLD_SEC = 1.5  # чуть дольше для надёжности


class ZoneMovementExercise(BaseExercise):
    exercise_id = "zone_movement"
    instruction = "Медленно перемещайте руку через кружки слева направо"
    max_score = 15
    required_hold_sec = 1.0
    min_hold_sec = 0.5
    max_duration_sec = 60.0

    def __init__(self, calibration: CalibrationProfile):
        super().__init__(calibration)
        self._zone_index = 0
        self._zone_hold_start: float | None = None
        self._zones_hit: list[bool] = [False] * len(ZONES)

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return frame.is_valid

    def feed(self, frame: TrackingFrame):
        super().feed(frame)
        if not frame.is_valid or self._zone_index >= len(ZONES):
            return
        center = compute_palm_center(frame)
        target = ZONES[self._zone_index]
        dist = math.hypot(center.x - target[0], center.y - target[1])
        now = time.monotonic()
        if dist <= ZONE_RADIUS:
            if self._zone_hold_start is None:
                self._zone_hold_start = now
            elif now - self._zone_hold_start >= ZONE_HOLD_SEC:
                self._zones_hit[self._zone_index] = True
                self._zone_index += 1
                self._zone_hold_start = None
        else:
            self._zone_hold_start = None

    def is_complete(self) -> bool:
        return self._zone_index >= len(ZONES)

    def current_zone(self) -> int:
        return self._zone_index

    def zone_hold_progress(self) -> float:
        """0..1 прогресс удержания в текущей зоне."""
        if self._zone_hold_start is None:
            return 0.0
        return min(1.0, (time.monotonic() - self._zone_hold_start) / ZONE_HOLD_SEC)

    def zones_hit(self) -> int:
        return sum(self._zones_hit)

    def evaluate(self):
        from src.processing.metrics import valid_tracking_ratio, hand_jitter
        from src.models import ExerciseStatus, ExerciseResult
        vtr = valid_tracking_ratio(self._frames)
        centers = [compute_palm_center(f) for f in self._frames if f.is_valid]
        jitter = hand_jitter(centers)

        hit = self.zones_hit()
        total = len(ZONES)

        if vtr < 0.65:
            return ExerciseResult(
                exercise_id=self.exercise_id,
                status=ExerciseStatus.UNRELIABLE,
                score=0, max_score=self.max_score,
                hold_time_sec=0.0,
                valid_tracking_ratio=vtr,
                metrics={"zones_hit": hit, "zones_total": total, "jitter": round(jitter, 4)},
                notes=["Низкое качество трекинга — результат технически ненадёжен"],
            )

        if hit >= total:
            status = ExerciseStatus.DONE
            score = self.max_score
        else:
            status = ExerciseStatus.PARTIAL
            score = int(self.max_score * hit / total)

        notes = []
        if hit < total:
            notes.append(f"Достигнуто {hit} из {total} зон")
        if jitter > 0.02:
            notes.append("Обнаружено дрожание руки")

        return ExerciseResult(
            exercise_id=self.exercise_id,
            status=status,
            score=score, max_score=self.max_score,
            hold_time_sec=float(hit * ZONE_HOLD_SEC),
            valid_tracking_ratio=vtr,
            metrics={"zones_hit": hit, "zones_total": total, "jitter": round(jitter, 4)},
            notes=notes,
        )


# ── Фабрика ───────────────────────────────────────────────────────────────────

EXERCISE_ORDER = [
    OpenPalmExercise,
    FistExercise,
    PinchExercise,
    PointGestureExercise,
    PalmFacingExercise,
    BackFacingExercise,
    ZoneMovementExercise,
]


def create_exercises(calibration: CalibrationProfile) -> list[BaseExercise]:
    return [cls(calibration) for cls in EXERCISE_ORDER]
