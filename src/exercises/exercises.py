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

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        tip_dist = avg_tip_to_palm_distance(frame, pw)
        curls = all_finger_curls(frame, pw)
        extended = sum(1 for c in curls if c < 0.40)
        spread = finger_spread(frame, pw)
        # 3 из 4 пальцев разогнуты, кончики далеко от ладони, пальцы разведены
        return extended >= 3 and tip_dist > 0.55 and spread > 0.25


# ── 2. Кулак ──────────────────────────────────────────────────────────────────

class FistExercise(BaseExercise):
    exercise_id = "fist"
    instruction = "Сожмите кулак и держите"
    max_score = 15
    required_hold_sec = 3.0
    min_hold_sec = 1.5

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        # Все 4 длинных пальца хорошо согнуты
        bent = sum(1 for c in curls if c > 0.50)
        return bent >= 3


# ── 3. Щипковый захват ────────────────────────────────────────────────────────

class PinchExercise(BaseExercise):
    exercise_id = "pinch"
    instruction = "Сведите большой и указательный пальцы вместе, остальные разогните"
    max_score = 15
    required_hold_sec = 3.0
    min_hold_sec = 1.5

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        d  = thumb_index_distance(frame, pw)
        if d >= 0.30:
            return False
        # Защита от кулака: хотя бы 1 из средний/безымянный/мизинец свободен (curl < 0.65)
        curls = all_finger_curls(frame, pw)   # [index, middle, ring, pinky]
        other_all_bent = sum(1 for c in curls[1:] if c > 0.65) == 3
        return not other_all_bent


# ── 4. Указательный жест ─────────────────────────────────────────────────────

class PointGestureExercise(BaseExercise):
    exercise_id = "point_gesture"
    instruction = "Вытяните указательный палец, остальные согните"
    max_score = 10
    required_hold_sec = 3.0
    min_hold_sec = 1.5

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)   # [index, middle, ring, pinky]
        index_curl = curls[0]
        # Средний, безымянный, мизинец должны быть хорошо согнуты (>= 2 из 3)
        others_bent = sum(1 for c in curls[1:] if c > 0.55)
        return index_curl < 0.40 and others_bent >= 2


# ── 5. Ладонь к камере ────────────────────────────────────────────────────────

class PalmFacingExercise(BaseExercise):
    exercise_id = "palm_facing"
    instruction = "Держите руку вертикально — пальцы вверх, ладонью к камере"
    max_score = 10
    required_hold_sec = 3.0
    min_hold_sec = 1.5

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return palm_facing_camera(frame) and fingers_pointing_up(frame)


# ── 6. Тыльная сторона к камере ───────────────────────────────────────────────

class BackFacingExercise(BaseExercise):
    exercise_id = "back_facing"
    instruction = "Держите руку вертикально — пальцы вверх, тыльной стороной к камере"
    max_score = 10
    required_hold_sec = 3.0
    min_hold_sec = 1.5

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return not palm_facing_camera(frame) and fingers_pointing_up(frame)


# ── 7. Перемещение по зонам экрана ────────────────────────────────────────────

ZONES = [
    (0.2, 0.2),   # верхний левый
    (0.8, 0.2),   # верхний правый
    (0.5, 0.5),   # центр
    (0.2, 0.8),   # нижний левый
    (0.8, 0.8),   # нижний правый
]
ZONE_RADIUS = 0.15   # нормированный радиус зоны
ZONE_HOLD_SEC = 1.0  # удержание в каждой зоне


class ZoneMovementExercise(BaseExercise):
    exercise_id = "zone_movement"
    instruction = "Перемещайте руку по кружкам на экране"
    max_score = 15
    required_hold_sec = 1.0   # для базового класса (не используется напрямую)
    min_hold_sec = 0.5
    max_duration_sec = 30.0

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

    def zones_hit(self) -> int:
        return sum(self._zones_hit)

    def evaluate(self):
        from src.processing.metrics import valid_tracking_ratio, hand_jitter
        vtr = valid_tracking_ratio(self._frames)
        centers = [compute_palm_center(f) for f in self._frames if f.is_valid]
        jitter = hand_jitter(centers)

        hit = self.zones_hit()
        total = len(ZONES)

        if vtr < 0.65:
            from src.models import ExerciseStatus, ExerciseResult
            return ExerciseResult(
                exercise_id=self.exercise_id,
                status=ExerciseStatus.UNRELIABLE,
                score=0, max_score=self.max_score,
                hold_time_sec=0.0,
                valid_tracking_ratio=vtr,
                metrics={"zones_hit": hit, "zones_total": total, "jitter": round(jitter, 4)},
                notes=["Низкое качество трекинга — результат технически ненадёжен"],
            )

        from src.models import ExerciseStatus, ExerciseResult
        if hit >= total:
            status = ExerciseStatus.DONE
            score = self.max_score
        elif hit >= total - 1:
            status = ExerciseStatus.PARTIAL
            score = int(self.max_score * hit / total)
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


# ── 8. Удержание руки неподвижно ──────────────────────────────────────────────

class HoldStillExercise(BaseExercise):
    exercise_id = "hold_still"
    instruction = "Держите руку неподвижно 4 секунды"
    max_score = 5
    required_hold_sec = 4.0
    min_hold_sec = 2.0
    max_duration_sec = 12.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        # Поза обнаружена если рука видна и стабильна (jitter считается в evaluate)
        return frame.is_valid

    def evaluate(self):
        from src.processing.metrics import valid_tracking_ratio, hand_jitter, compute_palm_center
        from src.models import ExerciseStatus, ExerciseResult
        vtr = valid_tracking_ratio(self._frames)
        centers = [compute_palm_center(f) for f in self._frames if f.is_valid]
        jitter = hand_jitter(centers)

        notes = []
        if vtr < 0.65:
            return ExerciseResult(
                exercise_id=self.exercise_id,
                status=ExerciseStatus.UNRELIABLE,
                score=0, max_score=self.max_score,
                hold_time_sec=self._hold_time,
                valid_tracking_ratio=vtr,
                metrics={"jitter": round(jitter, 4)},
                notes=["Низкое качество трекинга"],
            )

        # Штраф за дрожание: реалистичный порог — лёгкий тремор до 0.04 норма,
        # значимое дрожание > 0.07. Масштаб: 0..0.10 → штраф 0..max_score
        JITTER_OK    = 0.035   # нет штрафа
        JITTER_MAX   = 0.10    # максимальный штраф (все баллы)
        jitter_ratio = max(0.0, min(1.0,
            (jitter - JITTER_OK) / (JITTER_MAX - JITTER_OK)))
        jitter_penalty = round(self.max_score * jitter_ratio)

        if self._hold_time >= self.required_hold_sec:
            score = max(0, self.max_score - jitter_penalty)
            status = ExerciseStatus.DONE if score > 0 else ExerciseStatus.PARTIAL
        elif self._hold_time >= self.min_hold_sec:
            ratio = (self._hold_time - self.min_hold_sec) / max(
                0.001, self.required_hold_sec - self.min_hold_sec)
            base = round(self.max_score * (0.5 + 0.5 * min(1.0, ratio)))
            score = max(0, base - jitter_penalty)
            status = ExerciseStatus.PARTIAL
        else:
            score = 0
            status = ExerciseStatus.PARTIAL

        if jitter > 0.05:
            notes.append(f"Обнаружено дрожание руки (jitter={jitter:.3f})")

        return ExerciseResult(
            exercise_id=self.exercise_id,
            status=status,
            score=score, max_score=self.max_score,
            hold_time_sec=self._hold_time,
            valid_tracking_ratio=vtr,
            metrics={"jitter": round(jitter, 4)},
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
    HoldStillExercise,
]


def create_exercises(calibration: CalibrationProfile) -> list[BaseExercise]:
    return [cls(calibration) for cls in EXERCISE_ORDER]
