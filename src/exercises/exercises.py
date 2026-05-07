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
    instruction = "Открытая ладонь"
    details = [
        "Разогните 4 длинных пальца и отведите кончики от центра ладони.",
        "Разведите пальцы так, чтобы они были видны отдельно.",
        "Удерживайте позу 5 секунд.",
    ]
    max_score = 10
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        tip_dist = avg_tip_to_palm_distance(frame, pw)
        curls = all_finger_curls(frame, pw)
        extended = sum(1 for c in curls if c < 0.40)
        spread = finger_spread(frame, pw)
        return extended >= 3 and tip_dist > 0.55 and spread > 0.25

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        extended = sum(1 for c in curls if c < 0.40)
        if extended < 3:
            bent_names = ["указат.", "средн.", "безым.", "мизинец"]
            bent = [bent_names[i] for i, c in enumerate(curls) if c >= 0.40]
            return f"Разогните: {', '.join(bent)}"
        tip_dist = avg_tip_to_palm_distance(frame, pw)
        if tip_dist <= 0.55:
            return "Выпрямите пальцы дальше от ладони"
        spread = finger_spread(frame, pw)
        if spread <= 0.25:
            return "Разведите пальцы шире"
        return ""


# ── 2. Кулак ──────────────────────────────────────────────────────────────────

class FistExercise(BaseExercise):
    exercise_id = "fist"
    instruction = "Кулак"
    details = [
        "Плотно согните 4 длинных пальца к ладони.",
        "Кончики пальцев должны быть близко к центру ладони, не наполовину согнуты.",
        "Удерживайте кулак 5 секунд.",
    ]
    max_score = 15
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        return len(curls) == 4 and all(c > 0.72 for c in curls)

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        straight_names = ["указат.", "средн.", "безым.", "мизинец"]
        straight = [straight_names[i] for i, c in enumerate(curls) if c <= 0.72]
        if straight:
            return f"Согните сильнее: {', '.join(straight)}"
        return ""


# ── 3. Щипковый захват ────────────────────────────────────────────────────────

class PinchExercise(BaseExercise):
    exercise_id = "pinch"
    instruction = "Щипковый захват"
    details = [
        "Сведите большой и указательный пальцы до касания или почти до касания.",
        "Остальные пальцы не должны полностью закрывать ладонь.",
        "Удерживайте щипок 5 секунд.",
    ]
    max_score = 15
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        d  = thumb_index_distance(frame, pw)
        if d >= 0.25:
            return False
        curls = all_finger_curls(frame, pw)
        other_all_bent = sum(1 for c in curls[1:] if c > 0.65) == 3
        return not other_all_bent

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        pw = self.calibration.palm_width
        d = thumb_index_distance(frame, pw)
        if d >= 0.25:
            return "Сведите большой и указательный пальцы ближе"
        return ""


# ── 4. Указательный жест ─────────────────────────────────────────────────────

class PointGestureExercise(BaseExercise):
    exercise_id = "point_gesture"
    instruction = "Указательный жест"
    details = [
        "Выпрямите указательный палец.",
        "Согните минимум два из трех остальных длинных пальцев.",
        "Удерживайте жест 5 секунд.",
    ]
    max_score = 10
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        index_curl = curls[0]
        others_bent = sum(1 for c in curls[1:] if c > 0.55)
        return index_curl < 0.40 and others_bent >= 2

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        if curls[0] >= 0.40:
            return "Вытяните указательный палец"
        others_bent = sum(1 for c in curls[1:] if c > 0.55)
        if others_bent < 2:
            return "Согните остальные пальцы"
        return ""


# ── 5. Ладонь к камере ────────────────────────────────────────────────────────

class PalmFacingExercise(BaseExercise):
    exercise_id = "palm_facing"
    instruction = "Ладонь к камере"
    details = [
        "Поставьте кисть вертикально, пальцы направьте вверх.",
        "Поверните внутреннюю сторону ладони к камере.",
        "Удерживайте ориентацию 5 секунд.",
    ]
    max_score = 10
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return palm_facing_camera(frame) and fingers_pointing_up(frame)

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        if not palm_facing_camera(frame):
            return "Повернитесь ладонью к камере"
        if not fingers_pointing_up(frame):
            return "Поднимите пальцы вверх"
        return ""


# ── 6. Тыльная сторона к камере ───────────────────────────────────────────────

class BackFacingExercise(BaseExercise):
    exercise_id = "back_facing"
    instruction = "Тыльная сторона к камере"
    details = [
        "Поставьте кисть вертикально, пальцы направьте вверх.",
        "Поверните тыльную сторону кисти к камере.",
        "Удерживайте ориентацию 5 секунд.",
    ]
    max_score = 10
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return not palm_facing_camera(frame) and fingers_pointing_up(frame)

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        if palm_facing_camera(frame):
            return "Повернитесь тыльной стороной к камере"
        if not fingers_pointing_up(frame):
            return "Поднимите пальцы вверх"
        return ""


# ── 7. Перемещение по зонам экрана ────────────────────────────────────────────

ZONES = [
    (0.16, 0.48),
    (0.33, 0.36),
    (0.50, 0.52),
    (0.67, 0.36),
    (0.84, 0.48),
]
ZONE_RADIUS = 0.085
ZONE_HOLD_SEC = 1.0


class ZoneMovementExercise(BaseExercise):
    exercise_id = "zone_movement"
    instruction = "Перемещение по зонам"
    details = [
        "Перемещайте центр ладони по кругам в указанном порядке.",
        "Задержитесь в каждом круге примерно на 1 секунду.",
        "Баллы зависят от количества достигнутых зон.",
    ]
    max_score = 15
    required_hold_sec = 1.0
    min_hold_sec = 0.5
    max_duration_sec = 20.0

    def __init__(self, calibration: CalibrationProfile):
        super().__init__(calibration)
        self._zone_index = 0
        self._zone_hold_start: float | None = None
        self._zones_hit: list[bool] = [False] * len(ZONES)
        self._zone_just_hit: bool = False

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
                self._zone_just_hit = True
        else:
            self._zone_hold_start = None

    def is_complete(self) -> bool:
        return self._zone_index >= len(ZONES)

    def is_timeout(self) -> bool:
        return False

    def current_zone(self) -> int:
        return self._zone_index

    def zone_hold_progress(self) -> float:
        """0..1 прогресс удержания в текущей зоне."""
        if self._zone_hold_start is None:
            return 0.0
        return min(1.0, (time.monotonic() - self._zone_hold_start) / ZONE_HOLD_SEC)

    def zones_hit(self) -> int:
        return sum(self._zones_hit)

    def consume_zone_hit(self) -> bool:
        """True если зона только что засчитана (сбрасывается после вызова)."""
        val = self._zone_just_hit
        self._zone_just_hit = False
        return val

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
                metrics={
                    "zones_hit": hit,
                    "zones_total": total,
                    "jitter": round(jitter, 4),
                    "frames": len(self._frames),
                    "requiredZoneHoldSec": ZONE_HOLD_SEC,
                },
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
            metrics={
                "zones_hit": hit,
                "zones_total": total,
                "jitter": round(jitter, 4),
                "frames": len(self._frames),
                "requiredZoneHoldSec": ZONE_HOLD_SEC,
            },
            notes=notes,
        )


class HoldStillExercise(BaseExercise):
    exercise_id = "hold_still"
    instruction = "Удержание руки"
    details = [
        "Держите кисть в удобном положении перед камерой.",
        "Старайтесь не смещать ладонь и не выходить из кадра.",
        "Оценивается устойчивость в течение 5 секунд.",
    ]
    max_score = 5
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        return frame.is_valid

    def evaluate(self):
        from src.processing.metrics import valid_tracking_ratio, hand_jitter
        from src.models import ExerciseStatus, ExerciseResult

        vtr = valid_tracking_ratio(self._frames)
        centers = [compute_palm_center(f) for f in self._frames if f.is_valid]
        jitter = hand_jitter(centers)

        if vtr < 0.65:
            status = ExerciseStatus.UNRELIABLE
            score = 0
        else:
            if self._hold_time >= self.required_hold_sec:
                hold_score = self.max_score
                status = ExerciseStatus.DONE
            elif self._hold_time >= self.min_hold_sec:
                ratio = (self._hold_time - self.min_hold_sec) / max(
                    0.001, self.required_hold_sec - self.min_hold_sec)
                hold_score = round(self.max_score * (0.5 + 0.5 * min(1.0, ratio)))
                status = ExerciseStatus.PARTIAL
            else:
                hold_score = 0
                status = ExerciseStatus.PARTIAL

            if jitter <= 0.02:
                jitter_factor = 1.0
            elif jitter <= 0.05:
                jitter_factor = 0.7
            else:
                jitter_factor = 0.4
            score = round(hold_score * jitter_factor)
            if score < self.max_score:
                status = ExerciseStatus.PARTIAL

        notes = []
        if vtr < 0.65:
            notes.append("Низкое качество трекинга - результат технически ненадежен")
        if jitter > 0.02:
            notes.append("Обнаружено дрожание руки")

        return ExerciseResult(
            exercise_id=self.exercise_id,
            status=status,
            score=score,
            max_score=self.max_score,
            hold_time_sec=self._hold_time,
            valid_tracking_ratio=vtr,
            metrics={
                "jitter": round(jitter, 4),
                "frames": len(self._frames),
                "requiredHoldSec": self.required_hold_sec,
                "minHoldSec": self.min_hold_sec,
            },
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

