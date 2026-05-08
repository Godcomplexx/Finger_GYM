from __future__ import annotations
import time
import math
from src.models import TrackingFrame, CalibrationProfile
from src.exercises.base import BaseExercise
from src.scoring.icf import problem_percent_from_score, qualifier_from_problem_percent
from src.processing.metrics import (
    avg_tip_to_palm_distance,
    thumb_index_angle_deg,
    all_finger_curls,
    index_finger_curl,
    palm_facing_quality,
    back_facing_quality,
    compute_palm_center,
    normalized_distance,
    THUMB_TIP,
    LONG_FINGER_TIPS,
    LONG_FINGER_MCPS,
    LONG_FINGER_PIPS,
    LONG_FINGER_DIPS,
    finger_spread,
    fingers_pointing_up,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _quality_from_low_distance(distance: float, best: float, worst: float) -> float:
    if distance <= best:
        return 1.0
    if distance >= worst:
        return 0.0
    return 1.0 - (distance - best) / (worst - best)


def _quality_from_low_value(value: float, best: float, worst: float) -> float:
    if value <= best:
        return 1.0
    if value >= worst:
        return 0.0
    return 1.0 - (value - best) / (worst - best)


def _angle_deg(a, b, c) -> float:
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    mag = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2) * math.sqrt(
        bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2
    )
    if mag < 1e-9:
        return 180.0
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _joint_bend_deg(a, b, c) -> float:
    return max(0.0, 180.0 - _angle_deg(a, b, c))


def _quality_to_target_bend(bend_deg: float, min_deg: float, target_deg: float) -> float:
    if bend_deg <= min_deg:
        return 0.0
    return _clamp01((bend_deg - min_deg) / (target_deg - min_deg))


def _finger_joint_bend_quality(frame: TrackingFrame, finger_index: int) -> float:
    mcp = frame.landmarks[LONG_FINGER_MCPS[finger_index]]
    pip = frame.landmarks[LONG_FINGER_PIPS[finger_index]]
    dip = frame.landmarks[LONG_FINGER_DIPS[finger_index]]
    tip = frame.landmarks[LONG_FINGER_TIPS[finger_index]]
    pip_bend = _joint_bend_deg(mcp, pip, dip)
    dip_bend = _joint_bend_deg(pip, dip, tip)
    pip_quality = _quality_to_target_bend(
        pip_bend, FIST_PIP_MIN_BEND_DEG, FIST_PIP_TARGET_BEND_DEG
    )
    dip_quality = _quality_to_target_bend(
        dip_bend, FIST_DIP_MIN_BEND_DEG, FIST_DIP_TARGET_BEND_DEG
    )
    return pip_quality * 0.65 + dip_quality * 0.35


def _finger_extension_quality(frame: TrackingFrame, finger_index: int) -> float:
    mcp = frame.landmarks[LONG_FINGER_MCPS[finger_index]]
    pip = frame.landmarks[LONG_FINGER_PIPS[finger_index]]
    dip = frame.landmarks[LONG_FINGER_DIPS[finger_index]]
    tip = frame.landmarks[LONG_FINGER_TIPS[finger_index]]
    pip_bend = _joint_bend_deg(mcp, pip, dip)
    dip_bend = _joint_bend_deg(pip, dip, tip)
    pip_quality = _quality_from_low_value(
        pip_bend, POINT_INDEX_STRAIGHT_PIP_BEND_DEG, POINT_INDEX_FIST_PIP_BEND_DEG
    )
    dip_quality = _quality_from_low_value(
        dip_bend, POINT_INDEX_STRAIGHT_DIP_BEND_DEG, POINT_INDEX_FIST_DIP_BEND_DEG
    )
    return pip_quality * 0.65 + dip_quality * 0.35


PINCH_CLOSED_ANGLE_DEG = 12.0
PINCH_OPEN_ANGLE_DEG = 90.0
OPEN_PALM_IDEAL_CURL = 0.20
OPEN_PALM_MAX_CURL = 0.70
POINT_INDEX_STRAIGHT_PIP_BEND_DEG = 10.0
POINT_INDEX_STRAIGHT_DIP_BEND_DEG = 8.0
POINT_INDEX_FIST_PIP_BEND_DEG = 90.0
POINT_INDEX_FIST_DIP_BEND_DEG = 50.0
FIST_PIP_MIN_BEND_DEG = 25.0
FIST_DIP_MIN_BEND_DEG = 15.0
FIST_PIP_TARGET_BEND_DEG = 80.0
FIST_DIP_TARGET_BEND_DEG = 45.0
FIST_FINGER_DONE_QUALITY = 0.85
FIST_THUMB_NEAR_FINGER = 0.80
FIST_THUMB_FAR_FINGER = 1.60


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
        return extended >= 3 and tip_dist > 0.55 and spread > 0.25 and fingers_pointing_up(frame)

    def _pose_quality(self, frame: TrackingFrame) -> float:
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        extension_quality = sum(
            _quality_from_low_value(c, OPEN_PALM_IDEAL_CURL, OPEN_PALM_MAX_CURL)
            for c in curls
        ) / 4
        tip_quality = _clamp01(avg_tip_to_palm_distance(frame, pw) / 0.55)
        spread_quality = _clamp01(finger_spread(frame, pw) / 0.25)
        return extension_quality * 0.55 + tip_quality * 0.30 + spread_quality * 0.15

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
        "Выполняйте задание боковой стороной кисти к камере.",
        "Плотно согните 4 длинных пальца к ладони.",
        "Кончики пальцев должны быть близко к центру ладони, не наполовину согнуты.",
        "Удерживайте кулак 5 секунд.",
    ]
    max_score = 15
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _finger_joint_quality(self, frame: TrackingFrame, finger_index: int) -> float:
        return _finger_joint_bend_quality(frame, finger_index)

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        finger_qualities = [self._finger_joint_quality(frame, i) for i in range(4)]
        return all(q >= FIST_FINGER_DONE_QUALITY for q in finger_qualities)

    def _pose_quality(self, frame: TrackingFrame) -> float:
        pw = self.calibration.palm_width
        finger_quality = sum(self._finger_joint_quality(frame, i) for i in range(4)) / 4
        thumb_dist = min(
            normalized_distance(frame.landmarks[THUMB_TIP], frame.landmarks[idx], pw)
            for idx in LONG_FINGER_TIPS
        )
        thumb_quality = _quality_from_low_distance(
            thumb_dist, FIST_THUMB_NEAR_FINGER, FIST_THUMB_FAR_FINGER
        )
        return finger_quality * 0.85 + thumb_quality * 0.15

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        straight_names = ["указат.", "средн.", "безым.", "мизинец"]
        finger_qualities = [self._finger_joint_quality(frame, i) for i in range(4)]
        straight = [straight_names[i] for i, q in enumerate(finger_qualities) if q < FIST_FINGER_DONE_QUALITY]
        if straight:
            return f"Согните сильнее: {', '.join(straight)}"
        return ""


# ── 3. Щипковый захват ────────────────────────────────────────────────────────

class PinchExercise(BaseExercise):
    exercise_id = "pinch"
    instruction = "Щипковый захват"
    details = [
        "Выполняйте задание боковой стороной кисти к камере.",
        "Сведите большой и указательный пальцы до касания или почти до касания.",
        "Остальные пальцы не должны полностью закрывать ладонь.",
        "Удерживайте щипок 5 секунд.",
    ]
    max_score = 15
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        angle = thumb_index_angle_deg(frame)
        if angle > 35.0:
            return False
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        other_all_bent = sum(1 for c in curls[1:] if c > 0.65) == 3
        return not other_all_bent

    def _pose_quality(self, frame: TrackingFrame) -> float:
        angle = thumb_index_angle_deg(frame)
        angle_quality = 1.0 - (
            max(0.0, angle - PINCH_CLOSED_ANGLE_DEG)
            / (PINCH_OPEN_ANGLE_DEG - PINCH_CLOSED_ANGLE_DEG)
        )
        pinch_quality = _clamp01(angle_quality)
        pw = self.calibration.palm_width
        curls = all_finger_curls(frame, pw)
        other_all_bent = sum(1 for c in curls[1:] if c > 0.65) == 3
        posture_factor = 0.7 if other_all_bent else 1.0
        return pinch_quality * posture_factor

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        angle = thumb_index_angle_deg(frame)
        if angle > 35.0:
            return "Сведите большой и указательный пальцы ближе"
        return ""


# ── 4. Указательный жест ─────────────────────────────────────────────────────

class PointGestureExercise(BaseExercise):
    exercise_id = "point_gesture"
    instruction = "Указательный жест"
    details = [
        "Выполняйте задание боковой стороной кисти к камере.",
        "Выпрямите указательный палец.",
        "Согните минимум два из трех остальных длинных пальцев.",
        "Удерживайте жест 5 секунд.",
    ]
    max_score = 10
    required_hold_sec = 5.0
    min_hold_sec = 2.5
    max_duration_sec = 5.0

    def _pose_detected(self, frame: TrackingFrame) -> bool:
        index_quality = _finger_extension_quality(frame, 0)
        other_qualities = [_finger_joint_bend_quality(frame, i) for i in range(1, 4)]
        others_bent = sum(1 for q in other_qualities if q >= 0.65)
        return index_quality >= 0.75 and others_bent >= 2

    def _pose_quality(self, frame: TrackingFrame) -> float:
        index_quality = _finger_extension_quality(frame, 0)
        other_quality = sum(_finger_joint_bend_quality(frame, i) for i in range(1, 4)) / 3
        return index_quality * (0.75 + other_quality * 0.25)

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        index_quality = _finger_extension_quality(frame, 0)
        if index_quality < 0.75:
            return "Выпрямите указательный палец"
        other_qualities = [_finger_joint_bend_quality(frame, i) for i in range(1, 4)]
        others_bent = sum(1 for q in other_qualities if q >= 0.65)
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
        return palm_facing_quality(frame) >= 0.65 and fingers_pointing_up(frame)

    def _pose_quality(self, frame: TrackingFrame) -> float:
        if not fingers_pointing_up(frame):
            return 0.0
        return palm_facing_quality(frame)

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        if palm_facing_quality(frame) < 0.65:
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
        return back_facing_quality(frame) >= 0.65 and fingers_pointing_up(frame)

    def _pose_quality(self, frame: TrackingFrame) -> float:
        if not fingers_pointing_up(frame):
            return 0.0
        return back_facing_quality(frame)

    def pose_fail_reason(self, frame: TrackingFrame) -> str:
        if not frame.is_valid:
            return ""
        if back_facing_quality(frame) < 0.65:
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
        if not self._active_armed:
            return
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
        return self._zone_index >= len(ZONES) or self.elapsed() >= self.max_duration_sec

    def is_timeout(self) -> bool:
        return self.elapsed() >= self.max_duration_sec

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
            problem_percent = None
            icf_qualifier = None
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
                    "observationTimeSec": round(self.elapsed(), 2),
                    "problemPercent": problem_percent,
                    "icfQualifier": icf_qualifier,
                    "requiredZoneHoldSec": ZONE_HOLD_SEC,
                    "maxDurationSec": self.max_duration_sec,
                },
                notes=["Низкое качество трекинга — результат технически ненадёжен"],
            )

        if hit >= total:
            status = ExerciseStatus.DONE
            score = self.max_score
        else:
            status = ExerciseStatus.PARTIAL
            score = int(self.max_score * hit / total)
        if status == ExerciseStatus.UNRELIABLE:
            problem_percent = None
            icf_qualifier = None
        else:
            problem_percent = problem_percent_from_score(score, self.max_score)
            icf_qualifier = qualifier_from_problem_percent(problem_percent)

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
                "observationTimeSec": round(self.elapsed(), 2),
                "problemPercent": problem_percent,
                "icfQualifier": icf_qualifier,
                "requiredZoneHoldSec": ZONE_HOLD_SEC,
                "maxDurationSec": self.max_duration_sec,
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
            if jitter <= 0.02:
                jitter_factor = 1.0
            elif jitter <= 0.05:
                jitter_factor = 0.7
            else:
                jitter_factor = 0.4
            score = round(self.max_score * jitter_factor)
            status = ExerciseStatus.DONE if score == self.max_score else ExerciseStatus.PARTIAL

        if status == ExerciseStatus.UNRELIABLE:
            problem_percent = None
            icf_qualifier = None
        else:
            problem_percent = problem_percent_from_score(score, self.max_score)
            icf_qualifier = qualifier_from_problem_percent(problem_percent)

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
                "observationTimeSec": round(self.elapsed(), 2),
                "problemPercent": problem_percent,
                "icfQualifier": icf_qualifier,
                "requiredHoldSec": self.required_hold_sec,
                "minHoldSec": self.min_hold_sec,
                "maxDurationSec": self.max_duration_sec,
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
