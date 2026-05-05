from __future__ import annotations
import time
from abc import ABC, abstractmethod
from src.models import TrackingFrame, CalibrationProfile, ExerciseResult, ExerciseStatus
from src.processing.metrics import (
    valid_tracking_ratio, compute_palm_center, hand_jitter, hand_in_position,
)

MIN_TRACKING_RATIO = 0.65
PREPARE_SEC        = 3.0   # минимальное время показа инструкции

# Допуск на кратковременный пропуск позы (дрожание, моргание трекера)
# Если поза пропала меньше чем на HOLD_GRACE_SEC — удержание не сбрасывается
HOLD_GRACE_SEC = 0.35


class BaseExercise(ABC):
    """Базовый класс задания.

    Жизненный цикл:
      PREPARE  — показываем инструкцию; кадры не считаются; ждём PREPARE_SEC и подтверждения
      ACTIVE   — принимаем кадры, считаем удержание
      DONE     — удержание достигнуто, задание завершено
    """

    exercise_id: str = ""
    instruction: str = ""
    max_score: int = 10
    required_hold_sec: float = 2.5   # идеальное удержание
    min_hold_sec:      float = 1.5   # минимально допустимое
    max_duration_sec:  float = 20.0  # таймаут ACTIVE-фазы

    def __init__(self, calibration: CalibrationProfile):
        self.calibration  = calibration
        self._frames: list[TrackingFrame] = []
        self._hold_start: float | None = None
        self._hold_time:  float = 0.0
        self._lost_start: float | None = None   # когда поза пропала (grace period)
        self._prepare_start: float = time.monotonic()
        self._prepare_confirmed: bool = False  # оператор нажал Space/Enter
        self._active_start:  float | None = None
        self._done: bool = False
        self._position_hint: str = ""  # подсказка по позиционированию

    # ── Фазы ──────────────────────────────────────────────────────────────────

    def is_preparing(self) -> bool:
        """True пока идёт фаза показа инструкции."""
        elapsed = time.monotonic() - self._prepare_start
        if elapsed < PREPARE_SEC:
            return True
        # После минимального времени — ждём подтверждения (Space/Enter)
        return not self._prepare_confirmed

    def confirm_start(self):
        """Оператор/пациент нажал Space или Enter — начать задание."""
        if (time.monotonic() - self._prepare_start) >= PREPARE_SEC:
            self._prepare_confirmed = True

    def prepare_elapsed(self) -> float:
        return min(PREPARE_SEC, time.monotonic() - self._prepare_start)

    def _ensure_active_started(self):
        if self._active_start is None:
            self._active_start = time.monotonic()

    # ── Приём кадров ──────────────────────────────────────────────────────────

    def feed(self, frame: TrackingFrame):
        if self._done or self.is_preparing():
            return
        self._ensure_active_started()
        self._frames.append(frame)

        # Проверка позиционирования
        in_pos, hint = hand_in_position(frame)
        self._position_hint = hint

        # Считаем удержание только если рука в корректной позиции
        self._update_hold(frame, in_pos)

    def _update_hold(self, frame: TrackingFrame, in_position: bool = True):
        now = time.monotonic()
        pose_ok = frame.is_valid and in_position and self._pose_detected(frame)

        if pose_ok:
            self._lost_start = None
            if self._hold_start is None:
                self._hold_start = now
            self._hold_time = now - self._hold_start
        else:
            # Поза пропала — запускаем grace period
            if self._lost_start is None:
                self._lost_start = now

            grace_expired = (now - self._lost_start) > HOLD_GRACE_SEC
            if grace_expired:
                # Сбрасываем удержание только после истечения допуска
                self._hold_start = None
                self._hold_time  = 0.0
                # grace period не сбрасываем — он нужен до следующей успешной позы

    @abstractmethod
    def _pose_detected(self, frame: TrackingFrame) -> bool: ...

    # ── Статус ────────────────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        return self._hold_time >= self.min_hold_sec

    def is_timeout(self) -> bool:
        if self._active_start is None:
            return False
        return (time.monotonic() - self._active_start) >= self.max_duration_sec

    def elapsed(self) -> float:
        if self._active_start is None:
            return 0.0
        return time.monotonic() - self._active_start

    def current_hold(self) -> float:
        return self._hold_time

    def position_hint(self) -> str:
        """Подсказка по позиционированию для текущего кадра (пустая строка = всё хорошо)."""
        return self._position_hint

    # ── Оценка ────────────────────────────────────────────────────────────────

    def evaluate(self) -> ExerciseResult:
        vtr     = valid_tracking_ratio(self._frames)
        centers = [compute_palm_center(f) for f in self._frames if f.is_valid]
        jitter  = hand_jitter(centers)

        if vtr < MIN_TRACKING_RATIO:
            status = ExerciseStatus.UNRELIABLE
            score  = 0
        elif self._hold_time >= self.required_hold_sec:
            status = ExerciseStatus.DONE
            score  = self.max_score
        elif self._hold_time >= self.min_hold_sec:
            status = ExerciseStatus.PARTIAL
            # Пропорционально: от 50% до 100% max_score
            ratio = (self._hold_time - self.min_hold_sec) / max(
                0.001, self.required_hold_sec - self.min_hold_sec)
            score = round(self.max_score * (0.5 + 0.5 * min(1.0, ratio)))
        else:
            status = ExerciseStatus.PARTIAL
            score  = 0

        notes = []
        if vtr < MIN_TRACKING_RATIO:
            notes.append("Низкое качество трекинга — результат технически ненадёжен")
        if jitter > 0.02:
            notes.append("Обнаружено дрожание руки")

        return ExerciseResult(
            exercise_id=self.exercise_id,
            status=status,
            score=score,
            max_score=self.max_score,
            hold_time_sec=self._hold_time,
            valid_tracking_ratio=vtr,
            metrics={"jitter": round(jitter, 4)},
            notes=notes,
        )
