from __future__ import annotations
import time
from src.models import TrackingFrame, CalibrationProfile
from src.processing.metrics import (
    compute_palm_width, compute_palm_center,
    avg_tip_to_palm_distance, thumb_index_distance,
)

CALIBRATION_DURATION = 2.0      # секунд удержания открытой ладони
MIN_VALID_RATIO = 0.70           # минимум валидных кадров для принятия калибровки
MIN_PALM_WIDTH = 0.05            # защита от артефактов (рука слишком маленькая в кадре)


class CalibrationCollector:
    """
    Накапливает кадры при показе открытой ладони и вычисляет CalibrationProfile.
    Использование:
        collector.feed(frame)  — в цикле обработки кадров
        collector.is_done()    — True когда собрано достаточно данных
        collector.result()     — возвращает CalibrationProfile
    """

    def __init__(self, duration: float = CALIBRATION_DURATION):
        self._duration = duration
        self._frames: list[TrackingFrame] = []
        self._start_time: float | None = None

    def reset(self):
        self._frames = []
        self._start_time = None

    def feed(self, frame: TrackingFrame):
        if self._start_time is None and frame.is_valid:
            self._start_time = time.monotonic()
        if frame.is_valid:
            self._frames.append(frame)

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def is_done(self) -> bool:
        return self.elapsed() >= self._duration and len(self._frames) > 0

    def valid_ratio(self) -> float:
        elapsed = self.elapsed()
        if elapsed <= 0 or not self._frames:
            return 0.0
        # оцениваем по числу кадров за время (предполагаем ~30 fps)
        expected = elapsed * 30
        return min(1.0, len(self._frames) / expected)

    def result(self) -> CalibrationProfile:
        if not self._frames:
            return CalibrationProfile(
                palm_width=0.0, palm_center=None,
                base_tip_to_palm=0.0, base_thumb_index=0.0,
                is_ready=False,
            )

        valid_frames = [f for f in self._frames if f.is_valid]
        if not valid_frames:
            return CalibrationProfile(
                palm_width=0.0, palm_center=None,
                base_tip_to_palm=0.0, base_thumb_index=0.0,
                is_ready=False,
            )

        # Усредняем по всем валидным кадрам
        widths = [compute_palm_width(f) for f in valid_frames]
        palm_width = sum(widths) / len(widths)

        if palm_width < MIN_PALM_WIDTH:
            return CalibrationProfile(
                palm_width=palm_width, palm_center=None,
                base_tip_to_palm=0.0, base_thumb_index=0.0,
                is_ready=False,
            )

        centers = [compute_palm_center(f) for f in valid_frames]
        from src.models import Point2D
        palm_center = Point2D(
            x=sum(c.x for c in centers) / len(centers),
            y=sum(c.y for c in centers) / len(centers),
        )

        tip_to_palms = [avg_tip_to_palm_distance(f, palm_width) for f in valid_frames]
        base_tip_to_palm = sum(tip_to_palms) / len(tip_to_palms)

        thumb_indices = [thumb_index_distance(f, palm_width) for f in valid_frames]
        base_thumb_index = sum(thumb_indices) / len(thumb_indices)

        valid_ratio = len(valid_frames) / len(self._frames)

        return CalibrationProfile(
            palm_width=palm_width,
            palm_center=palm_center,
            base_tip_to_palm=base_tip_to_palm,
            base_thumb_index=base_thumb_index,
            is_ready=valid_ratio >= MIN_VALID_RATIO,
        )
