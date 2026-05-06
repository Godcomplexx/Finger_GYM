"""Звуковая обратная связь. Использует winsound на Windows, иначе — заглушка."""
from __future__ import annotations
import sys
import threading


def _beep_async(freq: int, duration_ms: int) -> None:
    if sys.platform != "win32":
        return
    try:
        import winsound
        threading.Thread(
            target=winsound.Beep,
            args=(freq, duration_ms),
            daemon=True,
        ).start()
    except Exception:
        pass


def sound_exercise_done() -> None:
    """Упражнение выполнено успешно."""
    _beep_async(880, 120)


def sound_zone_hit() -> None:
    """Зона достигнута при zone_movement."""
    _beep_async(660, 80)


def sound_calibration_ok() -> None:
    """Калибровка успешно завершена."""
    _beep_async(523, 100)


def sound_hand_lost() -> None:
    """Рука потеряна во время упражнения."""
    _beep_async(220, 80)


def sound_session_complete() -> None:
    """Сессия полностью завершена."""
    _beep_async(784, 150)
