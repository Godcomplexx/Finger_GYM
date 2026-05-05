"""
Finger Gym — модуль предварительного тестирования мелкой моторики.
Запуск: python main.py [--camera N] [--patient ID]
"""
from __future__ import annotations
import sys
import uuid
import argparse
import cv2
import numpy as np

sys.path.insert(0, __file__.replace("main.py", ""))

from src.models import Hand, TestSession
from src.tracking.adapter import TrackingAdapter
from src.processing.calibration import CalibrationCollector
from src.exercises.exercises import create_exercises
from src.scoring.engine import build_summary
from src.storage.session_storage import save_session
from src.presentation.renderer import Renderer

WINDOW = "Finger Gym"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera",  type=int, default=0)
    p.add_argument("--patient", type=str, default="patient-001")
    p.add_argument("--width",   type=int, default=1280)
    p.add_argument("--height",  type=int, default=720)
    return p.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру #{index}")
    return cap


def window_closed() -> bool:
    """True если пользователь закрыл окно кнопкой X."""
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def should_quit(key: int) -> bool:
    """True при нажатии Q/Esc или закрытии окна."""
    return key in (ord('q'), ord('Q'), 27) or window_closed()


def show(img: np.ndarray) -> int:
    """Показать кадр и вернуть нажатую клавишу (0xFF если не нажата)."""
    cv2.imshow(WINDOW, img)
    return cv2.waitKey(1) & 0xFF


def cleanup(cap: cv2.VideoCapture):
    cap.release()
    cv2.destroyAllWindows()


def run():
    args = parse_args()
    cap  = open_camera(args.camera, args.width, args.height)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)

    renderer = Renderer(args.width, args.height)

    with TrackingAdapter() as tracker:

        # ── Шаг 1: выбор руки ─────────────────────────────────────────────────
        selected_hand: Hand | None = None
        while selected_hand is None:
            ok, bgr = cap.read()
            if not ok:
                continue
            bgr   = cv2.flip(bgr, 1)
            frame = tracker.process(bgr)
            img   = renderer.draw_hand_select(bgr)
            img   = renderer.draw_tracking_overlay(img, frame)
            key   = show(img)

            if should_quit(key):
                cleanup(cap)
                return
            if key in (ord('r'), ord('R')):
                selected_hand = Hand.RIGHT
            elif key in (ord('l'), ord('L')):
                selected_hand = Hand.LEFT

        session = TestSession(
            session_id=str(uuid.uuid4())[:8],
            patient_id=args.patient,
            hand=selected_hand,
        )

        # ── Шаг 2: калибровка ─────────────────────────────────────────────────
        collector = CalibrationCollector(duration=2.0)
        while not collector.is_done():
            ok, bgr = cap.read()
            if not ok:
                continue
            bgr   = cv2.flip(bgr, 1)
            frame = tracker.process(bgr)
            collector.feed(frame)
            img   = renderer.draw_calibration(bgr, frame, collector.elapsed(), 2.0)
            img   = renderer.draw_tracking_overlay(img, frame)
            key   = show(img)

            if should_quit(key):
                cleanup(cap)
                return

        calibration = collector.result()
        if not calibration.is_ready:
            calibration.palm_width = calibration.palm_width or 0.15
            calibration.is_ready   = True
        session.calibration = calibration

        # ── Шаг 3: упражнения ─────────────────────────────────────────────────
        exercises = create_exercises(calibration)
        results   = []
        ex_index  = 0
        total_ex  = len(exercises)

        while ex_index < total_ex:
            exercise = exercises[ex_index]
            ok, bgr  = cap.read()
            if not ok:
                continue
            bgr   = cv2.flip(bgr, 1)
            frame = tracker.process(bgr)

            # ── Фаза подготовки: показываем инструкцию, ждём Space/Enter ────
            if exercise.is_preparing():
                img = renderer.draw_prepare(bgr, exercise,
                                            ex_index + 1, total_ex)
                img = renderer.draw_tracking_overlay(img, frame)
                key = show(img)
                if should_quit(key):
                    break
                if key in (ord('s'), ord('S')):
                    result = exercise.evaluate()
                    result.notes.append("Пропущено оператором")
                    results.append(result)
                    ex_index += 1
                elif key in (ord(' '), 13):  # Space или Enter
                    exercise.confirm_start()
                continue  # ждём конца фазы подготовки

            # ── Активная фаза: принимаем кадры и считаем удержание ───────────
            exercise.feed(frame)
            img = renderer.draw_exercise(bgr, exercise, frame,
                                         ex_index + 1, total_ex)
            img = renderer.draw_tracking_overlay(img, frame)
            key = show(img)

            if should_quit(key):
                break

            if key in (ord('s'), ord('S')):
                result = exercise.evaluate()
                result.notes.append("Пропущено оператором")
                results.append(result)
                ex_index += 1
                continue

            if exercise.is_complete() or exercise.is_timeout():
                results.append(exercise.evaluate())
                ex_index += 1

        # ── Шаг 4: итоги ──────────────────────────────────────────────────────
        if results:
            summary         = build_summary(results)
            session.summary = summary
            filepath        = save_session(session)
            print(f"[OK] {filepath}")
            print(f"     Балл: {summary.total_score}/100  |  {summary.recommendation.label}")

            while True:
                ok, bgr = cap.read()
                bgr     = cv2.flip(bgr, 1) if ok else np.zeros(
                    (args.height, args.width, 3), dtype=np.uint8)
                img = renderer.draw_summary(bgr, summary)
                key = show(img)

                if key != 255 or window_closed():
                    break

    cleanup(cap)


if __name__ == "__main__":
    run()
