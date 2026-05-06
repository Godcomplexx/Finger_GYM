"""
Finger Gym — модуль предварительного тестирования мелкой моторики.
Запуск: python main.py [--camera N] [--patient ID]
"""
from __future__ import annotations
import sys
import uuid
import argparse
import time
import cv2
import numpy as np

sys.path.insert(0, __file__.replace("main.py", ""))

from src.app_info import ALGORITHM_VERSION, MODULE_VERSION
from src.audit import log_event
from src.models import Hand, TestSession
from src.tracking.factory import create_tracker
from src.processing.calibration import CalibrationCollector
from src.exercises.exercises import create_exercises, EXERCISE_ORDER
from src.scoring.engine import build_summary
from src.storage.session_storage import save_session
from src.presentation.renderer import Renderer

WINDOW = "Finger Gym"
DWELL_SECONDS = 1.1

_mouse_state = {"pos": (-1, -1), "click": None}


def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    _mouse_state["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_state["click"] = (x, y)


def consume_mouse_click() -> tuple[int, int] | None:
    click = _mouse_state["click"]
    _mouse_state["click"] = None
    return click


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2



def hand_button_rects(width: int, height: int) -> dict[Hand, tuple[int, int, int, int]]:
    cw, ch = 560, 300
    cx = (width - cw) // 2
    cy = (height - ch) // 2
    return {
        Hand.RIGHT: (cx + 28, cy + 78, cx + cw // 2 - 18, cy + 178),
        Hand.LEFT: (cx + cw // 2 + 18, cy + 78, cx + cw - 28, cy + 178),
    }


def hit_target(point: tuple[int, int] | None, rects: dict) -> object | None:
    if point is None:
        return None
    for target, rect in rects.items():
        if point_in_rect(point, rect):
            return target
    return None


def hand_pointer(frame, width: int, height: int) -> tuple[int, int] | None:
    if not frame.is_valid or len(frame.landmarks) <= 8:
        return None
    tip = frame.landmarks[8]
    return int(tip.x * width), int(tip.y * height)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera",  type=int, default=0)
    p.add_argument("--patient", type=str, default="patient-001")
    p.add_argument("--width",   type=int, default=1280)
    p.add_argument("--height",  type=int, default=720)
    p.add_argument("--tracker", choices=("mediapipe",), default="mediapipe")
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


def read_frame(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
) -> tuple[bool, np.ndarray]:
    ok, bgr = cap.read()
    if not ok:
        return False, np.zeros((height, width, 3), dtype=np.uint8)
    return True, cv2.flip(bgr, 1)


def cleanup(cap: cv2.VideoCapture | None):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()



def show_startup_error(args: argparse.Namespace, renderer: Renderer, message: str) -> None:
    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    img = renderer.draw_error_message(blank, "Трекер недоступен", message)
    cv2.imshow(WINDOW, img)
    cv2.waitKey(0)


def run():
    args = parse_args()
    cap: cv2.VideoCapture | None = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)
    cv2.setMouseCallback(WINDOW, on_mouse)

    renderer = Renderer(args.width, args.height)

    try:
        tracker_context = create_tracker(args.tracker)
    except RuntimeError as exc:
        show_startup_error(args, renderer, str(exc))
        cleanup(cap)
        return

    with tracker_context as tracker:
        cap = open_camera(args.camera, args.width, args.height)

        # ── Шаг 1: выбор руки ─────────────────────────────────────────────────
        selected_hand: Hand | None = None
        hand_rects = hand_button_rects(args.width, args.height)
        dwell_target: Hand | None = None
        dwell_started = 0.0
        while selected_hand is None:
            ok, bgr = read_frame(cap, args.width, args.height)
            if not ok:
                continue
            frame = tracker.process(bgr)
            pointer = hand_pointer(frame, args.width, args.height)
            hand_hover = hit_target(pointer, hand_rects)
            mouse_hover = hit_target(_mouse_state["pos"], hand_rects)
            click_target = hit_target(consume_mouse_click(), hand_rects)

            if click_target:
                selected_hand = click_target
                break

            now = time.monotonic()
            if hand_hover is not None:
                if hand_hover != dwell_target:
                    dwell_target = hand_hover
                    dwell_started = now
                dwell_ratio = min(1.0, (now - dwell_started) / DWELL_SECONDS)
                if dwell_ratio >= 1.0:
                    selected_hand = hand_hover
                    break
            else:
                dwell_target = None
                dwell_started = 0.0
                dwell_ratio = 0.0

            hover = hand_hover or mouse_hover
            img   = renderer.draw_hand_select(
                bgr,
                hover_key=hover,
                dwell_ratio=dwell_ratio if hand_hover else 0.0,
                pointer=pointer,
            )
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
            module_version=MODULE_VERSION,
            algorithm_version=ALGORITHM_VERSION,
            model_name=tracker.model_name,
            model_sha256=tracker.model_sha256,
            tracking_source=tracker.source_name,
        )
        log_event(session, "session_started", "Сессия тестирования начата", details={
            "camera": args.camera,
            "width": args.width,
            "height": args.height,
            "hand": selected_hand.value,
            "tracker": tracker.source_name,
        })
        log_event(session, "model_loaded", "Модель трекинга загружена", details={
            "modelName": tracker.model_name,
            "modelSha256": tracker.model_sha256,
            "tracker": tracker.source_name,
        })

        # ── Шаг 2: калибровка ─────────────────────────────────────────────────
        collector = CalibrationCollector(duration=2.0)
        while not collector.is_done():
            ok, bgr = read_frame(cap, args.width, args.height)
            if not ok:
                continue
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
            session.calibration = calibration
            log_event(session, "calibration_failed", "Калибровка непригодна для интерпретации", details={
                "validRatio": round(collector.valid_ratio(), 3),
            })
            filepath = save_session(session)
            print(f"[WARN] Калибровка не пройдена. Протокол сохранён: {filepath}")
            cleanup(cap)
            return
        session.calibration = calibration
        log_event(session, "calibration_completed", "Калибровка завершена", details={
            "validRatio": round(collector.valid_ratio(), 3),
            "palmWidth": round(calibration.palm_width, 4),
        })

        # ── Шаг 3: упражнения ─────────────────────────────────────────────────
        exercises = create_exercises(calibration)
        results   = []
        ex_index  = 0
        total_ex  = len(exercises)

        while ex_index < total_ex:
            exercise = exercises[ex_index]
            ok, bgr = read_frame(cap, args.width, args.height)
            if not ok:
                continue
            frame = tracker.process(bgr)

            # ── Фаза подготовки: показываем инструкцию, автостарт ────────────
            if exercise.is_preparing():
                exercise.notify_hand_visible(frame)  # для автостарта по руке
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
                    log_event(session, "exercise_skipped", "Exercise skipped by operator", details={
                        "exerciseId": exercise.exercise_id,
                    })
                    ex_index += 1
                elif key in (ord(' '), 13):
                    exercise.confirm_start()
                    log_event(session, "exercise_started", "Exercise started", details={
                        "exerciseId": exercise.exercise_id,
                    })
                else:
                    # Автостарт: is_preparing вернул False на следующей итерации
                    if not exercise.is_preparing():
                        log_event(session, "exercise_started", "Exercise started", details={
                            "exerciseId": exercise.exercise_id,
                        })
                continue  # ждём конца фазы подготовки

            # ── Активная фаза: принимаем кадры и считаем удержание ───────────
            exercise.feed(frame)
            img = renderer.draw_exercise(bgr, exercise, frame,
                                         ex_index + 1, total_ex)
            img = renderer.draw_tracking_overlay(img, frame)
            key = show(img)

            if should_quit(key):
                break

            if key == ord(' '):  # Space — повторить упражнение заново
                exercises[ex_index] = EXERCISE_ORDER[ex_index](calibration)
                log_event(session, "exercise_restarted", "Exercise restarted by user", details={
                    "exerciseId": exercise.exercise_id,
                })
                continue

            if key in (ord('s'), ord('S')):
                result = exercise.evaluate()
                result.notes.append("Пропущено оператором")
                results.append(result)
                log_event(session, "exercise_skipped", "Exercise skipped by operator", details={
                    "exerciseId": exercise.exercise_id,
                })
                ex_index += 1
                continue

            if exercise.is_complete() or exercise.is_timeout():
                result = exercise.evaluate()
                results.append(result)
                log_event(session, "exercise_completed", "Exercise completed", details={
                    "exerciseId": exercise.exercise_id,
                    "status": result.status.value,
                    "score": result.score,
                    "validTrackingRatio": round(result.valid_tracking_ratio, 3),
                })
                ex_index += 1

        # ── Шаг 4: итоги ──────────────────────────────────────────────────────
        if results:
            summary         = build_summary(results)
            session.summary = summary
            log_event(session, "summary_built", "Session summary built", details={
                "totalScore": summary.total_score,
                "qualityCategory": summary.quality_category.value,
                "recommendation": summary.recommendation.mode.value,
            })
            filepath        = save_session(session)
            print(f"[OK] {filepath}")
            print(f"     Балл: {summary.total_score}/100  |  {summary.recommendation.label}")

            SUMMARY_SHOW_SEC = 12.0
            summary_start = time.monotonic()
            while True:
                ok, bgr = read_frame(cap, args.width, args.height)
                remaining = max(0.0, SUMMARY_SHOW_SEC - (time.monotonic() - summary_start))
                img = renderer.draw_summary(bgr, summary, autoclose_sec=remaining)
                key = show(img)

                if key != 255 or window_closed() or remaining <= 0:
                    break

    cleanup(cap)


if __name__ == "__main__":
    run()
