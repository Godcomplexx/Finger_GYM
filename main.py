"""
Finger Gym — модуль предварительного тестирования мелкой моторики.
Запуск: python main.py [--camera N] [--patient ID]
"""
from __future__ import annotations
import sys
import os
import uuid
import argparse
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app_info import ALGORITHM_VERSION, MODULE_VERSION
from src.audit import log_event
from src.models import Hand, TestSession, TrackingFrame
from src.tracking.factory import create_tracker
from src.processing.calibration import CalibrationCollector
from src.exercises.exercises import create_exercises, EXERCISE_ORDER
from src.scoring.engine import build_summary
from src.storage.session_storage import save_session
from src.storage.pdf_report import save_pdf_report
from src.presentation.renderer import Renderer
from src.audio import (
    sound_exercise_done, sound_zone_hit, sound_calibration_ok,
    sound_hand_lost, sound_session_complete,
)

WINDOW = "Finger Gym"
DWELL_SECONDS = 1.1
RESULT_DWELL_SECONDS = 1.8

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
    cw = min(1200, width - 60)
    ch = min(420, height - 120)
    cx = (width - cw) // 2
    cy = (height - ch) // 2
    gap = 70
    btn_top = cy + 118
    btn_bottom = cy + ch - 34
    btn_w = (cw - 56 - gap) // 2
    left_x1 = cx + 28
    right_x1 = left_x1 + btn_w + gap
    return {
        Hand.LEFT: (left_x1, btn_top, left_x1 + btn_w, btn_bottom),
        Hand.RIGHT: (right_x1, btn_top, right_x1 + btn_w, btn_bottom),
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


def wait_exercise_result_action(
    cap: cv2.VideoCapture,
    tracker,
    renderer: Renderer,
    exercise,
    result,
    exercise_num: int,
    total_exercises: int,
    width: int,
    height: int,
) -> str:
    dwell_target: str | None = None
    dwell_started = 0.0
    while True:
        ok, bgr = read_frame(cap, width, height)
        if not ok:
            continue
        frame = safe_process(tracker, bgr)
        pointer = hand_pointer(frame, width, height)
        rects = {
            "repeat": renderer.repeat_button_rect(),
            "next": renderer.next_button_rect(),
        }
        click_target = hit_target(consume_mouse_click(), rects)
        pointer_target = hit_target(pointer, rects)
        now = time.monotonic()
        if pointer_target is not None:
            if pointer_target != dwell_target:
                dwell_target = pointer_target
                dwell_started = now
            dwell_ratio = min(1.0, (now - dwell_started) / RESULT_DWELL_SECONDS)
        else:
            dwell_target = None
            dwell_started = 0.0
            dwell_ratio = 0.0
        img = renderer.draw_exercise_result(
            bgr,
            exercise,
            result,
            exercise_num,
            total_exercises,
            pointer=pointer,
            hover_target=dwell_target,
            dwell_ratio=dwell_ratio,
        )
        img = renderer.draw_tracking_overlay(img, frame)
        key = show(img)

        if should_quit(key):
            return "quit"
        if key in (13, 10) or click_target == "next" or (
            dwell_target == "next" and dwell_ratio >= 1.0
        ):
            return "next"
        if key in (ord(" "), ord("r"), ord("R")) or click_target == "repeat" or (
            dwell_target == "repeat" and dwell_ratio >= 1.0
        ):
            return "repeat"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera",  type=int, default=0)
    p.add_argument("--patient", type=str, default="patient-001")
    p.add_argument("--width",   type=int, default=1280)
    p.add_argument("--height",  type=int, default=720)
    p.add_argument("--tracker", choices=("mediapipe",), default="mediapipe")
    p.add_argument("--debug", action="store_true", help="Показывать панель curl пальцев")
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


def safe_process(tracker, bgr: np.ndarray) -> TrackingFrame:
    """Вызов tracker.process с перехватом исключений."""
    try:
        return tracker.process(bgr)
    except Exception as exc:
        print(f"[WARN] tracker.process error: {exc}")
        return TrackingFrame(timestamp=time.monotonic(), landmarks=[], is_valid=False)


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


def get_screen_size(default_width: int, default_height: int) -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return default_width, default_height


def configure_main_window(width: int, height: int) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    screen_w, screen_h = get_screen_size(width, height)
    available_w = max(640, screen_w - 40)
    available_h = max(480, screen_h - 100)
    scale = min(available_w / width, available_h / height)
    window_w = int(width * scale)
    window_h = int(height * scale)
    cv2.resizeWindow(WINDOW, window_w, window_h)
    cv2.moveWindow(WINDOW, max(0, (screen_w - window_w) // 2), 20)



def show_startup_error(args: argparse.Namespace, renderer: Renderer, message: str) -> None:
    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    img = renderer.draw_error_message(blank, "Трекер недоступен", message)
    cv2.imshow(WINDOW, img)
    cv2.waitKey(0)


def run():
    args = parse_args()
    cap: cv2.VideoCapture | None = None

    configure_main_window(args.width, args.height)
    cv2.setMouseCallback(WINDOW, on_mouse)

    renderer = Renderer(args.width, args.height, debug=args.debug)

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
            frame = safe_process(tracker, bgr)
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

        # ── Шаг 1b: ввод patient ID ───────────────────────────────────────────
        patient_id_text = args.patient if args.patient != "patient-001" else ""
        patient_id_error = ""
        while True:
            ok, bgr = read_frame(cap, args.width, args.height)
            img = renderer.draw_patient_id_input(bgr, patient_id_text, patient_id_error)
            key = show(img)
            if key == 27:  # Esc — использовать дефолт
                patient_id_text = patient_id_text or "patient-001"
                break
            if key in (13, 10):  # Enter — подтвердить
                stripped = patient_id_text.strip()
                if not stripped:
                    patient_id_error = "ID не может быть пустым"
                elif len(stripped) > 64:
                    patient_id_error = "ID слишком длинный (максимум 64 символа)"
                else:
                    patient_id_text = stripped
                    break
            elif key == 8:  # Backspace
                patient_id_text = patient_id_text[:-1]
                patient_id_error = ""
            elif 32 <= key <= 126:  # печатаемые ASCII-символы
                patient_id_text += chr(key)
                patient_id_error = ""
            elif window_closed():
                cleanup(cap)
                return

        session = TestSession(
            session_id=str(uuid.uuid4())[:8],
            patient_id=patient_id_text,
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

        # ── Шаг 2: позиционирование руки ─────────────────────────────────────
        from src.processing.metrics import compute_palm_width
        POSITION_OK_SEC   = 2.0   # сколько секунд держать «хорошее» положение
        position_ok_since: float | None = None
        while True:
            ok, bgr = read_frame(cap, args.width, args.height)
            if not ok:
                continue
            frame = safe_process(tracker, bgr)
            pw    = compute_palm_width(frame) if frame.is_valid else 0.0
            # 0.10–0.22 — «в норме»; < 0.10 — далеко; > 0.22 — близко
            distance_status = (
                "far"  if pw < 0.10 else
                "close" if pw > 0.22 else
                "ok"
            )
            now = time.monotonic()
            if frame.is_valid and distance_status == "ok":
                if position_ok_since is None:
                    position_ok_since = now
                held = now - position_ok_since
            else:
                position_ok_since = None
                held = 0.0

            img = renderer.draw_positioning_guide(
                bgr, frame, distance_status, held, POSITION_OK_SEC,
            )
            key = show(img)
            if should_quit(key):
                cleanup(cap)
                return
            if held >= POSITION_OK_SEC:
                break

        # ── Шаг 3: калибровка (с повтором при сбое) ───────────────────────────
        calibration_attempt = 0
        calibration = None
        while calibration is None:
            calibration_attempt += 1
            collector = CalibrationCollector(duration=2.0)
            while not collector.is_done():
                ok, bgr = read_frame(cap, args.width, args.height)
                if not ok:
                    continue
                frame = safe_process(tracker, bgr)
                collector.feed(frame)
                img   = renderer.draw_calibration(bgr, frame, collector.elapsed(), 2.0)
                img   = renderer.draw_tracking_overlay(img, frame)
                key   = show(img)

                if should_quit(key):
                    cleanup(cap)
                    return

            result = collector.result()
            if result.is_ready:
                calibration = result
            else:
                log_event(session, "calibration_failed", "Калибровка непригодна для интерпретации", details={
                    "attempt": calibration_attempt,
                    "validRatio": round(collector.valid_ratio(), 3),
                })
                # Показываем экран сбоя, ждём Space (повторить) или Esc (выйти)
                while True:
                    ok, bgr = read_frame(cap, args.width, args.height)
                    img = renderer.draw_calibration_failed(bgr, collector.valid_ratio(), calibration_attempt)
                    key = show(img)
                    if should_quit(key):
                        session.calibration = result
                        save_session(session)
                        cleanup(cap)
                        return
                    if key in (ord(' '), 13):
                        break  # повторить калибровку

        session.calibration = calibration
        sound_calibration_ok()
        log_event(session, "calibration_completed", "Калибровка завершена", details={
            "attempt": calibration_attempt,
            "validRatio": round(collector.valid_ratio(), 3),
            "palmWidth": round(calibration.palm_width, 4),
        })

        # ── Шаг 4: упражнения ─────────────────────────────────────────────────
        exercises = create_exercises(calibration)
        results   = []
        ex_index  = 0
        total_ex  = len(exercises)
        active_exercise_id: int | None = None
        prepare_countdown_started: float | None = None
        prepare_dwell_started = 0.0

        while ex_index < total_ex:
            exercise = exercises[ex_index]
            if id(exercise) != active_exercise_id:
                active_exercise_id = id(exercise)
                prepare_countdown_started = None
                prepare_dwell_started = 0.0
                exercise._prepare_start = time.monotonic()
                exercise._prepare_confirmed = False
                exercise._hand_detected_at = None
            ok, bgr = read_frame(cap, args.width, args.height)
            if not ok:
                continue
            frame = safe_process(tracker, bgr)

            if exercise.is_preparing() or prepare_countdown_started is not None:
                now = time.monotonic()
                pointer = hand_pointer(frame, args.width, args.height)
                start_rects = {"start": renderer.start_button_rect()}
                click_target = hit_target(consume_mouse_click(), start_rects)
                pointer_target = hit_target(pointer, start_rects)
                hover_start = False
                dwell_ratio = 0.0

                if prepare_countdown_started is None:
                    exercise._prepare_start = now
                    if pointer_target == "start":
                        hover_start = True
                        if prepare_dwell_started == 0.0:
                            prepare_dwell_started = now
                        dwell_ratio = min(1.0, (now - prepare_dwell_started) / RESULT_DWELL_SECONDS)
                    else:
                        prepare_dwell_started = 0.0
                    countdown_remaining = None
                else:
                    countdown_remaining = max(0.0, 3.0 - (now - prepare_countdown_started))
                    if countdown_remaining <= 0:
                        exercise._prepare_confirmed = True
                        prepare_countdown_started = None
                        log_event(session, "exercise_started", "Exercise started", details={
                            "exerciseId": exercise.exercise_id,
                        })
                        continue

                img = renderer.draw_prepare(
                    bgr, exercise, ex_index + 1, total_ex,
                    pointer=pointer,
                    hover_start=hover_start,
                    dwell_ratio=dwell_ratio,
                    countdown_remaining=countdown_remaining,
                )
                img = renderer.draw_tracking_overlay(img, frame)
                key = show(img)
                if should_quit(key):
                    break
                if key in (ord('s'), ord('S')):
                    result = exercise.evaluate()
                    result.notes.append("Пропущено оператором")
                    log_event(session, "exercise_skipped", "Exercise skipped by operator", details={
                        "exerciseId": exercise.exercise_id,
                    })
                    action = wait_exercise_result_action(
                        cap, tracker, renderer, exercise, result,
                        ex_index + 1, total_ex, args.width, args.height,
                    )
                    if action == "quit":
                        break
                    if action == "repeat":
                        exercises[ex_index] = EXERCISE_ORDER[ex_index](calibration)
                    else:
                        results.append(result)
                        ex_index += 1
                elif (
                    key in (ord(' '), 13)
                    or click_target == "start"
                    or (pointer_target == "start" and dwell_ratio >= 1.0)
                ):
                    prepare_countdown_started = time.monotonic()
                    prepare_dwell_started = 0.0
                continue

            # ── Активная фаза: принимаем кадры и считаем удержание ───────────
            was_hand_lost = not frame.is_valid and exercise.is_hand_lost()
            exercise.feed(frame)
            if not frame.is_valid and exercise.is_hand_lost() and not was_hand_lost:
                sound_hand_lost()
            from src.exercises.exercises import ZoneMovementExercise as _ZME
            if isinstance(exercise, _ZME) and exercise.consume_zone_hit():
                sound_zone_hit()
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
                log_event(session, "exercise_skipped", "Exercise skipped by operator", details={
                    "exerciseId": exercise.exercise_id,
                })
                action = wait_exercise_result_action(
                    cap, tracker, renderer, exercise, result,
                    ex_index + 1, total_ex, args.width, args.height,
                )
                if action == "quit":
                    break
                if action == "repeat":
                    exercises[ex_index] = EXERCISE_ORDER[ex_index](calibration)
                else:
                    results.append(result)
                    ex_index += 1
                continue

            if exercise.is_complete():
                result = exercise.evaluate()
                sound_exercise_done()
                log_event(session, "exercise_completed", "Exercise completed", details={
                    "exerciseId": exercise.exercise_id,
                    "status": result.status.value,
                    "score": result.score,
                    "validTrackingRatio": round(result.valid_tracking_ratio, 3),
                })
                action = wait_exercise_result_action(
                    cap, tracker, renderer, exercise, result,
                    ex_index + 1, total_ex, args.width, args.height,
                )
                if action == "quit":
                    break
                if action == "repeat":
                    exercises[ex_index] = EXERCISE_ORDER[ex_index](calibration)
                else:
                    results.append(result)
                    ex_index += 1

        # ── Шаг 5: итоги ──────────────────────────────────────────────────────
        if results:
            summary         = build_summary(results)
            session.summary = summary
            log_event(session, "summary_built", "Session summary built", details={
                "totalScore": summary.total_score,
                "qualityCategory": summary.quality_category.value,
                "recommendation": summary.recommendation.mode.value,
            })
            filepath        = save_session(session)
            pdf_path        = save_pdf_report(session, filepath)
            sound_session_complete()
            print(f"[OK] {filepath}")
            if pdf_path:
                print(f"[OK] {pdf_path}")
            print(f"     Балл: {summary.total_score}/80  |  {summary.recommendation.label}")
            if summary.icf_codes:
                icf_text = []
                for item in summary.icf_codes:
                    if item.problem_percent is None:
                        icf_text.append(item.formatted_code)
                    else:
                        icf_text.append(f"{item.formatted_code} ({item.problem_percent}%)")
                print(f"     МКФ: {', '.join(icf_text)}")

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
