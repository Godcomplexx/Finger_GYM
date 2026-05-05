from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import TrackingFrame, TestSummary
from src.exercises.base import BaseExercise
from src.exercises.exercises import ZoneMovementExercise, ZONES, ZONE_RADIUS

# ── Палитра (RGB для PIL, BGR для OpenCV) ─────────────────────────────────────
_BG        = (18,  18,  26)
_PANEL     = (28,  30,  42)
_BORDER    = (55,  60,  85)
_WHITE     = (240, 240, 250)
_GRAY      = (140, 145, 165)
_ACCENT    = (100, 200, 255)
_GREEN     = (70,  210, 120)
_YELLOW    = (230, 200,  60)
_RED       = (230,  80,  80)
_ORANGE    = (230, 160,  60)

# BGR-версии для OpenCV-примитивов (линии, круги)
def _bgr(rgb): return (rgb[2], rgb[1], rgb[0])

_BG_BGR     = _bgr(_BG)
_PANEL_BGR  = _bgr(_PANEL)
_BORDER_BGR = _bgr(_BORDER)
_GREEN_BGR  = _bgr(_GREEN)
_ACCENT_BGR = _bgr(_ACCENT)
_RED_BGR    = _bgr(_RED)
_WHITE_BGR  = _bgr(_WHITE)
_GRAY_BGR   = _bgr(_GRAY)

# ── Шрифты (Segoe UI — хорошая кириллица) ────────────────────────────────────
_FONT_PATH      = "C:/Windows/Fonts/segoeui.ttf"
_FONT_BOLD_PATH = "C:/Windows/Fonts/segoeuib.ttf"

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = _FONT_BOLD_PATH if bold else _FONT_PATH
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

_F_SM   = _font(18)
_F_MD   = _font(22)
_F_LG   = _font(28)
_F_XL   = _font(38)
_F_XXL  = _font(64, bold=True)
_F_MD_B = _font(22, bold=True)
_F_LG_B = _font(28, bold=True)
_F_XL_B = _font(38, bold=True)

GESTURE_LABELS = {
    "open_palm":     "Открытая ладонь",
    "fist":          "Кулак",
    "pinch":         "Щипковый захват",
    "point_gesture": "Указательный жест",
    "palm_facing":   "Поворот кисти — внутренняя сторона",
    "back_facing":   "Поворот кисти — тыльная сторона",
    "zone_movement": "Перемещение по зонам",
    "hold_still":    "Удержание руки",
}


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def _put(draw: ImageDraw.ImageDraw, text: str, pos: tuple,
         font: ImageFont.FreeTypeFont,
         color: tuple = _WHITE,
         shadow: bool = True):
    """Текст с тёмной тенью для читаемости поверх кадра."""
    x, y = pos
    if shadow:
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=color)

def _panel(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
           alpha: float = 0.82, bg: tuple = _PANEL_BGR):
    """Полупрозрачный прямоугольник поверх кадра."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), _BORDER_BGR, 1, cv2.LINE_AA)

def _progress_bar(img: np.ndarray, x: int, y: int, w: int, h: int,
                  ratio: float, fg: tuple = _GREEN_BGR):
    cv2.rectangle(img, (x, y), (x + w, y + h), _PANEL_BGR, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), _BORDER_BGR, 1)
    fill = max(0, min(w - 2, int((w - 2) * ratio)))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), fg, -1)

def _circle_progress(img: np.ndarray, cx: int, cy: int, r: int,
                     ratio: float, color: tuple = _GREEN_BGR, thick: int = 5):
    cv2.circle(img, (cx, cy), r, _BORDER_BGR, thick)
    if ratio > 0:
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0,
                    int(360 * ratio), color, thick, cv2.LINE_AA)

def _tracking_dot(img: np.ndarray, x: int, y: int, is_valid: bool):
    color = _GREEN_BGR if is_valid else _RED_BGR
    cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 9, color, 1,  cv2.LINE_AA)

def _darken(bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    dark = np.zeros_like(bgr)
    cv2.addWeighted(bgr, 1 - alpha, dark, alpha, 0, bgr)
    return bgr


# ── Главный рендерер ──────────────────────────────────────────────────────────

class Renderer:

    def __init__(self, width: int = 1280, height: int = 720):
        self.w = width
        self.h = height

    def _base(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.w, self.h))
        return _darken(img, 0.42)

    # ── Шапка ─────────────────────────────────────────────────────────────────

    def _draw_header(self, img: np.ndarray, subtitle: str = ""):
        _panel(img, 0, 0, self.w, 66, alpha=0.90, bg=_BG_BGR)
        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)
        _put(d, "Finger Gym", (22, 8),  _F_XL_B, _ACCENT)
        if subtitle:
            _put(d, subtitle, (22, 42), _F_SM, _GRAY)
        img[:] = _pil_to_bgr(pil)

    # ── Экран выбора руки ─────────────────────────────────────────────────────

    def draw_hand_select(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Модуль тестирования мелкой моторики кисти")

        # Центральная карточка
        cw, ch = 560, 300
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.88)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        _put(d, "Выберите руку для тестирования",
             (cx + 24, cy + 20), _F_LG_B, _WHITE)

        # Кнопка правой руки
        btn_y = cy + 76
        _put(d, "R", (cx + 56, btn_y), _F_XXL, _GREEN)
        _put(d, "Правая рука", (cx + 120, btn_y + 12), _F_LG, _GREEN)

        # Разделитель
        _put(d, "—", (cx + cw // 2 - 8, btn_y + 18), _F_LG, _GRAY)

        # Кнопка левой руки
        _put(d, "L", (cx + cw // 2 + 30, btn_y), _F_XXL, _GREEN)
        _put(d, "Левая рука", (cx + cw // 2 + 94, btn_y + 12), _F_LG, _GREEN)

        _put(d, "Поместите руку перед камерой и нажмите R или L",
             (cx + 24, cy + 216), _F_SM, _GRAY)
        _put(d, "Q — выход", (cx + 24, cy + 248), _F_SM, _GRAY)

        img[:] = _pil_to_bgr(pil)
        return img

    # ── Экран калибровки ──────────────────────────────────────────────────────

    def draw_calibration(self, frame_bgr: np.ndarray,
                         tracking: TrackingFrame,
                         elapsed: float,
                         duration: float) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Шаг 1 из 2 — Калибровка")

        ratio = min(1.0, elapsed / duration) if duration > 0 else 0.0
        color = _GREEN_BGR if tracking.is_valid else _bgr(_YELLOW)
        _circle_progress(img, W // 2, H // 2, 70, ratio, color, thick=7)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        pct = int(ratio * 100)
        text = f"{pct}%"
        bbox = d.textbbox((0, 0), text, font=_F_XL_B)
        tw   = bbox[2] - bbox[0]
        _put(d, text, (W // 2 - tw // 2, H // 2 - 22), _F_XL_B,
             _GREEN if tracking.is_valid else _YELLOW)

        img[:] = _pil_to_bgr(pil)

        # Нижняя панель
        _panel(img, 0, H - 120, W, H, alpha=0.88, bg=_BG_BGR)
        pil2 = _bgr_to_pil(img)
        d2   = ImageDraw.Draw(pil2)

        status_c = _GREEN if tracking.is_valid else _YELLOW
        status_t = ("Рука обнаружена — держите открытую ладонь неподвижно"
                    if tracking.is_valid
                    else "Поместите открытую ладонь в центр кадра")
        _put(d2, status_t, (24, H - 100), _F_MD, status_c)
        _put(d2, "Калибровка задаёт индивидуальный размер кисти",
             (24, H - 64), _F_SM, _GRAY)

        img[:] = _pil_to_bgr(pil2)
        _progress_bar(img, 24, H - 36, W - 48, 16, ratio,
                      fg=_GREEN_BGR if tracking.is_valid else _bgr(_YELLOW))
        _tracking_dot(img, W - 34, 34, tracking.is_valid)
        return img

    # ── Экран подготовки (перед каждым заданием) ──────────────────────────────

    def draw_prepare(self, frame_bgr: np.ndarray,
                     exercise: BaseExercise,
                     exercise_num: int,
                     total_exercises: int) -> np.ndarray:
        """Показывает инструкцию и обратный отсчёт перед стартом задания."""
        from src.exercises.base import PREPARE_SEC
        img = self._base(frame_bgr)
        W, H = self.w, self.h

        _panel(img, 0, 0, W, 76, alpha=0.90, bg=_BG_BGR)
        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)
        _put(d, f"Задание {exercise_num} из {total_exercises}",
             (22, 4), _F_SM, _GRAY)
        label = GESTURE_LABELS.get(exercise.exercise_id, exercise.exercise_id)
        _put(d, label, (22, 28), _F_LG_B, _ACCENT)
        img[:] = _pil_to_bgr(pil)

        # Большая карточка с инструкцией по центру
        cw, ch = min(800, W - 80), 220
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.90)

        pil2 = _bgr_to_pil(img)
        d2   = ImageDraw.Draw(pil2)

        _put(d2, "Подготовьтесь:", (cx + 28, cy + 22), _F_MD, _GRAY)

        # Инструкция — переносим по словам если длинная
        words = exercise.instruction.split()
        lines, line = [], ""
        for w in words:
            test = (line + " " + w).strip()
            bbox = d2.textbbox((0, 0), test, font=_F_LG_B)
            if bbox[2] - bbox[0] > cw - 60 and line:
                lines.append(line)
                line = w
            else:
                line = test
        if line:
            lines.append(line)

        for i, ln in enumerate(lines):
            _put(d2, ln, (cx + 28, cy + 62 + i * 38), _F_LG_B, _WHITE)

        # Обратный отсчёт / иконка готовности справа
        elapsed = exercise.prepare_elapsed()
        remaining = max(0.0, PREPARE_SEC - elapsed)
        if remaining > 0:
            countdown = str(int(remaining) + 1)
            clr = _GREEN if remaining > 1.5 else _YELLOW if remaining > 0.5 else _RED
        else:
            countdown = "►"
            clr = _GREEN
        bbox = d2.textbbox((0, 0), countdown, font=_F_XXL)
        tw   = bbox[2] - bbox[0]
        _put(d2, countdown,
             (cx + cw - tw - 36, cy + ch // 2 - 36), _F_XXL, clr)

        # После минимальной паузы — ждём подтверждения
        if elapsed >= PREPARE_SEC:
            _put(d2, "Нажмите  ПРОБЕЛ  или  ENTER  для старта",
                 (cx + 28, cy + ch - 32), _F_MD_B, _GREEN)
        else:
            _put(d2, "Начнётся через...",
                 (cx + 28, cy + ch - 32), _F_SM, _GRAY)

        img[:] = _pil_to_bgr(pil2)

        # Прогресс-бар внизу (только во время минимальной паузы)
        ratio = min(1.0, elapsed / PREPARE_SEC)
        bar_color = _ACCENT_BGR if elapsed < PREPARE_SEC else _bgr(_GREEN)
        _progress_bar(img, 24, H - 20, W - 48, 10, ratio, fg=bar_color)
        return img

    # ── Экран задания ─────────────────────────────────────────────────────────

    def draw_exercise(self, frame_bgr: np.ndarray,
                      exercise: BaseExercise,
                      tracking: TrackingFrame,
                      exercise_num: int,
                      total_exercises: int) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h

        # Шапка
        _panel(img, 0, 0, W, 76, alpha=0.90, bg=_BG_BGR)
        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        label = GESTURE_LABELS.get(exercise.exercise_id, exercise.exercise_id)
        _put(d, f"Задание {exercise_num} из {total_exercises}",
             (22, 4), _F_SM, _GRAY)
        _put(d, label, (22, 28), _F_LG_B, _ACCENT)

        # Точки прогресса
        dot_r   = 7
        dot_gap = 22
        dots_x  = W - total_exercises * dot_gap - 24
        for i in range(total_exercises):
            dx = dots_x + i * dot_gap + dot_r
            if i < exercise_num - 1:
                fill = _GREEN
            elif i == exercise_num - 1:
                fill = _ACCENT
            else:
                fill = _BORDER
            d.ellipse((dx - dot_r, 46 - dot_r, dx + dot_r, 46 + dot_r),
                      fill=fill)

        img[:] = _pil_to_bgr(pil)
        _tracking_dot(img, W - 34, 34, tracking.is_valid)

        # Нижняя панель инструкции
        _panel(img, 0, H - 130, W, H, alpha=0.90, bg=_BG_BGR)
        pil2 = _bgr_to_pil(img)
        d2   = ImageDraw.Draw(pil2)

        _put(d2, exercise.instruction, (24, H - 116), _F_MD, _WHITE)

        hold  = exercise.current_hold()
        req   = exercise.required_hold_sec
        ratio = min(1.0, hold / req) if req > 0 else 0.0
        bar_c = _GREEN if ratio >= 1.0 else _ACCENT

        # Подсказка позиционирования или счётчик удержания
        hint = exercise.position_hint()
        if hint:
            _put(d2, f"⚠  {hint}", (24, H - 76), _F_SM, _YELLOW)
        else:
            bar_label = "Выполнено!" if ratio >= 1.0 else f"{hold:.1f} с / {req:.0f} с"
            _put(d2, "Удержание:", (24, H - 76), _F_SM, _GRAY)
            _put(d2, bar_label,   (130, H - 76), _F_SM, bar_c)

        _put(d2, "ПРОБЕЛ — следующее   S — пропустить   Q — выйти",
             (24, H - 18), _F_SM, _GRAY)

        img[:] = _pil_to_bgr(pil2)
        _progress_bar(img, 24, H - 52, W - 48, 20, ratio, fg=_bgr(bar_c))

        if isinstance(exercise, ZoneMovementExercise):
            self._draw_zones(img, exercise)

        # Для упражнений на ориентацию ладони — показываем текущее состояние
        from src.exercises.exercises import PalmFacingExercise, BackFacingExercise
        from src.processing.metrics import palm_facing_camera
        if isinstance(exercise, (PalmFacingExercise, BackFacingExercise)):
            if tracking.is_valid:
                facing = palm_facing_camera(tracking)
                label_now = "Внутренняя сторона (ладонь)" if facing else "Тыльная сторона"
                pose_match = (
                    (isinstance(exercise, PalmFacingExercise) and facing) or
                    (isinstance(exercise, BackFacingExercise) and not facing)
                )
                color_now = _GREEN if pose_match else _RED
                status_icon = "✓" if pose_match else "✗"
                pil3 = _bgr_to_pil(img)
                d3   = ImageDraw.Draw(pil3)
                _put(d3, f"{status_icon}  Сейчас: {label_now}",
                     (W // 2 - 180, H - 160), _F_MD_B, color_now)
                img[:] = _pil_to_bgr(pil3)

        # Дебаг-панель: curl пальцев (правый верхний угол)
        self._draw_curl_debug(img, tracking, exercise)

        return img

    def _draw_curl_debug(self, img: np.ndarray,
                         tracking: TrackingFrame,
                         exercise: BaseExercise):
        """Маленькая панель с curl-значениями для отладки пороговых значений."""
        from src.processing.metrics import all_finger_curls, palm_facing_camera, hand_in_position
        W, H = self.w, self.h
        if not tracking.is_valid:
            return

        pw = exercise.calibration.palm_width
        curls = all_finger_curls(tracking, pw)
        names = ["Ук", "Ср", "Без", "Миз"]

        panel_w, panel_h = 130, 108
        px, py = W - panel_w - 10, 80
        _panel(img, px, py, px + panel_w, py + panel_h, alpha=0.80)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)
        _put(d, "curl пальцев", (px + 6, py + 4), _F_SM, _GRAY)
        img[:] = _pil_to_bgr(pil)

        for i, (name, curl) in enumerate(zip(names, curls)):
            bar_y = py + 26 + i * 20
            bar_x = px + 6
            bar_w = panel_w - 12
            # фон
            _progress_bar(img, bar_x, bar_y, bar_w, 14, curl,
                          fg=_RED_BGR if curl > 0.5 else _GREEN_BGR)
            pil2 = _bgr_to_pil(img)
            d2   = ImageDraw.Draw(pil2)
            _put(d2, f"{name} {curl:.2f}", (bar_x + 2, bar_y - 1), _F_SM, _WHITE)
            img[:] = _pil_to_bgr(pil2)

        # Статус позиционирования
        in_pos, hint = hand_in_position(tracking)
        pos_color = _GREEN if in_pos else _YELLOW
        pil3 = _bgr_to_pil(img)
        d3 = ImageDraw.Draw(pil3)
        pos_text = "✓ позиция" if in_pos else "⚠ " + hint[:12]
        _put(d3, pos_text, (px + 6, py + panel_h - 18), _F_SM, pos_color)
        img[:] = _pil_to_bgr(pil3)

    def _draw_zones(self, img: np.ndarray, ex: ZoneMovementExercise):
        W, H   = self.w, self.h
        current = ex.current_zone()
        for i, (zx, zy) in enumerate(ZONES):
            cx = int(zx * W)
            cy = int(zy * H)
            r  = int(ZONE_RADIUS * min(W, H))
            if i < current:
                color = _GREEN_BGR;  thick = 2
            elif i == current:
                color = _ACCENT_BGR; thick = 3
            else:
                color = _BORDER_BGR; thick = 1
            if i == current:
                overlay = img.copy()
                cv2.circle(overlay, (cx, cy), r, color, -1)
                cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.circle(img, (cx, cy), r, color, thick, cv2.LINE_AA)
            pil = _bgr_to_pil(img)
            d   = ImageDraw.Draw(pil)
            _put(d, str(i + 1), (cx - 8, cy - 14), _F_MD_B,
                 _GREEN if i < current else _ACCENT if i == current else _GRAY)
            img[:] = _pil_to_bgr(pil)

    # ── Экран итогов ──────────────────────────────────────────────────────────

    def draw_summary(self, frame_bgr: np.ndarray,
                     summary: TestSummary) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h

        pw  = min(920, W - 40)
        ph  = min(640, H - 40)
        px  = (W - pw) // 2
        py  = (H - ph) // 2
        _panel(img, px, py, px + pw, py + ph, alpha=0.93)

        score = summary.total_score
        if score >= 80:   score_c = _GREEN
        elif score >= 60: score_c = _YELLOW
        elif score >= 40: score_c = _ORANGE
        else:             score_c = _RED

        # Кольцо балла
        ring_cx = px + 80
        ring_cy = py + 98
        _circle_progress(img, ring_cx, ring_cy, 58,
                         score / 100, _bgr(score_c), thick=7)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        # Заголовок
        _put(d, "Результаты тестирования",
             (px + 24, py + 16), _F_XL_B, _WHITE)

        # Балл в центре кольца
        sc_txt = str(score)
        bbox   = d.textbbox((0, 0), sc_txt, font=_F_XL_B)
        tw     = bbox[2] - bbox[0]
        _put(d, sc_txt,
             (ring_cx - tw // 2, ring_cy - 24), _F_XL_B, score_c)
        _put(d, "/ 100",
             (ring_cx - 20, ring_cy + 18), _F_SM, _GRAY)

        # Рекомендация
        _put(d, "Рекомендация:",
             (px + 164, py + 62), _F_SM, _GRAY)
        _put(d, summary.recommendation.label,
             (px + 164, py + 88), _F_LG_B, score_c)

        # Блочные баллы — 2 колонки
        bs = summary.block_scores
        blocks = [
            ("Качество трекинга",   bs.tracking_quality, 20),
            ("Открытая ладонь",     bs.open_palm,        10),
            ("Кулак",               bs.fist,             15),
            ("Щипковый захват",     bs.pinch,            15),
            ("Указательный жест",   bs.point_gesture,    10),
            ("Поворот кисти",       bs.wrist_rotation,   10),
            ("Перемещение по зонам",bs.zone_movement,    15),
            ("Удержание",           bs.hold_stability,    5),
        ]
        col_w = (pw - 60) // 2
        bx    = px + 24
        by    = py + 152

        img[:] = _pil_to_bgr(pil)

        for i, (name, val, mx) in enumerate(blocks):
            col = i % 2
            row = i // 2
            x   = bx + col * (col_w + 12)
            y   = by + row * 58

            ratio  = val / mx if mx > 0 else 0
            bar_cv = _GREEN_BGR if ratio >= 0.8 else _bgr(_YELLOW) if ratio >= 0.5 else _RED_BGR

            pil2 = _bgr_to_pil(img)
            d2   = ImageDraw.Draw(pil2)
            _put(d2, f"{name}  {val}/{mx}", (x, y), _F_SM, _WHITE)
            img[:] = _pil_to_bgr(pil2)

            _progress_bar(img, x, y + 22, col_w - 24, 14, ratio, fg=bar_cv)

        # Статусы упражнений
        from src.models import ExerciseStatus
        STATUS_LABEL = {
            ExerciseStatus.DONE:       ("Выполнено",       _GREEN),
            ExerciseStatus.PARTIAL:    ("Частично",        _YELLOW),
            ExerciseStatus.UNRELIABLE: ("Не оценено",      _ORANGE),
            ExerciseStatus.SKIPPED:    ("Пропущено",       _GRAY),
        }
        note_y = py + ph - 110
        pil3 = _bgr_to_pil(img)
        d3   = ImageDraw.Draw(pil3)

        # Рекомендационные заметки (1 строка)
        if summary.recommendation.notes:
            _put(d3, f"• {summary.recommendation.notes[0]}",
                 (px + 24, note_y), _F_SM, _GRAY)
            note_y += 22

        # Краткая строка по каждому упражнению
        ex_results = summary.exercise_results
        cols = 2
        col_w2 = (pw - 60) // cols
        for idx, r in enumerate(ex_results):
            col = idx % cols
            row = idx // cols
            ex = int(idx // cols)
            x_pos = px + 24 + col * (col_w2 + 12)
            y_pos = note_y + row * 20
            if y_pos + 20 > py + ph - 32:
                break
            lbl = GESTURE_LABELS.get(r.exercise_id, r.exercise_id)
            st_text, st_color = STATUS_LABEL.get(r.status, ("?", _GRAY))
            _put(d3, f"{lbl[:22]}  →  {st_text}  {r.score}/{r.max_score}",
                 (x_pos, y_pos), _F_SM,
                 st_color if r.status == ExerciseStatus.DONE else _GRAY)

        _put(d3, "Нажмите любую клавишу для завершения",
             (px + 24, py + ph - 28), _F_SM, _BORDER)
        img[:] = _pil_to_bgr(pil3)
        return img

    # ── Скелет руки ───────────────────────────────────────────────────────────

    def draw_tracking_overlay(self, img: np.ndarray,
                              tracking: TrackingFrame) -> np.ndarray:
        from src.tracking.adapter import HAND_CONNECTIONS
        if not tracking.is_valid or not tracking.landmarks:
            return img
        h, w = img.shape[:2]
        pts  = [(int(p.x * w), int(p.y * h)) for p in tracking.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (80, 220, 80), 1, cv2.LINE_AA)
        for idx, p in enumerate(pts):
            r = 5 if idx == 0 else 3
            cv2.circle(img, p, r, (255, 255, 255), -1)
            cv2.circle(img, p, r, (80, 180, 80), 1)
        return img
