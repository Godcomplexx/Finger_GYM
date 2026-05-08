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

# ── Шрифты ───────────────────────────────────────────────────────────────────
import sys as _sys
import os as _os

def _find_font(bold: bool = False) -> str | None:
    """Ищет шрифт с поддержкой кириллицы на Windows/macOS/Linux."""
    if _sys.platform == "win32":
        candidates = [
            r"C:\Windows\Fonts\segoeui.ttf" if not bold else r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arial.ttf"   if not bold else r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
        ]
    elif _sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    for path in candidates:
        if _os.path.exists(path):
            return path
    return None

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = _find_font(bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
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

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


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


def _button(img: np.ndarray, rect: tuple[int, int, int, int], label: str,
            active: bool = False):
    x1, y1, x2, y2 = rect
    bg = _ACCENT_BGR if active else _PANEL_BGR
    border = _GREEN_BGR if active else _BORDER_BGR
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 2 if active else 1, cv2.LINE_AA)
    pil = _bgr_to_pil(img)
    d = ImageDraw.Draw(pil)
    bbox = d.textbbox((0, 0), label, font=_F_MD_B)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    color = _WHITE if active else _ACCENT
    _put(d, label, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 - th) // 2 - 2), _F_MD_B, color)
    img[:] = _pil_to_bgr(pil)

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


def _draw_hold_area_bounds(img: np.ndarray, panel_top: int) -> None:
    H, W = img.shape[:2]
    margin_x = max(28, int(W * 0.12))
    top = 92
    bottom = max(top + 120, panel_top - 24)
    left = margin_x
    right = W - margin_x
    corner = 42
    color = _BORDER_BGR
    thick = 2
    for sx, sy, dx, dy in [
        (left, top, +1, +1), (right, top, -1, +1),
        (left, bottom, +1, -1), (right, bottom, -1, -1),
    ]:
        cv2.line(img, (sx, sy), (sx + dx * corner, sy), color, thick, cv2.LINE_AA)
        cv2.line(img, (sx, sy), (sx, sy + dy * corner), color, thick, cv2.LINE_AA)

def _darken(bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    dark = np.zeros_like(bgr)
    cv2.addWeighted(bgr, 1 - alpha, dark, alpha, 0, bgr)
    return bgr


# ── Главный рендерер ──────────────────────────────────────────────────────────

class Renderer:

    def __init__(self, width: int = 1280, height: int = 720, debug: bool = False):
        self.w = width
        self.h = height
        self.debug = debug

    def _base(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.w, self.h))
        return _darken(img, 0.42)

    # ── Шапка ─────────────────────────────────────────────────────────────────

    def _draw_header(self, img: np.ndarray, subtitle: str = ""):
        _panel(img, 0, 0, self.w, 84, alpha=0.90, bg=_BG_BGR)
        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)
        _put(d, "Finger Gym", (22, 6),  _F_XL_B, _ACCENT)
        if subtitle:
            _put(d, subtitle, (22, 56), _F_SM, _GRAY)
        img[:] = _pil_to_bgr(pil)

    def draw_error_message(self, frame_bgr: np.ndarray, title: str, message: str) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Сообщение")

        cw, ch = 760, 300
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.92)

        pil = _bgr_to_pil(img)
        d = ImageDraw.Draw(pil)
        _put(d, title, (cx + 28, cy + 24), _F_LG_B, _RED)

        y = cy + 82
        for line in message.splitlines():
            _put(d, line, (cx + 28, y), _F_SM, _WHITE)
            y += 28

        _put(d, "Нажмите любую клавишу, чтобы вернуться к выбору.", (cx + 28, cy + 244), _F_SM, _GRAY)

        img[:] = _pil_to_bgr(pil)
        return img

    def draw_hand_select(
        self,
        frame_bgr: np.ndarray,
        hover_key=None,
        dwell_ratio: float = 0.0,
        pointer: tuple[int, int] | None = None,
    ) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Модуль тестирования мелкой моторики кисти")

        cw = min(1200, W - 60)
        ch = min(420, H - 120)
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.88)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        _put(d, "Выберите руку для тестирования",
             (cx + 24, cy + 20), _F_LG_B, _WHITE)
        _put(d, "Наведите палец на кнопку и подержите",
             (cx + 24, cy + 58), _F_MD, _GRAY)

        img[:] = _pil_to_bgr(pil)

        gap = 70
        btn_top = cy + 118
        btn_bottom = cy + ch - 34
        btn_w = (cw - 56 - gap) // 2
        left_x1 = cx + 28
        right_x1 = left_x1 + btn_w + gap
        buttons = {
            "left": (left_x1, btn_top, left_x1 + btn_w, btn_bottom),
            "right":  (right_x1, btn_top, right_x1 + btn_w, btn_bottom),
        }
        hover_value = getattr(hover_key, "value", hover_key)
        for key, rect in buttons.items():
            border = _GREEN_BGR if key == hover_value else _BORDER_BGR
            thickness = 4 if key == hover_value else 1
            cv2.rectangle(img, rect[:2], rect[2:], border, thickness, cv2.LINE_AA)
            if key == hover_value and dwell_ratio > 0:
                x1, _, x2, y2 = rect
                _progress_bar(img, x1 + 18, y2 - 34, x2 - x1 - 36, 16, dwell_ratio, _GREEN_BGR)

        pil_btn = _bgr_to_pil(img)
        d_btn = ImageDraw.Draw(pil_btn)
        for key, text in (("left", "Левая рука"), ("right", "Правая рука")):
            x1, y1, x2, y2 = buttons[key]
            bbox = d_btn.textbbox((0, 0), text, font=_F_XL_B)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            _put(
                d_btn,
                text,
                (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 - th) // 2 - 4),
                _F_XL_B,
                _GREEN,
            )
        img[:] = _pil_to_bgr(pil_btn)

        if pointer is not None:
            cv2.circle(img, pointer, 16, _ACCENT_BGR, 2, cv2.LINE_AA)
            cv2.circle(img, pointer, 5,  _WHITE_BGR, -1, cv2.LINE_AA)
        return img

    # ── Экран позиционирования руки ───────────────────────────────────────────

    def draw_positioning_guide(
        self,
        frame_bgr: np.ndarray,
        tracking: TrackingFrame,
        distance_status: str,     # "far" | "ok" | "close"
        held_sec: float,
        required_sec: float,
    ) -> np.ndarray:
        """Показывает инструкцию по расстоянию и положению руки перед калибровкой."""
        img = self._base(frame_bgr)
        W, H = self.w, self.h

        self._draw_header(img, "Шаг 1 из 2 — Расположите руку перед камерой")

        # ── Целевая зона: прямоугольник в центре кадра ────────────────────────
        zone_w = int(W * 0.46)
        zone_h = int(H * 0.58)
        zx1    = (W - zone_w) // 2
        zy1    = int(H * 0.14)
        zx2    = zx1 + zone_w
        zy2    = zy1 + zone_h

        if distance_status == "ok":
            zone_color = _GREEN_BGR
        elif distance_status == "far":
            zone_color = _bgr(_YELLOW)
        else:
            zone_color = _RED_BGR

        # Угловые маркеры вместо сплошного прямоугольника
        corner = 32
        thick  = 3
        for (sx, sy, dx, dy) in [
            (zx1, zy1, +1, +1), (zx2, zy1, -1, +1),
            (zx1, zy2, +1, -1), (zx2, zy2, -1, -1),
        ]:
            cv2.line(img, (sx, sy), (sx + dx * corner, sy), zone_color, thick, cv2.LINE_AA)
            cv2.line(img, (sx, sy), (sx, sy + dy * corner), zone_color, thick, cv2.LINE_AA)

        # ── Нижняя панель с инструкцией ───────────────────────────────────────
        _panel(img, 0, H - 180, W, H, alpha=0.92, bg=_BG_BGR)
        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        # Статус и цвет
        if not tracking.is_valid:
            status_text = "Руку не видно — поднесите ладонь к камере"
            status_color = _GRAY
        elif distance_status == "far":
            status_text = "Рука слишком далеко — придвиньтесь ближе к камере"
            status_color = _YELLOW
        elif distance_status == "close":
            status_text = "Рука слишком близко — отодвиньтесь немного"
            status_color = _RED
        else:
            status_text = "Отличное положение! Держите руку неподвижно..."
            status_color = _GREEN

        _put(d, status_text, (24, H - 164), _F_LG_B, status_color)
        _put(d, "Поместите открытую ладонь в рамку. Тест начнётся автоматически.",
             (24, H - 118), _F_MD, _GRAY)

        # Иконки расстояния (близко / хорошо / далеко)
        icon_y = H - 80
        segments = [
            ("далеко",  distance_status == "far",   W // 2 - 200),
            ("хорошо",  distance_status == "ok",    W // 2 - 40),
            ("близко",  distance_status == "close", W // 2 + 120),
        ]
        for label, active, ix in segments:
            col = _GREEN if (active and label == "хорошо") else _YELLOW if (active and label == "далеко") else _RED if active else _BORDER
            _put(d, ("●  " if active else "○  ") + label, (ix, icon_y), _F_MD_B if active else _F_MD, col)

        img[:] = _pil_to_bgr(pil)

        # Прогресс-бар удержания
        ratio = min(1.0, held_sec / required_sec) if required_sec > 0 else 0.0
        fg    = _GREEN_BGR if distance_status == "ok" else _BORDER_BGR
        _progress_bar(img, 24, H - 28, W - 48, 16, ratio, fg=fg)
        _tracking_dot(img, W - 34, 34, tracking.is_valid)

        # Скелет руки
        if tracking.is_valid and tracking.landmarks:
            hh, hw = img.shape[:2]
            pts = [(int(p.x * hw), int(p.y * hh)) for p in tracking.landmarks]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(img, pts[a], pts[b], zone_color, 1, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(img, pt, 3, _WHITE_BGR, -1)

        return img

    # ── Экран калибровки ──────────────────────────────────────────────────────

    def draw_calibration(self, frame_bgr: np.ndarray,
                         tracking: TrackingFrame,
                         elapsed: float,
                         duration: float) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Шаг 2 из 2 — Калибровка")

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

    # ── Экран ввода patient ID ────────────────────────────────────────────────

    def draw_patient_id_input(self, frame_bgr: np.ndarray,
                              current_text: str,
                              error: str = "") -> np.ndarray:
        """Экран ввода идентификатора пациента с клавиатуры."""
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Идентификация пациента")

        cw, ch = min(720, W - 80), 280
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.93)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        _put(d, "Введите ID пациента:", (cx + 28, cy + 22), _F_LG_B, _WHITE)
        _put(d, "Используйте клавиатуру. Enter — подтвердить. Backspace — удалить.",
             (cx + 28, cy + 62), _F_SM, _GRAY)

        # Поле ввода
        field_y = cy + 108
        img[:] = _pil_to_bgr(pil)
        _panel(img, cx + 24, field_y, cx + cw - 24, field_y + 52, alpha=0.60, bg=_BG_BGR)
        cv2.rectangle(img, (cx + 24, field_y), (cx + cw - 24, field_y + 52),
                      _ACCENT_BGR, 2, cv2.LINE_AA)

        pil2 = _bgr_to_pil(img)
        d2   = ImageDraw.Draw(pil2)
        display_text = current_text + "│"  # курсор
        _put(d2, display_text, (cx + 36, field_y + 10), _F_LG_B, _ACCENT)

        if error:
            _put(d2, f"[!] {error}", (cx + 28, cy + 176), _F_MD, _RED)

        _put(d2, "Enter — продолжить    Esc — использовать patient-001",
             (cx + 28, cy + ch - 32), _F_SM, _GRAY)

        img[:] = _pil_to_bgr(pil2)
        return img

    # ── Экран сбоя калибровки ─────────────────────────────────────────────────

    def draw_calibration_failed(self, frame_bgr: np.ndarray,
                                valid_ratio: float,
                                attempt: int) -> np.ndarray:
        """Экран плохого освещения / плохой калибровки с предложением повторить."""
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Калибровка не пройдена")

        cw, ch = min(760, W - 80), 320
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.93)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        _put(d, "Недостаточное качество трекинга", (cx + 28, cy + 22), _F_LG_B, _RED)

        tips = [
            f"Качество трекинга: {int(valid_ratio * 100)}%  (нужно ≥ 70%)",
            "● Улучшите освещение — включите свет перед собой",
            "● Держите ладонь открытой, пальцы разведены",
            "● Не двигайте рукой во время калибровки",
            "● Камера должна видеть всю ладонь целиком",
        ]
        for i, tip in enumerate(tips):
            color = _YELLOW if i == 0 else _GRAY
            _put(d, tip, (cx + 28, cy + 72 + i * 34), _F_MD, color)

        _put(d, f"Попытка {attempt}  —  Пробел: повторить    Esc: выйти",
             (cx + 28, cy + ch - 36), _F_MD_B, _ACCENT)

        img[:] = _pil_to_bgr(pil)
        return img

    # ── Экран подготовки (перед каждым заданием) ──────────────────────────────

    def start_button_rect(self) -> tuple[int, int, int, int]:
        W, H = self.w, self.h
        bw, bh = 260, 68
        return (W // 2 - bw // 2, H - 150, W // 2 + bw // 2, H - 150 + bh)

    def draw_prepare(self, frame_bgr: np.ndarray,
                     exercise: BaseExercise,
                     exercise_num: int,
                     total_exercises: int,
                     pointer: tuple[int, int] | None = None,
                     hover_start: bool = False,
                     dwell_ratio: float = 0.0,
                     countdown_remaining: float | None = None) -> np.ndarray:
        """Show centered task instructions, then a 3-second start countdown."""
        img = self._base(frame_bgr)
        W, H = self.w, self.h

        _panel(img, 0, 0, W, 76, alpha=0.90, bg=_BG_BGR)
        pil = _bgr_to_pil(img)
        d = ImageDraw.Draw(pil)
        _put(d, f"Задание {exercise_num} из {total_exercises}", (22, 4), _F_SM, _GRAY)
        label = GESTURE_LABELS.get(exercise.exercise_id, exercise.exercise_id)
        _put(d, label, (22, 28), _F_LG_B, _ACCENT)
        img[:] = _pil_to_bgr(pil)

        cw, ch = min(920, W - 80), 380
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.92)

        pil2 = _bgr_to_pil(img)
        d2 = ImageDraw.Draw(pil2)
        _put(d2, "Описание задания", (cx + 28, cy + 22), _F_MD, _GRAY)
        _put(d2, exercise.instruction, (cx + 28, cy + 58), _F_LG_B, _WHITE)
        detail_y = cy + 108
        for line in getattr(exercise, "details", []):
            _put(d2, line, (cx + 36, detail_y), _F_SM, _GRAY)
            detail_y += 32

        if countdown_remaining is not None:
            countdown = str(int(countdown_remaining) + 1)
            clr = _GREEN if countdown_remaining > 1.5 else _YELLOW if countdown_remaining > 0.5 else _RED
            bbox = d2.textbbox((0, 0), countdown, font=_F_XXL)
            tw = bbox[2] - bbox[0]
            _put(d2, countdown, (cx + cw - tw - 42, cy + ch // 2 - 36), _F_XXL, clr)
            _put(d2, "Верните руку в исходное положение. Задание начнется после отсчета.",
                 (cx + 28, cy + ch - 34), _F_SM, _GRAY)
        else:
            _put(d2, "Прочитайте описание и нажмите Начать.", (cx + 28, cy + ch - 34), _F_SM, _GRAY)
        img[:] = _pil_to_bgr(pil2)

        if countdown_remaining is None:
            rect = self.start_button_rect()
            _button(img, rect, "Начать", active=hover_start)
            if dwell_ratio > 0:
                x1, _, x2, y2 = rect
                _progress_bar(img, x1 + 14, y2 + 10, x2 - x1 - 28, 12, dwell_ratio, _GREEN_BGR)
        else:
            ratio = 1.0 - min(1.0, countdown_remaining / 3.0)
            _progress_bar(img, 24, H - 20, W - 48, 10, ratio, fg=_GREEN_BGR)

        if pointer is not None:
            cv2.circle(img, pointer, 16, _ACCENT_BGR, 2, cv2.LINE_AA)
            cv2.circle(img, pointer, 5, _WHITE_BGR, -1, cv2.LINE_AA)
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
        dots_x  = W - total_exercises * dot_gap - 86
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
        _tracking_dot(img, W - 34, 18, tracking.is_valid)

        hold  = exercise.current_hold()
        req   = exercise.required_hold_sec
        if isinstance(exercise, ZoneMovementExercise):
            zone_total = len(ZONES)
            zone_hit = exercise.zones_hit()
            ratio = zone_hit / zone_total if zone_total else 0.0
        else:
            zone_total = 0
            zone_hit = 0
            duration = getattr(exercise, "max_duration_sec", req)
            elapsed = exercise.elapsed()
            ratio = min(1.0, elapsed / duration) if duration > 0 else 0.0
        bar_c = _GREEN if ratio >= 1.0 else _ACCENT
        done  = ratio >= 1.0

        # Вспышка зелёного при завершении
        if done:
            flash = img.copy()
            flash[:] = _bgr(_GREEN)
            cv2.addWeighted(flash, 0.12, img, 0.88, 0, img)

        panel_top = H - 190
        if not isinstance(exercise, ZoneMovementExercise):
            _draw_hold_area_bounds(img, panel_top)

        # Нижняя панель инструкции
        _panel(img, 0, panel_top, W, H, alpha=0.92, bg=_BG_BGR)
        pil2 = _bgr_to_pil(img)
        d2   = ImageDraw.Draw(pil2)

        _put(d2, exercise.instruction, (24, panel_top + 14), _F_MD_B, _WHITE)

        if isinstance(exercise, ZoneMovementExercise):
            time_text = f"Время не ограничено   Зоны: {zone_hit}/{zone_total}"
            bbox_time = d2.textbbox((0, 0), time_text, font=_F_SM)
            _put(d2, time_text, (W - (bbox_time[2] - bbox_time[0]) - 28, panel_top + 18), _F_SM, _GRAY)
        # Подсказка позиционирования или причина незасчитанной позы
        hint = exercise.position_hint()
        fail_reason = "" if done else exercise.pose_fail_reason(tracking)
        if hint:
            _put(d2, f"[!] {hint}", (24, panel_top + 58), _F_MD, _YELLOW)
        elif fail_reason and tracking.is_valid and not done:
            _put(d2, f"[X] {fail_reason}", (24, panel_top + 58), _F_MD_B, _RED)
        else:
            if done:
                _put(d2, "[OK] Выполнено!", (24, panel_top + 58), _F_LG_B, _GREEN)
            else:
                if isinstance(exercise, ZoneMovementExercise):
                    _put(d2, "Прогресс:", (24, panel_top + 58), _F_MD, _GRAY)
                    _put(d2, f"{zone_hit} / {zone_total} зон", (190, panel_top + 58), _F_MD_B, bar_c)
                else:
                    bar_label = f"{hold:.1f} с  /  {req:.0f} с"
                    _put(d2, "Удержание:", (24, panel_top + 58), _F_MD, _GRAY)
                    _put(d2, bar_label, (190, panel_top + 58), _F_MD_B, bar_c)

        img[:] = _pil_to_bgr(pil2)

        # Крупный прогресс-бар удержания
        _progress_bar(img, 24, H - 72, W - 48, 36, ratio, fg=_bgr(bar_c))

        # Маленький текст под баром
        pil3 = _bgr_to_pil(img)
        d3 = ImageDraw.Draw(pil3)
        _put(d3, "Пробел - повторить   S - пропустить   Esc - выйти",
             (24, H - 26), _F_SM, _BORDER)
        img[:] = _pil_to_bgr(pil3)

        # Пауза при потере руки (grace period)
        if not tracking.is_valid and exercise.is_hand_lost():
            remaining = exercise.grace_remaining()
            pil_p = _bgr_to_pil(img)
            d_p   = ImageDraw.Draw(pil_p)
            pause_txt = f"Пауза: рука потеряна, {remaining:.1f} с"
            bbox = d_p.textbbox((0, 0), pause_txt, font=_F_LG_B)
            tw = bbox[2] - bbox[0]
            _put(d_p, pause_txt, (W // 2 - tw // 2, H // 2 - 24), _F_LG_B, _YELLOW)
            img[:] = _pil_to_bgr(pil_p)

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
                pil3 = _bgr_to_pil(img)
                d3   = ImageDraw.Draw(pil3)
                _put(d3, f"Сейчас: {label_now}",
                     (24, panel_top + 92), _F_MD_B, color_now)
                img[:] = _pil_to_bgr(pil3)

        # Дебаг-панель: curl пальцев (правый верхний угол, только с --debug)
        if self.debug:
            self._draw_curl_debug(img, tracking, exercise)

        return img

    def next_button_rect(self) -> tuple[int, int, int, int]:
        W, H = self.w, self.h
        bw, bh = 240, 64
        return (W // 2 + 16, H - 150, W // 2 + 16 + bw, H - 150 + bh)

    def repeat_button_rect(self) -> tuple[int, int, int, int]:
        W, H = self.w, self.h
        bw, bh = 240, 64
        return (W // 2 - 16 - bw, H - 150, W // 2 - 16, H - 150 + bh)

    def doctor_indication_rects(self) -> dict[str, tuple[int, int, int, int]]:
        W, H = self.w, self.h
        cw = min(900, W - 80)
        bw = (cw - 42) // 2
        bh = 70
        cx = (W - cw) // 2
        top = H // 2 - 44
        return {
            "normal": (cx, top, cx + bw, top + bh),
            "consult": (cx + bw + 42, top, cx + cw, top + bh),
            "repeat": (cx, top + bh + 24, cx + bw, top + bh * 2 + 24),
            "training": (cx + bw + 42, top + bh + 24, cx + cw, top + bh * 2 + 24),
            "skip": (W // 2 - 130, top + bh * 2 + 58, W // 2 + 130, top + bh * 2 + 118),
        }

    def draw_doctor_indication(
        self,
        frame_bgr: np.ndarray,
        summary: TestSummary,
        hover_target: str | None = None,
        dwell_ratio: float = 0.0,
        pointer: tuple[int, int] | None = None,
    ) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "Заключение врача")

        cw, ch = min(980, W - 72), min(520, H - 112)
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.94)

        pil = _bgr_to_pil(img)
        d = ImageDraw.Draw(pil)
        _put(d, "Выберите показание для сохранения в протокол", (cx + 28, cy + 26), _F_LG_B, _WHITE)
        _put(d, f"Итоговый балл: {summary.total_score} / 80", (cx + 28, cy + 72), _F_MD_B, _ACCENT)
        _put(d, "Можно выбрать мышью, рукой или клавишами 1-4. Esc - пропустить.",
             (cx + 28, cy + 106), _F_SM, _GRAY)
        img[:] = _pil_to_bgr(pil)

        labels = {
            "normal": "1  Без доп. показаний",
            "consult": "2  Консультация специалиста",
            "repeat": "3  Повторить тест позже",
            "training": "4  Тренировка кисти",
            "skip": "Пропустить",
        }
        for key, rect in self.doctor_indication_rects().items():
            _button(img, rect, labels[key], active=hover_target == key)
            if hover_target == key and dwell_ratio > 0:
                x1, _, x2, y2 = rect
                _progress_bar(img, x1 + 14, y2 + 8, x2 - x1 - 28, 10, dwell_ratio, _GREEN_BGR)

        if pointer is not None:
            cv2.circle(img, pointer, 16, _ACCENT_BGR, 2, cv2.LINE_AA)
            cv2.circle(img, pointer, 5, _WHITE_BGR, -1, cv2.LINE_AA)
        return img

    def icf_qualifier_rects(self) -> dict[str, tuple[int, int, int, int]]:
        W, H = self.w, self.h
        cw = min(1040, W - 80)
        cx = (W - cw) // 2
        top = H // 2 - 70
        gap = 20
        bw = (cw - gap * 3) // 4
        bh = 72
        return {
            "0": (cx, top, cx + bw, top + bh),
            "1": (cx + (bw + gap), top, cx + (bw + gap) + bw, top + bh),
            "2": (cx + (bw + gap) * 2, top, cx + (bw + gap) * 2 + bw, top + bh),
            "3": (cx + (bw + gap) * 3, top, cx + cw, top + bh),
            "4": (cx, top + bh + 24, cx + bw, top + bh * 2 + 24),
            "8": (cx + (bw + gap), top + bh + 24, cx + (bw + gap) + bw, top + bh * 2 + 24),
            "9": (cx + (bw + gap) * 2, top + bh + 24, cx + (bw + gap) * 2 + bw, top + bh * 2 + 24),
            "skip": (cx + (bw + gap) * 3, top + bh + 24, cx + cw, top + bh * 2 + 24),
        }

    def draw_icf_assessment(
        self,
        frame_bgr: np.ndarray,
        code: str,
        label: str,
        index: int,
        total: int,
        selected: dict[str, int],
        hover_target: str | None = None,
        dwell_ratio: float = 0.0,
        pointer: tuple[int, int] | None = None,
    ) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h
        self._draw_header(img, "МКФ-оценка врача")

        cw, ch = min(1120, W - 72), min(560, H - 112)
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.94)

        pil = _bgr_to_pil(img)
        d = ImageDraw.Draw(pil)
        _put(d, f"Код {index} из {total}: {code}", (cx + 28, cy + 24), _F_LG_B, _ACCENT)
        _put(d, label, (cx + 28, cy + 66), _F_MD_B, _WHITE)
        _put(d, "Выберите квалификатор МКФ. Можно мышью, рукой или клавишами 0-4, 8, 9.",
             (cx + 28, cy + 104), _F_SM, _GRAY)
        _put(d, "0 нет проблемы   1 легкая   2 умеренная   3 тяжелая   4 полная   8 не определено   9 не применимо",
             (cx + 28, cy + 132), _F_SM, _GRAY)

        if selected:
            y = cy + ch - 86
            parts = [f"{k}.{v}" for k, v in selected.items()]
            _put(d, "Уже выбрано: " + ", ".join(parts), (cx + 28, y), _F_SM, _GREEN)
        _put(d, "S - пропустить код   Esc - закончить МКФ-оценку",
             (cx + 28, cy + ch - 44), _F_SM, _BORDER)
        img[:] = _pil_to_bgr(pil)

        labels = {
            "0": "0  Нет",
            "1": "1  Легкая",
            "2": "2  Умеренная",
            "3": "3  Тяжелая",
            "4": "4  Полная",
            "8": "8  Не определено",
            "9": "9  Не применимо",
            "skip": "Пропустить",
        }
        for key, rect in self.icf_qualifier_rects().items():
            _button(img, rect, labels[key], active=hover_target == key)
            if hover_target == key and dwell_ratio > 0:
                x1, _, x2, y2 = rect
                _progress_bar(img, x1 + 14, y2 + 8, x2 - x1 - 28, 10, dwell_ratio, _GREEN_BGR)

        if pointer is not None:
            cv2.circle(img, pointer, 16, _ACCENT_BGR, 2, cv2.LINE_AA)
            cv2.circle(img, pointer, 5, _WHITE_BGR, -1, cv2.LINE_AA)
        return img

    def draw_exercise_result(
        self,
        frame_bgr: np.ndarray,
        exercise: BaseExercise,
        result,
        exercise_num: int,
        total_exercises: int,
        pointer: tuple[int, int] | None = None,
        hover_target: str | None = None,
        dwell_ratio: float = 0.0,
    ) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h

        _panel(img, 0, 0, W, 76, alpha=0.90, bg=_BG_BGR)
        pil = _bgr_to_pil(img)
        d = ImageDraw.Draw(pil)
        _put(d, f"Задание {exercise_num} из {total_exercises}", (22, 4), _F_SM, _GRAY)
        _put(d, exercise.instruction, (22, 28), _F_LG_B, _ACCENT)
        img[:] = _pil_to_bgr(pil)

        cw, ch = min(760, W - 80), 300
        cx = (W - cw) // 2
        cy = (H - ch) // 2
        _panel(img, cx, cy, cx + cw, cy + ch, alpha=0.94)

        score_ratio = result.score / result.max_score if result.max_score else 0.0
        if result.status.value == "unreliable":
            status_text = "Не оценено достоверно"
            status_color = _ORANGE
        elif score_ratio >= 0.95:
            status_text = "Выполнено"
            status_color = _GREEN
        elif score_ratio > 0:
            status_text = "Выполнено частично"
            status_color = _YELLOW
        else:
            status_text = "Не выполнено"
            status_color = _RED

        pil2 = _bgr_to_pil(img)
        d2 = ImageDraw.Draw(pil2)
        _put(d2, status_text, (cx + 28, cy + 24), _F_LG_B, status_color)
        _put(d2, f"Балл: {result.score} / {result.max_score}", (cx + 28, cy + 76), _F_MD_B, _WHITE)
        _put(d2, f"Удержание: {result.hold_time_sec:.1f} с", (cx + 28, cy + 112), _F_MD, _GRAY)
        _put(d2, f"Качество трекинга: {result.valid_tracking_ratio:.2f}", (cx + 28, cy + 144), _F_MD, _GRAY)
        if result.notes:
            _put(d2, result.notes[0], (cx + 28, cy + 182), _F_SM, _YELLOW)
        else:
            _put(d2, "Нажмите Далее, чтобы перейти к следующему заданию.", (cx + 28, cy + 182), _F_SM, _GRAY)
        img[:] = _pil_to_bgr(pil2)

        repeat_rect = self.repeat_button_rect()
        next_rect = self.next_button_rect()
        _button(img, repeat_rect, "Повторить", active=hover_target == "repeat")
        _button(img, next_rect, "Далее", active=hover_target == "next")
        if hover_target in ("repeat", "next") and dwell_ratio > 0:
            rect = repeat_rect if hover_target == "repeat" else next_rect
            x1, _, x2, y2 = rect
            _progress_bar(img, x1 + 14, y2 + 10, x2 - x1 - 28, 12, dwell_ratio, _GREEN_BGR)

        if pointer is not None:
            cv2.circle(img, pointer, 16, _ACCENT_BGR, 2, cv2.LINE_AA)
            cv2.circle(img, pointer, 5, _WHITE_BGR, -1, cv2.LINE_AA)
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
        pos_text = "OK позиция" if in_pos else "! " + hint[:12]
        _put(d3, pos_text, (px + 6, py + panel_h - 18), _F_SM, pos_color)
        img[:] = _pil_to_bgr(pil3)

    def _draw_zones(self, img: np.ndarray, ex: ZoneMovementExercise):
        W, H    = self.w, self.h
        current = ex.current_zone()
        hold_p  = ex.zone_hold_progress()
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

            # Полупрозрачная заливка активной и выполненных зон
            overlay = img.copy()
            fill_alpha = 0.22 if i == current else (0.10 if i < current else 0.0)
            if fill_alpha > 0:
                cv2.circle(overlay, (cx, cy), r, color, -1)
                cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)

            cv2.circle(img, (cx, cy), r, color, thick, cv2.LINE_AA)

            # Дуга прогресса удержания в текущей зоне
            if i == current and hold_p > 0:
                cv2.ellipse(img, (cx, cy), (r, r), -90, 0,
                            int(360 * hold_p), _ACCENT_BGR, 5, cv2.LINE_AA)

            pil = _bgr_to_pil(img)
            d   = ImageDraw.Draw(pil)
            num_txt = str(i + 1)
            bbox = d.textbbox((0, 0), num_txt, font=_F_MD_B)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            num_color = _GREEN if i < current else _ACCENT if i == current else _GRAY
            _put(d, num_txt, (cx - tw // 2, cy - th // 2 - 2), _F_MD_B, num_color)
            img[:] = _pil_to_bgr(pil)

    # ── Экран итогов ──────────────────────────────────────────────────────────

    def draw_summary(self, frame_bgr: np.ndarray,
                     summary: TestSummary,
                     autoclose_sec: float = 0.0) -> np.ndarray:
        img = self._base(frame_bgr)
        W, H = self.w, self.h

        pw  = min(1020, W - 72)
        ph  = min(600, H - 72)
        px  = (W - pw) // 2
        py  = (H - ph) // 2
        _panel(img, px, py, px + pw, py + ph, alpha=0.93)

        score = summary.total_score
        max_total = 80
        if score >= 64:   score_c = _GREEN
        elif score >= 48: score_c = _YELLOW
        elif score >= 32: score_c = _ORANGE
        else:             score_c = _RED

        # Кольцо балла
        ring_cx = px + 78
        ring_cy = py + 136
        ring_r = 44
        _circle_progress(img, ring_cx, ring_cy, ring_r,
                         score / max_total, _bgr(score_c), thick=7)

        pil = _bgr_to_pil(img)
        d   = ImageDraw.Draw(pil)

        # Заголовок
        _put(d, "Результаты тестирования",
            (px + 30, py + 28), _F_LG_B, _WHITE)

        # Балл в центре кольца
        sc_txt = str(score)
        bbox   = d.textbbox((0, 0), sc_txt, font=_F_LG_B)
        tw     = bbox[2] - bbox[0]
        _put(d, sc_txt,
             (ring_cx - tw // 2, ring_cy - 18), _F_LG_B, score_c)

        # Итоговый балл (текст рядом с кольцом)
        _put(d, "Итого",
             (px + 150, py + 112), _F_SM, _GRAY)
        _put(d, f"{score} / {max_total}",
             (px + 150, py + 138), _F_LG_B, score_c)

        if summary.icf_codes:
            _put(d, "МКФ", (px + 360, py + 112), _F_SM, _GRAY)
            icf_x = px + 360
            icf_y = py + 138
            for item in summary.icf_codes[:4]:
                if item.problem_percent is None:
                    value = item.formatted_code
                else:
                    value = f"{item.formatted_code} ({item.problem_percent}%)"
                _put(d, value, (icf_x, icf_y), _F_SM, _WHITE)
                icf_y += 24

        # Блочные баллы — 2 колонки
        bs = summary.block_scores
        blocks = [
            ("Открытая ладонь",     bs.open_palm,        10),
            ("Кулак",               bs.fist,             15),
            ("Щипковый захват",     bs.pinch,            15),
            ("Указательный жест",   bs.point_gesture,    10),
            ("Поворот кисти",       bs.wrist_rotation,   10),
            ("Перемещение по зонам",bs.zone_movement,    15),
            ("Удержание руки",       bs.hold_stability,   5),
        ]
        col_w = (pw - 96) // 2
        bx    = px + 32
        by    = py + 238

        img[:] = _pil_to_bgr(pil)

        for i, (name, val, mx) in enumerate(blocks):
            col = i % 2
            row = i // 2
            x   = bx + col * (col_w + 32)
            y   = by + row * 68

            ratio  = val / mx if mx > 0 else 0
            bar_cv = _GREEN_BGR if ratio >= 0.8 else _bgr(_YELLOW) if ratio >= 0.5 else _RED_BGR

            pil2 = _bgr_to_pil(img)
            d2   = ImageDraw.Draw(pil2)
            _put(d2, f"{name}  {val}/{mx}", (x, y), _F_SM, _WHITE)
            img[:] = _pil_to_bgr(pil2)

            _progress_bar(img, x, y + 30, col_w - 8, 16, ratio, fg=bar_cv)

        pil3 = _bgr_to_pil(img)
        d3   = ImageDraw.Draw(pil3)

        if autoclose_sec > 0:
            secs = int(autoclose_sec) + 1
            close_txt = f"Закроется через {secs} с..."
        else:
            close_txt = "Завершено"
        _put(d3, close_txt, (px + 24, py + ph - 28), _F_SM, _BORDER)
        img[:] = _pil_to_bgr(pil3)

        # Полоска автозакрытия
        if autoclose_sec > 0:
            close_ratio = autoclose_sec / 12.0
            _progress_bar(img, px, py + ph - 6, pw, 6, close_ratio, fg=_ACCENT_BGR)
        return img

    # ── Скелет руки ───────────────────────────────────────────────────────────

    def draw_tracking_overlay(self, img: np.ndarray,
                              tracking: TrackingFrame) -> np.ndarray:
        if not tracking.is_valid or not tracking.landmarks:
            status = "Tracking: no hand"
            color = _RED_BGR
            cv2.rectangle(img, (18, self.h - 46), (390, self.h - 14), _PANEL_BGR, -1)
            cv2.rectangle(img, (18, self.h - 46), (390, self.h - 14), _BORDER_BGR, 1)
            cv2.circle(img, (36, self.h - 30), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                img,
                status,
                (52, self.h - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _WHITE_BGR,
                1,
                cv2.LINE_AA,
            )
            return img
        h, w = img.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in tracking.landmarks]
        for a, b in HAND_CONNECTIONS:
            if a >= len(pts) or b >= len(pts):
                continue
            cv2.line(img, pts[a], pts[b], (80, 220, 80), 1, cv2.LINE_AA)
        for idx, p in enumerate(pts):
            r = 5 if idx == 0 else 3
            cv2.circle(img, p, r, (255, 255, 255), -1)
            cv2.circle(img, p, r, (80, 180, 80), 1)
        return img
