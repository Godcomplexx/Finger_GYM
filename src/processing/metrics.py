from __future__ import annotations
import math
import numpy as np
from src.models import TrackingFrame, Point2D, CalibrationProfile

# ── Индексы ключевых точек MediaPipe ──────────────────────────────────────────
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_TIP = 12
RING_MCP = 13; RING_TIP = 16
PINKY_MCP = 17; PINKY_TIP = 20

LONG_FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
LONG_FINGER_MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Опорные точки ладони для расчёта palmWidth (запястье — основание мизинца)
PALM_BASE_A = WRIST
PALM_BASE_B = PINKY_MCP


def _dist(a: Point2D, b: Point2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


# ── Калибровочные метрики ──────────────────────────────────────────────────────

def compute_palm_width(frame: TrackingFrame) -> float:
    """Расстояние запястье-основание мизинца. Используется как нормировочная база."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return 0.0
    return _dist(frame.landmarks[PALM_BASE_A], frame.landmarks[PALM_BASE_B])


def compute_palm_center(frame: TrackingFrame) -> Point2D:
    """Центр ладони — среднее между запястьем и основаниями 4 пальцев."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return Point2D(0.5, 0.5)
    pts = [frame.landmarks[i] for i in [WRIST] + LONG_FINGER_MCPS]
    return Point2D(
        x=sum(p.x for p in pts) / len(pts),
        y=sum(p.y for p in pts) / len(pts),
    )


# ── Нормированные расстояния ───────────────────────────────────────────────────

def normalized_distance(a: Point2D, b: Point2D, palm_width: float) -> float:
    """normalizedDistance = distance / palmWidth"""
    if palm_width <= 0:
        return 0.0
    return _dist(a, b) / palm_width


def thumb_index_distance(frame: TrackingFrame, palm_width: float) -> float:
    """thumbIndexDistance / palmWidth — признак щипкового захвата."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return 1.0
    return normalized_distance(frame.landmarks[THUMB_TIP], frame.landmarks[INDEX_TIP], palm_width)


def avg_tip_to_palm_distance(frame: TrackingFrame, palm_width: float) -> float:
    """Среднее расстояние кончиков 4 длинных пальцев до центра ладони."""
    if not frame.is_valid or len(frame.landmarks) < 21 or palm_width <= 0:
        return 0.0
    center = compute_palm_center(frame)
    dists = [normalized_distance(frame.landmarks[t], center, palm_width)
             for t in LONG_FINGER_TIPS]
    return sum(dists) / len(dists)


# ── Сгибание пальцев ───────────────────────────────────────────────────────────

def finger_curl(tip: Point2D, mcp: Point2D, palm_width: float) -> float:
    """
    Степень сгибания пальца: 0.0 (разогнут) → 1.0 (согнут).
    Использует нормированное расстояние кончик-основание.
    При разогнутом пальце расстояние велико, при согнутом — мало.
    """
    if palm_width <= 0:
        return 0.0
    d = normalized_distance(tip, mcp, palm_width)
    # В норме при открытой ладони d ≈ 1.2–1.6, при кулаке d ≈ 0.3–0.6
    # Нормируем: curl = 1 - clamp(d / 1.4, 0, 1)
    return max(0.0, min(1.0, 1.0 - d / 1.4))


def all_finger_curls(frame: TrackingFrame, palm_width: float) -> list[float]:
    """Возвращает curl для 4 длинных пальцев [указательный, средний, безымянный, мизинец]."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        finger_curl(frame.landmarks[tip], frame.landmarks[mcp], palm_width)
        for tip, mcp in zip(LONG_FINGER_TIPS, LONG_FINGER_MCPS)
    ]


def index_finger_curl(frame: TrackingFrame, palm_width: float) -> float:
    if not frame.is_valid or len(frame.landmarks) < 21:
        return 0.0
    return finger_curl(frame.landmarks[INDEX_TIP], frame.landmarks[INDEX_MCP], palm_width)


# ── Стабильность и тремор ──────────────────────────────────────────────────────

def hand_jitter(centers: list[Point2D]) -> float:
    """std отклонение координат центра ладони — мера тремора."""
    if len(centers) < 2:
        return 0.0
    xs = [p.x for p in centers]
    ys = [p.y for p in centers]
    return float(np.std(xs) + np.std(ys))


# ── Ориентация ладони ──────────────────────────────────────────────────────────

THUMB_MCP_IDX = 2
THUMB_CMC = 1

def palm_facing_camera(frame: TrackingFrame) -> bool:
    """
    Определяет ориентацию кисти: ладонь к камере или тыл.

    Метод: 3D cross-product нормали ладони по Z-компоненте.
    Используем три точки: WRIST(0), INDEX_MCP(5), PINKY_MCP(17).
    Нормаль n = (index_mcp - wrist) × (pinky_mcp - wrist).
    Если n.z > 0 — ладонь смотрит «на» камеру (отрицательное Z MediaPipe = ближе к камере,
    но знак нормали по XY определяет сторону поверхности).

    MediaPipe задаёт z: запястье = 0, кончики пальцев < 0 (ближе к камере при вытянутой руке).
    При ладони к камере большой палец БЛИЖЕ (z_thumb < z_pinky для правой руки в зеркале).
    Используем разность z больших пальцев как подтверждающий признак.
    """
    if not frame.is_valid or len(frame.landmarks) < 21:
        return False

    wrist     = frame.landmarks[WRIST]
    index_mcp = frame.landmarks[INDEX_MCP]
    pinky_mcp = frame.landmarks[PINKY_MCP]
    thumb_mcp = frame.landmarks[THUMB_MCP_IDX]

    # Векторы от запястья в 3D
    vi = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)
    vp = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)

    # Cross product → нормаль плоскости ладони
    nx = vi[1]*vp[2] - vi[2]*vp[1]
    ny = vi[2]*vp[0] - vi[0]*vp[2]
    nz = vi[0]*vp[1] - vi[1]*vp[0]

    # Z-компонент нормали: знак определяет, куда «смотрит» ладонь
    # Калибруем по метке руки: MediaPipe отдаёт зеркальные координаты
    # при flip(bgr, 1) перед обработкой.
    # После горизонтального flip(bgr, 1) MediaPipe видит зеркальный кадр.
    # Знак nz инвертирован относительно реального мира:
    # nz < 0 → ладонь к камере для "Right" (реальная правая рука)
    # nz > 0 → ладонь к камере для "Left"  (реальная левая рука)
    if frame.hand_label == "Right":
        return nz < 0
    else:
        return nz > 0


# ── valid tracking ratio ───────────────────────────────────────────────────────

def valid_tracking_ratio(frames: list[TrackingFrame]) -> float:
    if not frames:
        return 0.0
    return sum(1 for f in frames if f.is_valid) / len(frames)


# ── Позиционирование руки в кадре ─────────────────────────────────────────────

# Минимальный нормированный размер ладони (palm_width) — рука достаточно близко
MIN_PALM_WIDTH_NORM = 0.08
# Поля: рука не должна касаться края кадра
FRAME_MARGIN = 0.07


def hand_in_position(frame: TrackingFrame) -> tuple[bool, str]:
    """
    Проверяет, находится ли рука в корректной позиции для оценки.
    Возвращает (ok, hint) где hint — текст подсказки если ok=False.

    Критерии:
      1. Запястье не у края кадра по X (руку не нужно держать ровно по центру)
      2. Palm width достаточная — рука не слишком далеко от камеры
      Пальцы могут выходить за верхний край — это нормально при упражнениях.
    """
    if not frame.is_valid or len(frame.landmarks) < 21:
        return False, "Поместите руку перед камерой"

    # Проверяем только запястье и MCP по X — не обрезаем пальцы сверху
    wrist = frame.landmarks[WRIST]
    mcps  = [frame.landmarks[i] for i in [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]]
    key_pts = [wrist] + mcps

    m = FRAME_MARGIN
    xs = [p.x for p in key_pts]
    ys = [p.y for p in key_pts]

    if min(xs) < m or max(xs) > 1 - m:
        return False, "Рука у края — сдвиньте к центру"
    if max(ys) > 1 - m:
        return False, "Рука слишком низко — поднимите"

    pw = compute_palm_width(frame)
    if pw < MIN_PALM_WIDTH_NORM:
        return False, "Поднесите руку ближе к камере"

    return True, ""
