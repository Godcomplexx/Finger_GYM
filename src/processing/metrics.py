from __future__ import annotations
import math
import numpy as np
from src.models import TrackingFrame, Point2D, CalibrationProfile

# ── Индексы ключевых точек MediaPipe ──────────────────────────────────────────
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_MCP = 5;  INDEX_PIP = 6;  INDEX_DIP = 7;  INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13;  RING_PIP = 14;  RING_DIP = 15;  RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

LONG_FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
LONG_FINGER_MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
LONG_FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
LONG_FINGER_DIPS = [INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]

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


def thumb_index_angle_deg(frame: TrackingFrame) -> float:
    """Angle between thumb and index fingertips around the palm center."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return 180.0
    center = compute_palm_center(frame)
    thumb = frame.landmarks[THUMB_TIP]
    index = frame.landmarks[INDEX_TIP]
    vt = (thumb.x - center.x, thumb.y - center.y, thumb.z - center.z)
    vi = (index.x - center.x, index.y - center.y, index.z - center.z)
    mag = math.sqrt(vt[0] ** 2 + vt[1] ** 2 + vt[2] ** 2) * math.sqrt(
        vi[0] ** 2 + vi[1] ** 2 + vi[2] ** 2
    )
    if mag < 1e-9:
        return 0.0
    dot = vt[0] * vi[0] + vt[1] * vi[1] + vt[2] * vi[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


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


def _angle_3pt(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Angle at point b in degrees (2D, using x/y only)."""
    ba_x, ba_y = a.x - b.x, a.y - b.y
    bc_x, bc_y = c.x - b.x, c.y - b.y
    dot = ba_x * bc_x + ba_y * bc_y
    mag = math.hypot(ba_x, ba_y) * math.hypot(bc_x, bc_y)
    if mag < 1e-9:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def finger_curl_angle(mcp: Point2D, pip: Point2D, tip: Point2D) -> float:
    """
    Angle-based curl: 0.0 (straight, ~180°) → 1.0 (fully bent, ~30°).
    Uses MCP→PIP→TIP angle for better depth-independent detection.
    """
    angle = _angle_3pt(mcp, pip, tip)
    # 170° = straight (0 curl), 40° = fully bent (1.0 curl)
    return max(0.0, min(1.0, (170.0 - angle) / 130.0))


def all_finger_curls(frame: TrackingFrame, palm_width: float) -> list[float]:
    """Возвращает curl для 4 длинных пальцев [указательный, средний, безымянный, мизинец].
    Смешивает дистанционный и угловой методы для устойчивости."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return [0.0, 0.0, 0.0, 0.0]
    result = []
    for tip_idx, mcp_idx, pip_idx in zip(LONG_FINGER_TIPS, LONG_FINGER_MCPS, LONG_FINGER_PIPS):
        dist_curl = finger_curl(frame.landmarks[tip_idx], frame.landmarks[mcp_idx], palm_width)
        ang_curl = finger_curl_angle(frame.landmarks[mcp_idx], frame.landmarks[pip_idx], frame.landmarks[tip_idx])
        result.append((dist_curl + ang_curl) / 2.0)
    return result


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


def palm_normal_z_ratio(frame: TrackingFrame) -> float:
    """Signed -1..1 palm normal Z ratio: +1 palm-facing, -1 back-facing, 0 edge-on."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return 0.0

    wrist     = frame.landmarks[WRIST]
    index_mcp = frame.landmarks[INDEX_MCP]
    pinky_mcp = frame.landmarks[PINKY_MCP]

    vi = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)
    vp = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)

    nx = vi[1]*vp[2] - vi[2]*vp[1]
    ny = vi[2]*vp[0] - vi[0]*vp[2]
    nz = vi[0]*vp[1] - vi[1]*vp[0]
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-9:
        return 0.0

    if frame.hand_label == "Right":
        return max(-1.0, min(1.0, -nz / mag))
    return max(-1.0, min(1.0, nz / mag))


def palm_facing_quality(frame: TrackingFrame) -> float:
    """0..1: 1 when palm faces camera, 0 when edge-on or back-facing."""
    return max(0.0, palm_normal_z_ratio(frame))


def back_facing_quality(frame: TrackingFrame) -> float:
    """0..1: 1 when back of hand faces camera, 0 when edge-on or palm-facing."""
    return max(0.0, -palm_normal_z_ratio(frame))


def palm_facing_camera(frame: TrackingFrame) -> bool:
    """True if palm side is closer to facing the camera than the back side."""
    return palm_facing_quality(frame) > 0.0


def finger_spread(frame: TrackingFrame, palm_width: float) -> float:
    """Average normalized distance between adjacent fingertips — proxy for spread."""
    if not frame.is_valid or len(frame.landmarks) < 21 or palm_width <= 0:
        return 0.0
    tips = [frame.landmarks[i] for i in LONG_FINGER_TIPS]
    pairs = [(tips[i], tips[i+1]) for i in range(len(tips) - 1)]
    return sum(_dist(a, b) / palm_width for a, b in pairs) / len(pairs)


def fingers_pointing_up(frame: TrackingFrame) -> bool:
    """True if majority of fingertips are above (lower Y) than their MCPs."""
    if not frame.is_valid or len(frame.landmarks) < 21:
        return False
    up_count = sum(
        1 for tip_idx, mcp_idx in zip(LONG_FINGER_TIPS, LONG_FINGER_MCPS)
        if frame.landmarks[tip_idx].y < frame.landmarks[mcp_idx].y
    )
    return up_count >= 3


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
