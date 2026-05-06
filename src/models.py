from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class Hand(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class ExerciseStatus(str, Enum):
    DONE = "done"
    PARTIAL = "partial"
    UNRELIABLE = "unreliable"
    SKIPPED = "skipped"


class RecommendationMode(str, Enum):
    STANDARD = "standard"
    ADAPTED = "adapted"
    TRAINING = "training"
    REPEAT = "repeat"
    INDIVIDUAL = "individual"


class QualityCategory(str, Enum):
    GOOD = "good"
    MEDIUM = "medium"
    POOR = "poor"
    UNRELIABLE = "unreliable"


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Point2D:
    x: float  # нормировано [0,1] относительно ширины кадра
    y: float  # нормировано [0,1] относительно высоты кадра
    z: float = 0.0  # глубина от MediaPipe (относительная, < 0 — ближе к камере)

    def distance_to(self, other: "Point2D") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class TrackingFrame:
    timestamp: float                    # время в секундах (time.monotonic())
    landmarks: list[Point2D]            # 21 точка MediaPipe
    is_valid: bool                      # рука распознана корректно
    hand_label: str = "Unknown"         # "Left" / "Right" от MediaPipe
    source: str = "unknown"
    coordinate_system: str = "image_normalized"
    confidence: float | None = None
    framerate: float | None = None


@dataclass
class CalibrationProfile:
    palm_width: float                   # нормированное расстояние запястье-мизинец
    palm_center: Point2D
    base_tip_to_palm: float             # нормированное расстояние кончиков до ладони при открытой ладони
    base_thumb_index: float             # нормированное расстояние большой-указательный при открытой ладони
    is_ready: bool = False


@dataclass
class ExerciseResult:
    exercise_id: str
    status: ExerciseStatus
    score: int                          # 0..max_score
    max_score: int
    hold_time_sec: float
    valid_tracking_ratio: float
    metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class BlockScores:
    tracking_quality: int = 0
    open_palm: int = 0
    fist: int = 0
    pinch: int = 0
    point_gesture: int = 0
    wrist_rotation: int = 0
    zone_movement: int = 0
    hold_stability: int = 0

    def total(self) -> int:
        return (
            self.tracking_quality + self.open_palm + self.fist +
            self.pinch + self.point_gesture + self.wrist_rotation +
            self.zone_movement
        )


@dataclass
class Recommendation:
    mode: RecommendationMode
    label: str
    notes: list[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    event_type: str
    severity: EventSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)


@dataclass
class TestSummary:
    valid_tracking_ratio: float
    block_scores: BlockScores
    total_score: int
    quality_category: QualityCategory
    recommendation: Recommendation
    exercise_results: list[ExerciseResult] = field(default_factory=list)


@dataclass
class TestSession:
    session_id: str
    patient_id: str
    hand: Hand
    started_at: float = field(default_factory=time.time)
    module_version: str = "unknown"
    algorithm_version: str = "unknown"
    model_name: str = "unknown"
    model_sha256: str | None = None
    tracking_source: str = "unknown"
    calibration: Optional[CalibrationProfile] = None
    summary: Optional[TestSummary] = None
    events: list[AuditEvent] = field(default_factory=list)
