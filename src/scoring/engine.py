from __future__ import annotations
from src.models import (
    ExerciseResult, ExerciseStatus, BlockScores,
    QualityCategory, Recommendation, RecommendationMode, TestSummary,
)
from src.processing.metrics import valid_tracking_ratio

# Максимальные баллы блоков из методики
MAX_TRACKING_QUALITY = 20
MAX_OPEN_PALM = 10
MAX_FIST = 15
MAX_PINCH = 15
MAX_POINT_GESTURE = 10
MAX_WRIST_ROTATION = 10   # среднее palm_facing + back_facing
MAX_ZONE_MOVEMENT = 15
MAX_HOLD_STABILITY = 5

UNRELIABLE_THRESHOLD = 0.65


def make_quality_category(total_score: int, avg_vtr: float) -> QualityCategory:
    if avg_vtr < UNRELIABLE_THRESHOLD:
        return QualityCategory.UNRELIABLE
    if total_score >= 75:
        return QualityCategory.GOOD
    if total_score >= 50:
        return QualityCategory.MEDIUM
    return QualityCategory.POOR


def _tracking_quality_score(results: list[ExerciseResult]) -> int:
    """Блок 'Качество трекинга' (0–20): среднее validTrackingRatio по всем заданиям."""
    if not results:
        return 0
    avg_vtr = sum(r.valid_tracking_ratio for r in results) / len(results)
    return round(avg_vtr * MAX_TRACKING_QUALITY)


def _find(results: list[ExerciseResult], exercise_id: str) -> ExerciseResult | None:
    return next((r for r in results if r.exercise_id == exercise_id), None)


def _wrist_rotation_score(results: list[ExerciseResult]) -> int:
    """Среднее баллов palm_facing и back_facing нормировано к MAX_WRIST_ROTATION."""
    palm = _find(results, "palm_facing")
    back = _find(results, "back_facing")
    total = 0
    count = 0
    for r in [palm, back]:
        if r is not None:
            total += r.score / r.max_score
            count += 1
    if count == 0:
        return 0
    return round((total / count) * MAX_WRIST_ROTATION)


def compute_block_scores(results: list[ExerciseResult]) -> BlockScores:
    def _score(ex_id: str) -> int:
        r = _find(results, ex_id)
        return r.score if r else 0

    return BlockScores(
        tracking_quality=_tracking_quality_score(results),
        open_palm=_score("open_palm"),
        fist=_score("fist"),
        pinch=_score("pinch"),
        point_gesture=_score("point_gesture"),
        wrist_rotation=_wrist_rotation_score(results),
        zone_movement=_score("zone_movement"),
        hold_stability=_score("hold_still"),
    )


def compute_valid_tracking_ratio(results: list[ExerciseResult]) -> float:
    if not results:
        return 0.0
    return sum(r.valid_tracking_ratio for r in results) / len(results)


def make_recommendation(total_score: int, avg_vtr: float) -> Recommendation:
    if avg_vtr < UNRELIABLE_THRESHOLD:
        return Recommendation(
            mode=RecommendationMode.REPEAT,
            label="Повторите тестирование",
            notes=[
                "Результат технически ненадёжен: низкое качество трекинга",
                "Проверьте освещение, положение руки и фон",
            ],
        )
    if total_score >= 80:
        return Recommendation(
            mode=RecommendationMode.STANDARD,
            label="Стандартный VR-сценарий",
            notes=["Функциональная готовность кисти высокая"],
        )
    if total_score >= 60:
        return Recommendation(
            mode=RecommendationMode.ADAPTED,
            label="Адаптированный режим",
            notes=[
                "Функциональная готовность достаточная, но возможны ограничения",
                "Рекомендуются увеличенные зоны взаимодействия и мягкие допуски",
            ],
        )
    if total_score >= 40:
        return Recommendation(
            mode=RecommendationMode.TRAINING,
            label="Тренировочный режим",
            notes=[
                "Готовность ограничена",
                "Рекомендуется тренировочный режим без строгой оценки результата",
            ],
        )
    return Recommendation(
        mode=RecommendationMode.INDIVIDUAL,
        label="Индивидуальная настройка",
        notes=[
            "Полноценное прохождение может быть затруднено",
            "Рекомендуется подготовительный этап или индивидуальная настройка сценария",
        ],
    )


def build_summary(results: list[ExerciseResult]) -> TestSummary:
    block_scores = compute_block_scores(results)
    total = block_scores.total()
    avg_vtr = compute_valid_tracking_ratio(results)
    quality_category = make_quality_category(total, avg_vtr)
    recommendation = make_recommendation(total, avg_vtr)
    return TestSummary(
        valid_tracking_ratio=round(avg_vtr, 3),
        block_scores=block_scores,
        total_score=total,
        quality_category=quality_category,
        recommendation=recommendation,
        exercise_results=results,
    )
