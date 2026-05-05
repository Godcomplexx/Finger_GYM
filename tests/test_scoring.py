"""Тесты модуля src/scoring/engine.py"""
import pytest
from src.models import ExerciseResult, ExerciseStatus, RecommendationMode
from src.scoring.engine import (
    compute_block_scores, compute_valid_tracking_ratio,
    make_recommendation, build_summary,
)


# ── Хелперы ───────────────────────────────────────────────────────────────────

def _result(exercise_id: str, score: int, max_score: int,
            vtr: float = 1.0,
            status: ExerciseStatus = ExerciseStatus.DONE) -> ExerciseResult:
    return ExerciseResult(
        exercise_id=exercise_id,
        status=status,
        score=score,
        max_score=max_score,
        hold_time_sec=2.0,
        valid_tracking_ratio=vtr,
    )


def _full_results(vtr: float = 1.0) -> list[ExerciseResult]:
    """Набор результатов с максимальными баллами по всем заданиям."""
    return [
        _result("open_palm",     10, 10, vtr),
        _result("fist",          15, 15, vtr),
        _result("pinch",         15, 15, vtr),
        _result("point_gesture", 10, 10, vtr),
        _result("palm_facing",   10, 10, vtr),
        _result("back_facing",   10, 10, vtr),
        _result("zone_movement", 15, 15, vtr),
        _result("hold_still",     5,  5, vtr),
    ]


# ── Тесты compute_valid_tracking_ratio ────────────────────────────────────────

class TestComputeVTR:
    def test_empty(self):
        assert compute_valid_tracking_ratio([]) == 0.0

    def test_all_perfect(self):
        r = compute_valid_tracking_ratio(_full_results(vtr=1.0))
        assert r == 1.0

    def test_partial(self):
        results = [_result("open_palm", 5, 10, vtr=0.7),
                   _result("fist",      5, 15, vtr=0.9)]
        r = compute_valid_tracking_ratio(results)
        assert abs(r - 0.8) < 1e-6


# ── Тесты compute_block_scores ────────────────────────────────────────────────

class TestComputeBlockScores:
    def test_max_score_with_perfect_results(self):
        bs = compute_block_scores(_full_results(vtr=1.0))
        # tracking_quality = round(1.0 * 20) = 20
        assert bs.tracking_quality == 20
        assert bs.open_palm == 10
        assert bs.fist == 15
        assert bs.pinch == 15
        assert bs.point_gesture == 10
        assert bs.zone_movement == 15
        assert bs.hold_stability == 5

    def test_wrist_rotation_averages_palm_and_back(self):
        results = [
            _result("palm_facing",  8, 10, vtr=1.0),
            _result("back_facing",  6, 10, vtr=1.0),
        ]
        bs = compute_block_scores(results)
        # (8/10 + 6/10) / 2 * 10 = 7
        assert bs.wrist_rotation == 7

    def test_total_max_is_100(self):
        bs = compute_block_scores(_full_results(vtr=1.0))
        assert bs.total() == 100

    def test_zero_scores_with_zero_results(self):
        results = [_result("open_palm", 0, 10, vtr=0.0,
                           status=ExerciseStatus.UNRELIABLE)]
        bs = compute_block_scores(results)
        assert bs.open_palm == 0

    def test_missing_exercise_returns_zero(self):
        bs = compute_block_scores([])
        assert bs.open_palm == 0
        assert bs.fist == 0


# ── Тесты make_recommendation ─────────────────────────────────────────────────

class TestMakeRecommendation:
    def test_unreliable_tracking(self):
        r = make_recommendation(90, avg_vtr=0.60)
        assert r.mode == RecommendationMode.REPEAT

    def test_standard_80_plus(self):
        r = make_recommendation(85, avg_vtr=0.90)
        assert r.mode == RecommendationMode.STANDARD

    def test_adapted_60_to_79(self):
        r = make_recommendation(70, avg_vtr=0.90)
        assert r.mode == RecommendationMode.ADAPTED

    def test_training_40_to_59(self):
        r = make_recommendation(50, avg_vtr=0.90)
        assert r.mode == RecommendationMode.TRAINING

    def test_individual_below_40(self):
        r = make_recommendation(30, avg_vtr=0.90)
        assert r.mode == RecommendationMode.INDIVIDUAL

    def test_boundary_80_is_standard(self):
        r = make_recommendation(80, avg_vtr=0.90)
        assert r.mode == RecommendationMode.STANDARD

    def test_boundary_60_is_adapted(self):
        r = make_recommendation(60, avg_vtr=0.90)
        assert r.mode == RecommendationMode.ADAPTED

    def test_boundary_40_is_training(self):
        r = make_recommendation(40, avg_vtr=0.90)
        assert r.mode == RecommendationMode.TRAINING

    def test_vtr_threshold_exactly_065(self):
        # 0.65 — граница: равно или выше → не REPEAT
        r = make_recommendation(80, avg_vtr=0.65)
        assert r.mode == RecommendationMode.STANDARD

    def test_recommendation_has_notes(self):
        r = make_recommendation(85, avg_vtr=0.90)
        assert len(r.notes) > 0
        assert r.label != ""


# ── Тесты build_summary ───────────────────────────────────────────────────────

class TestBuildSummary:
    def test_full_perfect_summary(self):
        s = build_summary(_full_results(vtr=1.0))
        assert s.total_score == 100
        assert s.recommendation.mode == RecommendationMode.STANDARD

    def test_zero_tracking_unreliable(self):
        results = [_result(eid, 0, mx, vtr=0.0, status=ExerciseStatus.UNRELIABLE)
                   for eid, mx in [
                       ("open_palm", 10), ("fist", 15), ("pinch", 15),
                       ("point_gesture", 10), ("palm_facing", 10),
                       ("back_facing", 10), ("zone_movement", 15), ("hold_still", 5)
                   ]]
        s = build_summary(results)
        assert s.recommendation.mode == RecommendationMode.REPEAT

    def test_exercise_results_preserved(self):
        results = _full_results()
        s = build_summary(results)
        assert len(s.exercise_results) == 8

    def test_valid_tracking_ratio_stored(self):
        results = _full_results(vtr=0.88)
        s = build_summary(results)
        assert abs(s.valid_tracking_ratio - 0.88) < 0.01
