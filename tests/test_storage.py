"""Тесты модуля src/storage/session_storage.py"""
import json
import os
import pytest
from src.models import (
    Hand, TestSession, CalibrationProfile, Point2D,
    TestSummary, BlockScores, Recommendation, RecommendationMode,
    ExerciseResult, ExerciseStatus,
)
from src.storage.session_storage import save_session


def _make_session(with_summary: bool = True) -> TestSession:
    calib = CalibrationProfile(
        palm_width=0.15,
        palm_center=Point2D(0.5, 0.5),
        base_tip_to_palm=1.2,
        base_thumb_index=0.8,
        is_ready=True,
    )
    summary = None
    if with_summary:
        results = [
            ExerciseResult("open_palm", ExerciseStatus.DONE,
                           8, 10, 2.1, 0.92),
            ExerciseResult("fist", ExerciseStatus.PARTIAL,
                           7, 15, 1.2, 0.78),
        ]
        summary = TestSummary(
            valid_tracking_ratio=0.85,
            block_scores=BlockScores(
                tracking_quality=17, open_palm=8, fist=7,
                pinch=0, point_gesture=0, wrist_rotation=0,
                zone_movement=0, hold_stability=0,
            ),
            total_score=32,
            recommendation=Recommendation(
                mode=RecommendationMode.TRAINING,
                label="Тренировочный режим",
                notes=["Готовность ограничена"],
            ),
            exercise_results=results,
        )
    return TestSession(
        session_id="test-save-001",
        patient_id="patient-test",
        hand=Hand.RIGHT,
        calibration=calib,
        summary=summary,
    )


class TestSaveSession:
    def test_file_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        session = _make_session()
        path = save_session(session)
        assert os.path.exists(path)

    def test_json_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        path = save_session(_make_session())
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_required_fields_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        path = save_session(_make_session())
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert d["sessionId"] == "test-save-001"
        assert d["patientId"] == "patient-test"
        assert d["hand"] == "right"
        assert "startedAt" in d
        assert "exercises" in d
        assert "totalScore" in d
        assert "recommendation" in d

    def test_exercises_serialized(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        path = save_session(_make_session())
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert len(d["exercises"]) == 2
        ex = d["exercises"][0]
        assert ex["id"] == "open_palm"
        assert ex["status"] == "done"
        assert ex["score"] == 8

    def test_total_score_correct(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        path = save_session(_make_session())
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert d["totalScore"] == 32

    def test_recommendation_mode_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        path = save_session(_make_session())
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert d["recommendation"]["mode"] == "training"

    def test_session_without_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        path = save_session(_make_session(with_summary=False))
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert d["totalScore"] is None
        assert d["recommendation"] is None

    def test_calibration_saved(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.storage.session_storage.SESSIONS_DIR", str(tmp_path)
        )
        path = save_session(_make_session())
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert d["calibration"]["palmWidth"] == 0.15
        assert d["calibration"]["palmCenter"]["x"] == 0.5
