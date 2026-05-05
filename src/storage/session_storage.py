from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from dataclasses import asdict
from src.models import TestSession, Hand


SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "sessions")


def _make_serializable(obj):
    """Рекурсивно конвертирует dataclasses и Enum в JSON-сериализуемые типы."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _make_serializable(asdict(obj))
    if hasattr(obj, "value"):   # Enum
        return obj.value
    return obj


def save_session(session: TestSession) -> str:
    """Сохраняет сессию в JSON-файл. Возвращает путь к файлу."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    dt = datetime.fromtimestamp(session.started_at, tz=timezone.utc)
    filename = f"{session.session_id}_{dt.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(SESSIONS_DIR, filename)

    summary = session.summary
    calib = session.calibration

    payload = {
        "sessionId": session.session_id,
        "patientId": session.patient_id,
        "hand": session.hand.value,
        "startedAt": dt.isoformat(),
        "calibration": {
            "palmWidth": round(calib.palm_width, 4) if calib else None,
            "palmCenter": (
                {"x": round(calib.palm_center.x, 4), "y": round(calib.palm_center.y, 4)}
                if calib and calib.palm_center else None
            ),
        } if calib else None,
        "validTrackingRatio": summary.valid_tracking_ratio if summary else None,
        "exercises": [
            {
                "id": r.exercise_id,
                "status": r.status.value,
                "score": r.score,
                "maxScore": r.max_score,
                "holdTimeSec": round(r.hold_time_sec, 2),
                "validTrackingRatio": round(r.valid_tracking_ratio, 3),
                "metrics": r.metrics,
                "notes": r.notes,
            }
            for r in (summary.exercise_results if summary else [])
        ],
        "blockScores": {
            "trackingQuality": summary.block_scores.tracking_quality,
            "openPalm": summary.block_scores.open_palm,
            "fist": summary.block_scores.fist,
            "pinch": summary.block_scores.pinch,
            "pointGesture": summary.block_scores.point_gesture,
            "wristRotation": summary.block_scores.wrist_rotation,
            "zoneMovement": summary.block_scores.zone_movement,
            "holdStability": summary.block_scores.hold_stability,
        } if summary else None,
        "totalScore": summary.total_score if summary else None,
        "recommendation": {
            "mode": summary.recommendation.mode.value,
            "label": summary.recommendation.label,
            "notes": summary.recommendation.notes,
        } if summary else None,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath
