from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from dataclasses import asdict
from src.models import TestSession, Hand
from src.app_info import MODULE_NAME, runtime_root


SESSIONS_DIR = os.path.join(runtime_root(), "sessions")


def safe_patient_id(patient_id: str) -> str:
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in patient_id
    )[:64] or "unknown"


def session_patient_dir(session: TestSession) -> str:
    return os.path.join(SESSIONS_DIR, safe_patient_id(session.patient_id))


def session_file_stem(session: TestSession) -> str:
    dt = datetime.fromtimestamp(session.started_at, tz=timezone.utc)
    return f"{session.session_id}_{dt.strftime('%Y%m%d_%H%M%S')}"


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
    patient_dir = session_patient_dir(session)
    os.makedirs(patient_dir, exist_ok=True)

    dt = datetime.fromtimestamp(session.started_at, tz=timezone.utc)
    filename = f"{session_file_stem(session)}.json"
    filepath = os.path.join(patient_dir, filename)

    summary = session.summary
    calib = session.calibration

    payload = {
        "module": {
            "name": MODULE_NAME,
            "version": session.module_version,
            "algorithmVersion": session.algorithm_version,
            "modelName": session.model_name,
            "modelSha256": session.model_sha256,
            "trackingSource": session.tracking_source,
        },
        "sessionId": session.session_id,
        "patientId": session.patient_id,
        "hand": session.hand.value,
        "startedAt": dt.isoformat(),
        "videoPath": session.video_path,
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
            "openPalm": summary.block_scores.open_palm,
            "fist": summary.block_scores.fist,
            "pinch": summary.block_scores.pinch,
            "pointGesture": summary.block_scores.point_gesture,
            "wristRotation": summary.block_scores.wrist_rotation,
            "zoneMovement": summary.block_scores.zone_movement,
            "holdStability": summary.block_scores.hold_stability,
        } if summary else None,
        "totalScore": summary.total_score if summary else None,
        "qualityCategory": summary.quality_category.value if summary else None,
        "icfCodes": [
            {
                "code": item.code,
                "domain": item.domain,
                "qualifier": item.qualifier,
                "formattedCode": item.formatted_code,
                "label": item.label,
                "problemPercent": item.problem_percent,
                "source": item.source,
                "notes": item.notes,
            }
            for item in (summary.icf_codes if summary else [])
        ] if summary else None,
        "requiresSpecialistConfirmation": summary is not None,
        "technicalValidity": {
            "isInterpretable": bool(summary and summary.valid_tracking_ratio >= 0.65),
            "reason": (
                None if summary and summary.valid_tracking_ratio >= 0.65
                else "tracking_quality_below_threshold_or_no_summary"
            ),
        },
        "recommendation": {
            "mode": summary.recommendation.mode.value,
            "label": summary.recommendation.label,
            "notes": summary.recommendation.notes,
        } if summary else None,
        "expertAssessment": session.expert_assessment or None,
        "events": [
            {
                "timestamp": datetime.fromtimestamp(e.timestamp, tz=timezone.utc).isoformat(),
                "type": e.event_type,
                "severity": e.severity.value,
                "message": e.message,
                "details": e.details,
            }
            for e in session.events
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath
