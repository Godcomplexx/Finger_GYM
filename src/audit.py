from __future__ import annotations

from typing import Any

from src.models import AuditEvent, EventSeverity, TestSession


def log_event(
    session: TestSession,
    event_type: str,
    message: str,
    severity: EventSeverity = EventSeverity.INFO,
    details: dict[str, Any] | None = None,
) -> None:
    session.events.append(
        AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            details=details or {},
        )
    )
