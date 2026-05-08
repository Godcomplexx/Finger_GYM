from __future__ import annotations

import os
import textwrap
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

from src.app_info import MODULE_NAME
from src.models import ExerciseStatus, TestSession
from src.presentation.renderer import GESTURE_LABELS


def _find_font(bold: bool = False) -> str | None:
    if os.name == "nt":
        candidates = [
            r"C:\Windows\Fonts\segoeui.ttf" if not bold else r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\arial.ttf" if not bold else r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = _find_font(bold)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _status_label(status: ExerciseStatus) -> str:
    labels = {
        ExerciseStatus.DONE: "Выполнено",
        ExerciseStatus.PARTIAL: "Частично",
        ExerciseStatus.UNRELIABLE: "Не оценено",
        ExerciseStatus.SKIPPED: "Пропущено",
    }
    return labels.get(status, status.value)


def _icf_qualifier_label(qualifier: int) -> str:
    labels = {
        0: "нет проблемы",
        1: "легкая проблема",
        2: "умеренная проблема",
        3: "тяжелая проблема",
        4: "полная проблема",
        8: "не определено",
        9: "не применимо",
    }
    return labels.get(qualifier, str(qualifier))


def save_pdf_report(session: TestSession, json_path: str) -> str | None:
    if session.summary is None:
        return None

    pdf_path = os.path.splitext(json_path)[0] + ".pdf"
    summary = session.summary
    started = datetime.fromtimestamp(session.started_at, tz=timezone.utc)

    page = Image.new("RGB", (1240, 1754), "white")  # A4 at ~150 dpi.
    draw = ImageDraw.Draw(page)
    title_f = _font(38, bold=True)
    h1_f = _font(26, bold=True)
    text_f = _font(22)
    small_f = _font(18)

    x = 70
    y = 60
    line = 34

    draw.text((x, y), "Отчет по результатам тестирования", font=title_f, fill=(20, 20, 20))
    y += 62
    draw.text((x, y), MODULE_NAME, font=text_f, fill=(70, 70, 70))
    y += 48

    meta = [
        ("ID пациента", session.patient_id),
        ("ID сессии", session.session_id),
        ("Дата/время UTC", started.strftime("%Y-%m-%d %H:%M:%S")),
    ]
    for label, value in meta:
        draw.text((x, y), f"{label}: ", font=text_f, fill=(35, 35, 35))
        draw.text((x + 260, y), str(value), font=text_f, fill=(35, 35, 35))
        y += line

    if summary.icf_codes:
        y += 24
        draw.text((x, y), "ICF / MKF", font=h1_f, fill=(20, 20, 20))
        y += 44
        for item in summary.icf_codes:
            percent = "" if item.problem_percent is None else f", {item.problem_percent}%"
            text = f"{item.formatted_code}  {item.label}{percent}  [{item.source}]"
            draw.text((x, y), text, font=text_f, fill=(35, 35, 35))
            y += line

    expert_icf = (session.expert_assessment or {}).get("icf") or {}
    if expert_icf:
        y += 24
        draw.text((x, y), "МКФ-оценка врача", font=h1_f, fill=(20, 20, 20))
        y += 44
        for code, value in expert_icf.items():
            qualifier = value.get("qualifier")
            formatted = value.get("formattedCode") or f"{code}.{qualifier}"
            qualifier_text = _icf_qualifier_label(qualifier)
            text = f"{formatted}  {qualifier_text}"
            draw.text((x, y), text, font=text_f, fill=(35, 35, 35))
            y += line

    y += 24
    draw.text((x, y), "Итог", font=h1_f, fill=(20, 20, 20))
    y += 44
    draw.text((x, y), f"Общий балл: {summary.total_score} / 80", font=text_f, fill=(20, 20, 20))
    y += line
    draw.text((x, y), f"Категория качества: {summary.quality_category.value}", font=text_f, fill=(20, 20, 20))
    y += line
    draw.text((x, y), f"Качество трекинга: {summary.valid_tracking_ratio:.2f}", font=text_f, fill=(20, 20, 20))
    y += line
    draw.text((x, y), f"Рекомендация: {summary.recommendation.label}", font=text_f, fill=(20, 20, 20))
    y += line + 12
    for note in summary.recommendation.notes:
        for wrapped in textwrap.wrap(note, width=88):
            draw.text((x + 20, y), wrapped, font=small_f, fill=(70, 70, 70))
            y += 26

    y += 24
    draw.text((x, y), "Блоки оценки", font=h1_f, fill=(20, 20, 20))
    y += 44
    bs = summary.block_scores
    block_rows = [
        ("Открытая ладонь", bs.open_palm, 10),
        ("Кулак", bs.fist, 15),
        ("Щипковый захват", bs.pinch, 15),
        ("Указательный жест", bs.point_gesture, 10),
        ("Поворот кисти", bs.wrist_rotation, 10),
        ("Перемещение по зонам", bs.zone_movement, 15),
        ("Удержание руки", bs.hold_stability, 5),
    ]
    for name, score, max_score in block_rows:
        draw.text((x, y), name, font=text_f, fill=(35, 35, 35))
        draw.text((x + 520, y), f"{score} / {max_score}", font=text_f, fill=(35, 35, 35))
        y += line

    y += 24
    draw.text((x, y), "Упражнения", font=h1_f, fill=(20, 20, 20))
    y += 44
    for result in summary.exercise_results:
        label = GESTURE_LABELS.get(result.exercise_id, result.exercise_id)
        status = _status_label(result.status)
        text = f"{label}: {status}, {result.score} / {result.max_score}"
        draw.text((x, y), text, font=text_f, fill=(35, 35, 35))
        y += line

    y = max(y + 32, 1580)
    footer = (
        "Отчет сформирован автоматически. Результаты требуют интерпретации специалистом "
        "и не являются самостоятельным медицинским заключением."
    )
    for wrapped in textwrap.wrap(footer, width=95):
        draw.text((x, y), wrapped, font=small_f, fill=(90, 90, 90))
        y += 26

    page.save(pdf_path, "PDF", resolution=150.0)
    return pdf_path
