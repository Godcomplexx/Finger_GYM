from __future__ import annotations

from src.models import BlockScores, IcfCodeAssessment


DOMAIN_BY_PREFIX = {
    "b": "body_functions",
    "s": "body_structures",
    "d": "activities_and_participation",
    "e": "environmental_factors",
}

QUALIFIER_LABELS = {
    0: "no_problem",
    1: "mild_problem",
    2: "moderate_problem",
    3: "severe_problem",
    4: "complete_problem",
    8: "not_specified",
    9: "not_applicable",
}


def domain_for_code(code: str) -> str:
    if not code:
        return "unknown"
    return DOMAIN_BY_PREFIX.get(code[0].lower(), "unknown")


def qualifier_from_problem_percent(problem_percent: float | int) -> int:
    value = max(0.0, min(100.0, float(problem_percent)))
    if value <= 4:
        return 0
    if value <= 24:
        return 1
    if value <= 49:
        return 2
    if value <= 95:
        return 3
    return 4


def problem_percent_from_score(score: int, max_score: int) -> int:
    if max_score <= 0:
        return 100
    ratio = max(0.0, min(1.0, score / max_score))
    return round((1.0 - ratio) * 100)


def make_icf_assessment(
    code: str,
    qualifier: int,
    *,
    problem_percent: int | None = None,
    source: str = "measured",
    notes: list[str] | None = None,
) -> IcfCodeAssessment:
    return IcfCodeAssessment(
        code=code,
        domain=domain_for_code(code),
        qualifier=qualifier,
        formatted_code=f"{code}.{qualifier}",
        label=QUALIFIER_LABELS.get(qualifier, "unknown"),
        problem_percent=problem_percent,
        source=source,
        notes=notes or [],
    )


def make_score_based_icf_assessment(
    code: str,
    score: int,
    max_score: int,
    *,
    source: str,
    notes: list[str] | None = None,
) -> IcfCodeAssessment:
    problem_percent = problem_percent_from_score(score, max_score)
    qualifier = qualifier_from_problem_percent(problem_percent)
    return make_icf_assessment(
        code,
        qualifier,
        problem_percent=problem_percent,
        source=source,
        notes=notes,
    )


def make_not_specified_icf_assessment(
    code: str,
    *,
    source: str = "not_measured",
    notes: list[str] | None = None,
) -> IcfCodeAssessment:
    return make_icf_assessment(
        code,
        8,
        problem_percent=None,
        source=source,
        notes=notes,
    )


def build_icf_profile(block_scores: BlockScores, total_score: int) -> list[IcfCodeAssessment]:
    """Build an ICF-compatible profile from the current hand test results."""
    hand_motor_score = (
        block_scores.open_palm
        + block_scores.fist
        + block_scores.pinch
        + block_scores.point_gesture
        + block_scores.wrist_rotation
        + block_scores.hold_stability
    )
    hand_motor_max = 65

    return [
        make_not_specified_icf_assessment(
            "s110",
            notes=["Body structure is not measured by the webcam hand test."],
        ),
        make_score_based_icf_assessment(
            "b7302",
            hand_motor_score,
            hand_motor_max,
            source="functional_proxy",
            notes=[
                "Derived from hand movement tasks; it is not a direct dynamometry measurement.",
            ],
        ),
        make_score_based_icf_assessment(
            "d520",
            total_score,
            80,
            source="functional_proxy",
            notes=[
                "Derived from observed hand task performance; self-care is not directly observed.",
            ],
        ),
        make_not_specified_icf_assessment(
            "e310",
            notes=["Immediate family/environment factors are not measured by the test."],
        ),
    ]
