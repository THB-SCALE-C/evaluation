from typing import Any

from evaluation.types.assessment_types import BaseMetricType


# ---------------------------------------------------
# Assessment Accessors
# ---------------------------------------------------
def format_criterion_key(key: str) -> str:
    return key.replace("_", " ").strip()


def assessment_value(assessment: Any, key: str) -> Any:
    return assessment.get(key) if isinstance(assessment, dict) else getattr(assessment, key, None)


def is_assessment_dict(value: Any) -> bool:
    return isinstance(value, dict) and "score" in value and "feedback" in value


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


# ---------------------------------------------------
# Assessment Scoring
# ---------------------------------------------------
def is_positive_assessment(assessment: Any) -> bool:
    score = assessment_value(assessment, "score")
    max_value = assessment_value(assessment, "max")

    if isinstance(score, str):
        normalized_score = score.lower()
        if isinstance(max_value, str):
            return normalized_score == max_value.lower()
        return normalized_score == "yes"

    if not isinstance(score, (int, float)):
        return False

    numeric_max = to_float(max_value)
    return numeric_max is not None and float(score) >= numeric_max


def is_negative_assessment(assessment: Any) -> bool:
    score = assessment_value(assessment, "score")
    min_value = assessment_value(assessment, "min")

    if isinstance(score, str):
        normalized_score = score.lower()
        if normalized_score in {"yes", "no"}:
            return normalized_score == "no"
        return isinstance(min_value, str) and normalized_score == min_value.lower()

    if not isinstance(score, (int, float)):
        return False

    numeric_min = to_float(min_value)
    return numeric_min is not None and float(score) <= numeric_min


def score_to_numeric(assessment: Any) -> float:
    score = assessment_value(assessment, "score")

    if isinstance(score, str):
        normalized_score = score.lower()
        if normalized_score == "yes":
            return 1.0
        if normalized_score == "no":
            return 0.0
        return to_float(score) or 0.0

    if not isinstance(score, (int, float)):
        return 0.0

    min_value = to_float(assessment_value(assessment, "min"))
    max_value = to_float(assessment_value(assessment, "max"))
    if min_value is None or max_value is None or max_value == min_value:
        return float(score)
    return (float(score) - min_value) / (max_value - min_value)


def normalize_metric(metric: BaseMetricType) -> dict[str, float | int]:
    if isinstance(metric.score, str):
        max_value = assessment_value(metric, "max")
        positive = isinstance(max_value, str) and metric.score.lower() == max_value.lower()
        return {"min": 0, "max": 1, "score": 1 if positive else 0}

    if not isinstance(metric.score, (int, float)):
        raise ValueError("no correct value of `score`")

    numeric_min = to_float(assessment_value(metric, "min"))
    numeric_max = to_float(assessment_value(metric, "max"))
    if numeric_min is None or numeric_max is None or numeric_max == numeric_min:
        raise ValueError("numeric `min` and `max` are required to normalize numeric `score`")

    normalized_score = (float(metric.score) - numeric_min) / (numeric_max - numeric_min)
    return {"min": 0, "max": 1, "score": normalized_score}


# ---------------------------------------------------
# Assessment Rendering
# ---------------------------------------------------
def escape_markdown_cell(value: Any) -> str:
    return "" if value is None else str(value).replace("|", "\\|").replace("\n", "<br>")
