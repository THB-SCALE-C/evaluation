from typing import Any
import dspy

from evaluation.metrics.base import BaseMetric
from evaluation.types.assessment_types import BaseAssessment


class Evaluation(dspy.Prediction):
    def __init__(self, results: dict[str, dict[str, dict[str, BaseMetric]]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = results

    def __repr__(self):
        return self.get_assessment()

    def get_assessment(self, exclude_positive_feedback: bool = False, md_header_level:int=1):
        return (
            f"Total Score: {self._compute_overall_score():.4f}\n---\nFeedback:\n"
            f"{self._combine_feedback_markdown(exclude_positive_feedback,md_header_level)}"
        )

    def get_assessment_dict(self, exclude_positive_feedback: bool = False):
        serialized = self._to_plain_dict(
            self.results, exclude_positive_feedback=exclude_positive_feedback
        )
        if isinstance(serialized, dict):
            return serialized
        return {"results": serialized}

    def get_total_score(self) -> float:
        return self._compute_overall_score()

    def _compute_overall_score(self) -> float:
        scores = [self._score_to_numeric(
            a) for a in self._iter_assessments(self.results)]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _combine_feedback_markdown(self, exclude_positive_feedback: bool = False,md_header_level=1) -> str:
        sections = self._build_markdown_sections(
            data=self.results,
            heading_level=md_header_level,
            exclude_positive_feedback=exclude_positive_feedback,
        )
        return "\n\n".join(s for s in sections if s.strip())

    def get_feedback(self, exclude_positive_feedback: bool = False, json_formatted: bool = False, md_header_level:int=1) -> str:
        if json_formatted:
            return str(self.results.__dict__)
        return self._combine_feedback_markdown(exclude_positive_feedback=exclude_positive_feedback,md_header_level=md_header_level)

    def _iter_assessments(self, data: Any):
        if isinstance(data, BaseMetric):
            yield from self._assessments_from_metric(data)
            return
        if isinstance(data, dict):
            for value in data.values():
                yield from self._iter_assessments(value)

    def _assessments_from_metric(self, metric: BaseMetric):
        for key in metric.model_dump():
            value = getattr(metric, key, None)
            if isinstance(value, BaseAssessment):
                if not value.criterion:
                    desc = metric.__class__.model_fields.get(key)
                    value.criterion = desc.description if desc else key
                yield value

    def _build_markdown_sections(
        self,
        data: Any,
        heading_level: int,
        exclude_positive_feedback: bool,
    ) -> list[str]:
        if isinstance(data, BaseMetric):
            table = self._metric_table_markdown(
                data, exclude_positive_feedback=exclude_positive_feedback)
            return [table] if table else []

        if not isinstance(data, dict):
            return []

        sections: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, BaseMetric)):
                sections.append(f"{'#' * min(heading_level, 6)} {key}")
                sections.extend(
                    self._build_markdown_sections(
                        data=value,
                        heading_level=heading_level + 1,
                        exclude_positive_feedback=exclude_positive_feedback,
                    )
                )
        return sections

    def _metric_table_markdown(self, metric: BaseMetric, exclude_positive_feedback: bool) -> str:
        rows: list[tuple[str, str, str, str]] = []
        for key in metric.model_fields:
            assessment = getattr(metric, key, None)
            if not isinstance(assessment, BaseAssessment):
                continue
            if exclude_positive_feedback and self._is_positive_assessment(assessment):
                continue
            criterion = assessment.criterion or key
            score = str(assessment.score)
            scale = str(getattr(assessment, "scale", ""))
            feedback = assessment.feedback
            rows.append(
                (
                    self._escape_md(criterion),
                    self._escape_md(score),
                    self._escape_md(scale),
                    self._escape_md(feedback),
                )
            )

        if not rows:
            return "_No feedback entries._"

        headers = ("Criterion", "Score", "Scale", "Feedback")
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def fmt_row(values: tuple[str, str, str, str]) -> str:
            padded = [cell.ljust(widths[i]) for i, cell in enumerate(values)]
            return f"| {padded[0]} | {padded[1]} | {padded[2]} | {padded[3]} |"

        header = fmt_row(headers)
        separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        body = [fmt_row(row) for row in rows]
        return "\n".join([header, separator, *body])

    def _is_positive_assessment(self, assessment: BaseAssessment) -> bool:
        score = assessment.score
        max_value = getattr(assessment, "max", None)

        if isinstance(score, str):
            if not isinstance(max_value, str):
                return score.lower() == "yes"
            return score.lower() == max_value.lower()

        if isinstance(score, (int, float)):
            numeric_max = self._to_float(max_value)
            if numeric_max is None:
                return False
            return float(score) >= numeric_max

        return False

    def _score_to_numeric(self, assessment: BaseAssessment) -> float:
        score = assessment.score

        if isinstance(score, str):
            lowered = score.lower()
            if lowered == "yes":
                return 1.0
            if lowered == "no":
                return 0.0
            parsed = self._to_float(score)
            if parsed is not None:
                return parsed
            return 0.0

        if isinstance(score, (int, float)):
            return float(score)

        return 0.0

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _escape_md(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>")

    def _to_plain_dict(self, value: Any, exclude_positive_feedback: bool) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, BaseAssessment):
            if exclude_positive_feedback and self._is_positive_assessment(value):
                return None
            return {
                "score": f"`{value.score}` of `{getattr(value, "scale", None)}`",
                "feedback": value.feedback,
            }

        if isinstance(value, BaseMetric):
            out: dict[str, Any] = {}
            for key in value.model_dump():
                converted = self._to_plain_dict(
                    getattr(value, key, None),
                    exclude_positive_feedback=exclude_positive_feedback,
                )
                if converted is not None:
                    out[key] = converted
            return out

        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                converted = self._to_plain_dict(
                    item, exclude_positive_feedback=exclude_positive_feedback
                )
                if converted is not None:
                    out[str(key)] = converted
            return out

        if isinstance(value, (list, tuple, set)):
            return [
                converted
                for converted in (
                    self._to_plain_dict(
                        item, exclude_positive_feedback=exclude_positive_feedback
                    )
                    for item in value
                )
                if converted is not None
            ]

        if hasattr(value, "model_dump") and callable(value.model_dump):
            return self._to_plain_dict(
                value.model_dump(), exclude_positive_feedback=exclude_positive_feedback
            )

        if hasattr(value, "__dict__"):
            return self._to_plain_dict(
                vars(value), exclude_positive_feedback=exclude_positive_feedback
            )

        return str(value)
