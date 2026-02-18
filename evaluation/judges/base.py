from typing import Any, Literal, Tuple
import dspy
from pydantic import BaseModel
from evaluation.metrics.base import BaseMetric
from evaluation.types.assessment_types import BaseAssessment, BinaryAssessment


class Evaluation(dspy.Prediction):
    def __init__(self, metrics: list[type[BaseMetric]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def __repr__(self):
        return f"Total Score: {self.get_total_score()}\n---\nFeedback:\n{self.get_feedback()}"

    def get_feedback(self, exclude_positive_feedback=False) -> str:
        return "\n\n".join(f"## {k}\n{feedback}" for k, feedback in self.get_feedback_per_metric(exclude_positive_feedback).items())

    def get_total_score(self) -> float:
        scores = self.get_total_score_per_metric().values()
        return sum(scores) / len(scores)

    def get_feedback_per_metric(self, exclude_positive_feedback=False) -> dict[str, str]:
        _metrics = self._get_metrics()
        return {
            k: self._aggregate_feedbacks(
                v, exclude_positive_feedback)  # type:ignore
            for k, v in _metrics.items()
        }

    def get_total_score_per_metric(self) -> dict[str, float]:
        _metrics = self._get_metrics()
        return {
            k: self._aggregate_scores([v2.score for v2 in v.values()]) for k, v in _metrics.items()
        }

    def _get_metrics(self):
        _metrics: dict[str, dict[str, BinaryAssessment | Any]] = {}
        for key,metric in self.items():
            if isinstance(metric, BaseMetric):
                metric_keys = metric.model_dump().keys()
                metric_result = {k: getattr(metric, k) for k in metric_keys}
                metric_id = f"{metric.metric_type}:{key}"
                _metrics[metric_id] = metric_result
        return _metrics

    @classmethod
    def _aggregate_scores(cls, scores: list, weights: list[float | int] | None = None):
        if not weights:
            weights = [1.0/len(scores)]*len(scores)
        else:
            if len(weights) != len(scores):
                raise ValueError(
                    "Weights and assessments must be of equal length.")
            if sum(weights) != 1.0:
                raise ValueError("Weights must be summed to 1.0.")

        _scores = [1 if s == "yes" else 0 if s == "no" else s for s in scores]

        return sum([w*s for w, s in zip(weights, _scores)])

    @classmethod
    def _aggregate_feedbacks(cls, metrics: dict[str, BaseAssessment|Any], exclude_positive_feedback=False):
        rows = []
        for m, a in metrics.items():
            if not isinstance(a, BaseAssessment):
                continue
            include_feedback = not (
                exclude_positive_feedback and a.score == a.max)
            feedback = a.feedback if include_feedback else ""
            rows.append((m, str(a.score), str(a.scale), str(feedback)))

        headers = ("Metric", "Score", "Scale", "Feedback")
        widths = [len(h) for h in headers]
        for row in rows:
            for i in range(len(headers)):
                if len(row[i]) > widths[i]:
                    widths[i] = len(row[i])

        def fmt_row(values):
            return (
                f"{values[0]:<{widths[0]}} | "
                f"{values[1]:<{widths[1]}} | "
                f"{values[2]:<{widths[2]}} | "
                f"{values[3]:<{widths[3]}}"
            )

        lines = [fmt_row(headers)]
        lines.append("-" * (sum(widths) + 9))
        for row in rows:
            lines.append(fmt_row(row))

        return "\n".join(lines)
