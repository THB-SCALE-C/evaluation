import json
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

    def get_feedback(self, exclude_positive_feedback=False, json_formatted=True) -> str:
        base = f"The evaluation assigned a score of {self.get_total_score()}/1. Here is a detailed evaluation report."
        if json_formatted:
            dict_desc="Note: The report is written in JSON-format. Each assessment contains a description on what was assessed in `criterion`, the assessment score in `score`, its respective scale in `scale`, and detailed feedback in `feedback`."
            return base + "\n\n" + dict_desc + "\n\n" + json.dumps(self.get_feedback_dict(exclude_positive_feedback), indent=4)
        return base+"\n\n"+"\n\n".join(f"### {k}\n{feedback}" for k, feedback in self.get_feedback_per_metric(exclude_positive_feedback).items())
    
    def get_feedback_dict(self, exclude_positive_feedback=False) -> dict[str, Any]:
        metrics = self._get_metrics()
        metrics = {k: {k2: {**dict(v2),"scale":str(v2.scale)} for k2, v2 in v.items() if (not exclude_positive_feedback) or (v2.score != v2.max)} for k, v in metrics.items()}
        return metrics

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
        _metrics: dict[str, dict[str, BaseAssessment]] = {}
        for key,metric in self.items():
            if isinstance(metric, BaseMetric):
                metric_items = metric.model_dump().items()
                _sub_metrics = {}
                # SUB METRICS WHICH ARE OF TYPE BASE_ASSESSMENT
                for k,_ in metric_items:
                    v = getattr(metric,k)
                    if not isinstance(v, BaseAssessment):
                        continue
                    sub_metric = getattr(metric, k)
                    if isinstance(sub_metric,BaseAssessment):
                        crit = metric.__class__.model_fields.get(k)
                        sub_metric.criterion = crit.description if crit and hasattr(crit,"description") else k
                        _sub_metrics[k] = sub_metric
                metric_id = f"{metric.metric_type}:{key}"
                _metrics[metric_id] = _sub_metrics
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
    def _aggregate_feedbacks(cls, metrics: dict[str, BaseAssessment], exclude_positive_feedback=False):
        rows = []
        for m, a in metrics.items():
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
