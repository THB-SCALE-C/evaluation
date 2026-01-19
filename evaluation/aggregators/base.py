from typing import Literal
import dspy

from evaluation.types.metric_types import BaseMetricType, BinaryMetricType
from evaluation.types.assessment_types import BaseAssessment, BinaryAssessment


class BaseAggregator(dspy.Module):
    @classmethod
    def _aggregate_scores(cls, scores: list[Literal['yes', 'no'] | int | float], weights: list[float | int] | None = None):
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
    def _aggregate_feedbacks(cls,metrics:list[BaseMetricType], assessments: list[BaseAssessment], exclude_positive_feedback=False):
        if len(metrics) != len(assessments):
            raise ValueError("Metrics and assessments must be of equal length.")

        rows = []
        for m, a in zip(metrics, assessments):
            include_feedback = not (exclude_positive_feedback and a.score == m.max)
            feedback = a.feedback if include_feedback else ""
            rows.append((str(m.name), str(a.score), str(m.scale), str(feedback)))

        headers = ("Metric", "Score", "Scale", "Feedback")
        widths = [len(h) for h in headers]
        for row in rows:
            if len(row[0]) > widths[0]:
                widths[0] = len(row[0])
            if len(row[1]) > widths[1]:
                widths[1] = len(row[1])
            if len(row[2]) > widths[2]:
                widths[2] = len(row[2])
            if len(row[3]) > widths[3]:
                widths[3] = len(row[3])

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
    
    @classmethod
    def aggregate_binary(cls, result:dspy.Prediction, exclude_positive_feedback=False):
        result_dict:dict[str,BinaryAssessment] = {k:v for k,v in dict(result).items() if k != "reasoning"}
        metrics = [BinaryMetricType(name=k) for k in result_dict.keys()]
        total_score = cls._aggregate_scores([v.score for  v in result_dict.values()])
        total_feedback = cls._aggregate_feedbacks(metrics, result_dict.values(),exclude_positive_feedback) # type:ignore
        return total_score,total_feedback
    
    def __init__(self, cot=False):
        pass


