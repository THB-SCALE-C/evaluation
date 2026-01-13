from typing import Literal
import dspy

from evaluation.types.metric_types import BaseMetricType
from evaluation.types.assessment_types import BaseAssessment


class BaseAggregator(dspy.Module):
    @classmethod
    def aggregate_scores(cls, scores: list[Literal['yes', 'no'] | int | float], weights: list[float | int] | None = None):
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
    def aggregate_feedbacks(cls,metrics:list[BaseMetricType], assessments: list[BaseAssessment], exclude_positive_feedback=False):
        if len(metrics) != len(assessments):
            raise ValueError("Metrics and assessments must be of equal length.")
        
        joined_feedbacks = [
            f"For `{m.name}` achieved `{a.score}` of `{m.scale}`{("" if exclude_positive_feedback and a.score == m.max else f", because: {a.feedback}")}" for m,a in zip(metrics, assessments)
        ]

        return "\n".join(joined_feedbacks)

    def __init__(self, cot=False):
        pass
