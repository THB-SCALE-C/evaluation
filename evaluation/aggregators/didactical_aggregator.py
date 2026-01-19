
import dspy
from .base import BaseAggregator
from evaluation.judges.pointwise.binary.didactical.attention_hook import AttentionHookJudge
from evaluation.types.assessment_types import BinaryAssessment
from evaluation.types.metric_types import BinaryMetricType

class DidacticalAggregator(BaseAggregator):
    def __init__(self, cot=False, attention_hook=True):
        self.attention_hook_judge = None
        if attention_hook:
            if cot:
                self.attention_hook_judge = dspy.ChainOfThought(AttentionHookJudge)
            else:
                self.attention_hook_judge = dspy.Predict(AttentionHookJudge)

    def forward(self, attention_hook:str, learning_objective:str, further_content:str, exclude_positive_feedback=False):
        if self.attention_hook_judge is None:
            raise ValueError("AttentionHookJudge is disabled")
        result = self.attention_hook_judge(slide=attention_hook, learning_objective=learning_objective, further_content=further_content)
        return self.aggregate_binary(result)
