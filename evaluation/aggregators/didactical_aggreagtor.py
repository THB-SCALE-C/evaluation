
import dspy
from .base import BaseAggregator
from evaluation.judges.pointwise.binary.didactical.attention_hook import AttentionHookJudge
from evaluation.types.assessment_types import BinaryAssessment
from evaluation.types.metric_types import BinaryMetricType

class DidacticalAggregator(BaseAggregator):
    def __init__(self, cot=False):
        if cot:
            self.attention_hook_judge = dspy.ChainOfThought(AttentionHookJudge)
        else:
            self.attention_hook_judge = dspy.Predict(AttentionHookJudge)

    def forward(self, attention_hook:str, learning_objective:str, further_content:str, exclude_positive_feedback=False):
        result = self.attention_hook_judge(slide=attention_hook, learning_objective=learning_objective, further_content=further_content)
        result_dict:dict[str,BinaryAssessment] = {k:v for k,v in dict(result).items() if k != "reasoning"}
        metrics = [BinaryMetricType(name=k) for k in result_dict.keys()]
        total_score = self.aggregate_scores([v.score for  v in result_dict.values()])
        total_feedback = self.aggregate_feedbacks(metrics, result_dict.values(),exclude_positive_feedback) # type:ignore
        return total_score, total_feedback