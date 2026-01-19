import dspy
from evaluation.aggregators.base import BaseAggregator
from evaluation.judges.pointwise.binary.didactical.elicit_performance import ElicitPerformanceJudge
from evaluation.judges.pointwise.binary.formality.cloze_text import ClozeTextJudge


class ClozeTestAggregator(BaseAggregator):
    def __init__(self):
        self.elicit_performance = dspy.ChainOfThought(ElicitPerformanceJudge)
        self.formality_check = ClozeTextJudge()

    def forward(self, cloze_test:dict[str,str], learning_objective:str, previous_content:str):
        result_ep = self.elicit_performance(slide=cloze_test, learning_objective=learning_objective, previous_content=previous_content)
        result_form = self.formality_check(slide=cloze_test)
        result = dspy.Prediction(**result_ep, **result_form)
        return self.aggregate_binary(result)