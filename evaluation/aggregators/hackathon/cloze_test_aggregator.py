import dspy
from evaluation.aggregators.base import BaseAggregator
from evaluation.judges.pointwise.binary.didactical.elicit_performance import ElicitPerformanceJudge
from evaluation.judges.pointwise.binary.formality.cloze_text.rule_check import ClozeTextRuleChecker
from evaluation.judges.pointwise.binary.formality.cloze_text.cloze_test_judge import ClozeTestJudge


class ClozeTestAggregator(BaseAggregator):
    def __init__(self):
        self.elicit_performance = dspy.ChainOfThought(ElicitPerformanceJudge)
        self.formality_check = ClozeTextRuleChecker()
        self.formality_judge = dspy.ChainOfThought(ClozeTestJudge)

    def forward(self, cloze_test:dict[str,str], learning_objective:str, previous_content:str):
        result_ep = self.elicit_performance(slide=cloze_test, learning_objective=learning_objective, previous_content=previous_content)
        result_form_rule_checker = self.formality_check(slide=cloze_test)
        result_form_judge = self.formality_judge(slide=cloze_test)
        full_result = {**result_ep, **result_form_rule_checker, **result_form_judge}
        return self.aggregate_binary(dspy.Prediction(**full_result))