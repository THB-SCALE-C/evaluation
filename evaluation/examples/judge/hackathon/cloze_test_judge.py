from typing import Tuple
import dspy
from evaluation.judges.judge import Judge
from evaluation.examples.rubrics.didactical.elicit_performance import ElicitPerformanceMetric
from evaluation.examples.rubrics.rule_based.drag_text import DragTextRuleBased


class ClozeTestJudge(Judge):
    def __init__(
        self,
        llm: dspy.LM,
        cot: bool = False,
        reduce_to_signature_level: bool = False,
        **context: dict[str, Tuple[str, type]],
    ):
        base_instructions = "You are a very critical judge -- Judging over this cloze-style test."
        metrics = [ElicitPerformanceMetric, DragTextRuleBased]
        super().__init__(
            llm=llm,
            metrics=metrics,
            cot=cot,
            base_instructions=base_instructions,
            reduce_to_signature_level=reduce_to_signature_level,
            **context,
        )
