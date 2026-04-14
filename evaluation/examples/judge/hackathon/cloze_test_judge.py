from typing import Tuple

import dspy
from evaluation.judges.judge import Judge
from evaluation.metrics.base import BaseMetric
from evaluation.examples.metrics.didactical.elicit_performance import ElicitPerformanceMetric
from evaluation.examples.metrics.rule_based.drag_text import DragTextRuleBased


class ClozeTestJudge(Judge):
    def __init__(self, llm: dspy.LM, cot=False, **context: dict[str, Tuple[str, type]]):
        base_instructions = "You are a very critical judge -- Judging over this cloze-style test."
        metrics = [ElicitPerformanceMetric, DragTextRuleBased]
        super().__init__(llm, metrics, cot, base_instructions, **context)