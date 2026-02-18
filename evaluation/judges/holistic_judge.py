from typing import Any, List, Tuple
import dspy
from evaluation.metrics.base import BaseMetric
from evaluation.signatures.holistic_judgement import HolisticJudgement
from .base import Evaluation


class HolisticJudge(dspy.Module):
    """Evaluates all configured metrics in a single LLM call, ideally with a powerful model.
    Metrics are defined by the provided metrics list, and base instructions can be used to
    tune overall judging behavior (for example, increasing criticalness).

    Import metrics via `evaluation.metrics`

    Add `dspy.InputFields` under `**context`, provide a tuple of description adn type
    """

    def __init__(self, llm: dspy.LM, metrics: list[type[BaseMetric]], cot=False, base_instructions: str | None = None, **context: dict[str, Tuple[str, type]]):
        self.metrics = metrics
        self.judgement = HolisticJudgement
        #
        # APPEND CONTEXT INPUTFIELDS
        for key, (desc, type) in context.items():
            self.judgement = self.judgement.append(
                key, dspy.InputField(desc=desc), type)
        #
        # APPEND OUTPUT FIELDS FOR EACH METRIC
        for metric in metrics:
            self.judgement = self.judgement.append(
                metric.metric_name, dspy.OutputField(desc=metric.__doc__), metric)
        #
        if base_instructions:
            self.judgement = self.judgement.with_instructions(
                instructions=base_instructions)
        #
        # INIT PREDICTOR ACCORDINGLY
        self.judge = dspy.Predict(self.judgement)
        if cot:
            self.judge = dspy.ChainOfThought(self.judgement)
        self.judge.set_lm(llm)

    def forward(self, slides: List,  **context: dict[str, Any]):
        result = self.judge(slides=slides, **context)
        return Evaluation(**result, metrics=self.metrics)

    def __call__(self, slides: List, *args,  **context: dict[str, Any]) -> dspy.Prediction:
        return super().__call__(slides, *args, **context)
