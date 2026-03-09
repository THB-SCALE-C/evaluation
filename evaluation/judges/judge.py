from typing import Any, List, Tuple
import dspy
from evaluation.metrics.base import BaseMetric
from evaluation.signatures.judgement import Judgement
from .base import Evaluation


class Judge(dspy.Module):
    """Evaluates all configured metrics in a single LLM call, ideally with a powerful model.
    Metrics are defined by the provided metrics list, and base instructions can be used to
    tune overall judging behavior (for example, increasing criticalness).

    Import metrics via `evaluation.metrics`

    Add `dspy.InputFields` under `**context`, provide a tuple of description adn type
    """

    def __init__(self, llm: dspy.LM, metrics: list[type[BaseMetric]], cot=False, base_instructions: str | None = None, **context: dict[str, Tuple[str, type]]):
        self.metrics = metrics
        self.judgement = Judgement
        #
        # APPEND CONTEXT INPUTFIELDS
        for key, (desc, type) in context.items():
            self.judgement = self.judgement.append(
                key, dspy.InputField(desc=desc), type)
        #
        # APPEND OUTPUT FIELDS FOR EACH METRIC THAT IS NOT RULE BASED
        for metric in metrics:
            if metric.is_llm_judge:
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
        result_rule_based = self._handle_rule_based_metrics(slides)
        return Evaluation(**result_rule_based, **result, metrics=self.metrics)

    def __call__(self, slides: List, *args,  **context: dict[str, Any]) -> dspy.Prediction:
        return super().__call__(slides, *args, **context)


    def _handle_rule_based_metrics(self, slides:List[dict|Any]) -> dict[str,BaseMetric]:
        results = {}
        for metric in self.metrics:
            if metric.is_llm_judge:
                continue
            for i,slide in enumerate(slides):
                slide = dict(slide)
                slide_type = slide.get("type")
                if metric.required_slide_type == slide_type:
                    result = metric(**slide, slide_idx=i)
                    results[f"{metric.metric_name}-slide-{i}"]=result
        return results

