from typing import Any, List, Tuple
import dspy
from evaluation.metrics import BaseMetric
from evaluation.metrics import BaseRuleMetric
from evaluation.signatures.judgement import Judgement
from .evaluation import Evaluation
from creator.dspy_components.__base__ import BaseComponent



class Judge(dspy.Module):
    """Evaluates all configured metrics in a single LLM call, ideally with a powerful model.
    Metrics are defined by the provided metrics list, and base instructions can be used to
    tune overall judging behavior (for example, increasing criticalness).

    Import metrics via `evaluation.metrics`

    Add `dspy.InputFields` under `**context`, provide a tuple of description adn type
    """

    def __init__(self, llm: dspy.LM, metrics: list[type[BaseMetric | BaseRuleMetric]], cot=False, base_instructions: str | None = None, **context: dict[str, Tuple[str, type]]):
        self.metrics = metrics
        self.judgement = Judgement
        #
        # APPEND CONTEXT INPUTFIELDS
        for key, (desc, type) in context.items():
            self.judgement = self.judgement.append(
                key, dspy.InputField(desc=desc), type)
        #
        # APPEND OUTPUT FIELDS FOR EACH METRIC THAT IS NOT RULE BASED
        self.judge_metrics = []
        for metric in metrics:
            if getattr(metric, "is_llm_judge"):
                self.judge_metrics.append((
                    metric.metric_name, dspy.OutputField(desc=metric.__doc__), metric))
        #
        if base_instructions:
            self.judgement = self.judgement.with_instructions(
                instructions=base_instructions)
        #
        # INIT PREDICTOR ACCORDINGLY
        if self.judge_metrics:
            for m in self.judge_metrics:
                self.judgement =self.judgement.append(*m)
        self.judge = dspy.Predict(self.judgement)
        if cot:
            self.judge = dspy.ChainOfThought(self.judgement)
        self.judge.set_lm(llm)

    def forward(self, slides: List[BaseComponent],  **context: dict[str, Any]) -> Evaluation:
        final_results:dict[str,dict[str,dict[str,BaseMetric]]] = {}
        if self.judge_metrics:
            result_judge = self.judge(
                slides=[s.model_dump() for s in slides], **context)
            final_results = self._handle_judge_metric_results(
                result_judge,final_results)
        final_results = self._handle_rule_based_metrics(slides,final_results)
        return Evaluation(final_results)

    def __call__(self, slides: List, *args,  **context: dict[str, Any]) -> Evaluation:
        return super().__call__(slides, *args, **context) # type:ignore

    def _handle_judge_metric_results(self, result: dspy.Prediction, processed_results) -> dict:
        result_dict = result.toDict()
        for key in result_dict:
            metric_result: BaseMetric = getattr(result, key)
            if not metric_result.required_slide_type:
                processed_results.setdefault("unit_level",{}).setdefault(metric_result.metric_type, {}).setdefault(metric_result.metric_name, metric_result)
            else:
                processed_results.setdefault("slide_level",{}).setdefault(
                            f"slide-{metric_result.index_}-{metric_result.metric_type}",{}).setdefault(metric_result.metric_name, metric_result)
        return processed_results

    def _handle_rule_based_metrics(self, slides: List[BaseComponent], processed_results: dict) -> dict:
        for i, slide in enumerate(slides):
            for metric in self.metrics:
                if getattr(metric, "metric_type") == "rule_based":
                    slide_type = slide.slide_type
                    if metric.required_slide_type == slide_type:
                        result = metric(slide, index=i)  # type:ignore
                        processed_results.setdefault("slide_level",{}).setdefault(
                            f"slide-{i}-{slide_type}",{}).setdefault("rule_based", result)
        return processed_results
