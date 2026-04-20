from typing import Any, ClassVar, List, Tuple, Type, get_origin
import dspy
from evaluation.rubrics import BaseRubric
from evaluation.rubrics import BaseRuleRubric
from evaluation.signatures.judgement import Judgement
from .evaluation import Evaluation
from creator.dspy_components.__base__ import BaseComponent


class Judge(dspy.Module):
    """Evaluate slides with configured LLM and rule-based rubrics.

    Parameters:
        llm: DSPy language model used for LLM-judge metrics.
        metrics: Rubric classes to evaluate. LLM rubrics and rule-based rubrics can be mixed.
        cot: If True, use `dspy.ChainOfThought`; otherwise use `dspy.Predict`.
        base_instructions: Optional global instructions added to the judging signature.
        reduce_to_signature_level: If True, flatten rubric output fields into signature-level outputs.
        **context: Additional input fields appended to the signature as
            `name=(description, type)`.
    """

    def __init__(self, llm: dspy.LM, metrics: list[type[BaseRubric | BaseRuleRubric]], 
                 cot=False, base_instructions: str | None = None,
                   reduce_to_signature_level: bool = False,
                     **context: dict[str, Tuple[str, type]]):
        self.metrics = metrics
        self.reduce_to_signature_level = reduce_to_signature_level
        self.judgement = Judgement
        self._flattened_metric_map: dict[str, tuple[str, str]] = {}
        self._rubric_models_by_name: dict[str,
                                          type[BaseRubric]] = {}  # type:ignore
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
                self._rubric_models_by_name[metric.metric_name] = metric
                self.judge_metrics.append((
                    metric.metric_name, dspy.OutputField(desc=metric.__doc__), metric))
        #
        if base_instructions:
            self.judgement = self.judgement.with_instructions(
                instructions=base_instructions)
        #
        # INIT PREDICTOR ACCORDINGLY
        if self.judge_metrics:
            if self.reduce_to_signature_level:
                self.judgement = self._reduce_to_signature_level()
            else:
                for m in self.judge_metrics:
                    self.judgement = self.judgement.append(*m)
        self.judge = dspy.Predict(self.judgement)
        if cot:
            self.judge = dspy.ChainOfThought(self.judgement)
        self.judge.set_lm(llm)

    def forward(self, slides: List[BaseComponent],  **context: dict[str, Any]) -> Evaluation:
        final_results: dict[str, dict[str, dict[str, BaseRubric]]] = {}
        if self.judge_metrics:
            result_judge = self.judge(
                slides=[s.model_dump() for s in slides], **context)
            final_results = self._handle_llm_judge_metric_results(
                result_judge, final_results)
        final_results = self._handle_rule_based_metrics(slides, final_results)
        return Evaluation(final_results)

    def __call__(self, slides: List, *args,  **context: dict[str, Any]) -> Evaluation:
        return super().__call__(slides, *args, **context)  # type:ignore

    def _handle_llm_judge_metric_results(self, result: dspy.Prediction, processed_results) -> dict:
        if self.reduce_to_signature_level:
            metric_results = self._restore_metrics_from_signature(result)
        else:
            metric_results = []
            for key in result.toDict():
                metric_result = getattr(result, key, None)
                if isinstance(metric_result, BaseRubric):
                    metric_results.append(metric_result)

        for metric_result in metric_results:
            if not metric_result.required_slide_type:
                processed_results.setdefault("unit_level", {}).setdefault(
                    metric_result.metric_type, {}).setdefault(metric_result.metric_name, metric_result)
            else:
                processed_results.setdefault("slide_level", {}).setdefault(
                    f"slide-{metric_result.index_}-{metric_result.metric_type}", {}).setdefault(metric_result.metric_name, metric_result)
        return processed_results

    def _restore_metrics_from_signature(self, result: dspy.Prediction) -> list[BaseRubric]:
        payload_by_metric: dict[str, dict[str, Any]] = {}
        for output_name, value in result.toDict().items():
            mapped = self._flattened_metric_map.get(output_name)
            if not mapped:
                continue
            metric_name, field_name = mapped
            payload_by_metric.setdefault(metric_name, {})[field_name] = value

        restored: list[BaseRubric] = []
        for metric_name, payload in payload_by_metric.items():
            rubric_model = self._rubric_models_by_name.get(metric_name)
            if rubric_model:
                restored.append(rubric_model(**payload))
        return restored

    def _handle_rule_based_metrics(self, slides: List[BaseComponent], processed_results: dict) -> dict:

        for metric in self.metrics:
            if not getattr(metric, "metric_type") == "rule_based":
                continue
            required_slide_type = metric.required_slide_type
            if not required_slide_type:
                result = metric(slides) # type:ignore
                processed_results.setdefault("unit_level", {}).setdefault(
                            f"rule_based", {}).setdefault(metric.metric_name, result)
                continue
            # UNIT LEVEL RULE APPLICATION


            # SLIDE BASED RULE APPLICATION
            for i, slide in enumerate(slides):
                    slide_type = slide.slide_type
                    if required_slide_type == slide_type:
                        result = metric(slide, index=i)  # type:ignore
                        processed_results.setdefault("slide_level", {}).setdefault(
                            f"slide-{i}-{slide_type}", {}).setdefault("rule_based", result)
        # alphabetically sort keys inside slide_level dict
        processed_results["slide_level"] = {
            key: processed_results["slide_level"][key] for key in
            sorted(processed_results["slide_level"].keys())
        }
        return processed_results

    def _reduce_to_signature_level(self):
        signature = self.judgement
        flattened_fields: dict[str, tuple[str, str]] = {}

        for metric_name, _, rubric in self.judge_metrics:
            for field_name, field_info in rubric.model_fields.items():
                if get_origin(field_info.annotation) is ClassVar:
                    continue
                if not field_info.is_required():
                    continue
                output_name = f"{metric_name}_{field_name}"
                signature = signature.append(
                    output_name,
                    dspy.OutputField(
                        desc=field_info.description or f"{metric_name}.{field_name}"
                    ),
                    field_info.annotation,
                )
                flattened_fields[output_name] = (metric_name, field_name)

        self._flattened_metric_map = flattened_fields
        self.judgement = signature
        return self.judgement
