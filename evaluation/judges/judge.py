import asyncio
from typing import Any, Mapping, cast

import dspy
from creator.schemas.base import BaseComponent
from evaluation.lib.judge_utils import (
    FlattenedMetricMap,
    JudgeMetricSpec,
    MetricResultMap,
    apply_metric_criteria_from_field_names,
    reduce_signature_to_metric_fields,
    restore_metrics_from_signature,
    sort_slide_level_results,
    store_metric_result,
)
from evaluation.rubrics import BaseRubric, BaseRuleRubric
from evaluation.signatures.judgement import Judgement
from evaluation.types.assessment_types import BaseMetricType

from .evaluation import Evaluation


class Judge(dspy.Module):
    """Evaluate slides with LLM and rule-based rubrics."""

    # ---------------------------------------------------
    # Main Functionality
    # ---------------------------------------------------
    def __init__(
        self,
        llm: dspy.LM | None,
        metrics: list[type[BaseRubric | BaseRuleRubric]],
        cot: bool = False,
        base_instructions: str | None = None,
        reduce_to_signature_level: bool = False,
        one_call_per_metric: bool = False,
        async_calls: bool = True,
        omit_signature_prefix: bool = False,
        **context: tuple[str | None, type | None],
    ):
        super().__init__()
        self.metrics = metrics
        self.reduce_to_signature_level = reduce_to_signature_level
        self.one_call_per_metric = one_call_per_metric
        self.async_calls = async_calls
        self.llm = llm
        self.cot = cot
        self.base_instructions = base_instructions
        self.omit_signature_prefix = omit_signature_prefix

        self.judgement = self._build_base_signature(context)
        self.judge_metrics: list[JudgeMetricSpec] = []
        self._rubric_models_by_name: dict[str, type[BaseRubric]] = {}
        self._collect_llm_metrics()

        self.judge = None
        self.judges_by_metric: dict[str, dspy.Module] = {}
        self._flattened_metric_map: FlattenedMetricMap = {}
        self._flattened_metric_maps_by_metric: dict[str, FlattenedMetricMap] = {
        }
        self._initialize_llm_judges()

    def forward(self, slides: list[BaseComponent], **context: dict[str, Any]) -> Evaluation:
        if self._should_use_parallel_async_calls():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.aforward(slides, **context))

        results: MetricResultMap = {}
        slide_payload = [slide.model_dump() for slide in slides]

        self._run_llm_judges(slide_payload, context, results)
        self._run_rule_based_metrics(slides, results)
        return Evaluation(results)

    async def aforward(self, slides: list[BaseComponent], **context: dict[str, Any]) -> Evaluation:
        results: MetricResultMap = {}
        slide_payload = [slide.model_dump() for slide in slides]

        await self._run_llm_judges_async(slide_payload, context, results)
        self._run_rule_based_metrics(slides, results)
        return Evaluation(results)

    def __call__(self, *args, **context) -> Evaluation:
        return super().__call__(*args, **context)  # type: ignore

    # ---------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------
    def _build_base_signature(self, context: Mapping[str, tuple[str | None, type | None]]):
        signature = Judgement
        for key, (desc, value_type) in context.items():
            signature = signature.append(
                key, dspy.InputField(desc=desc) if desc else dspy.InputField(), value_type)
        return signature

    def _collect_llm_metrics(self) -> None:
        for metric in self.metrics:
            if not getattr(metric, "is_llm_judge", False):
                continue
            # Runtime guard above ensures this metric is intended for LLM judging.
            llm_metric = cast(type[BaseRubric], metric)
            self._rubric_models_by_name[llm_metric.metric_name] = llm_metric
            self.judge_metrics.append(
                (llm_metric.metric_name, dspy.OutputField(
                    desc=llm_metric.__doc__), llm_metric)
            )

    def _initialize_llm_judges(self) -> None:
        if not self.judge_metrics:
            return
        if self.llm is None:
            raise ValueError(
                "`llm` is required when at least one LLM-judge metric is configured.")

        if self.one_call_per_metric:
            for metric_name, metric_field, metric_rubric in self.judge_metrics:
                signature = self.judgement
                if self.reduce_to_signature_level:
                    signature, flattened_map = reduce_signature_to_metric_fields(
                        signature=signature,
                        judge_metrics=[
                            (metric_name, metric_field, metric_rubric)],
                        omit_signature_prefix=self.omit_signature_prefix,
                    )
                    self._flattened_metric_maps_by_metric[metric_name] = flattened_map
                else:
                    signature = signature.append(
                        metric_name, metric_field, metric_rubric)

                signature = self._with_base_instructions(signature)
                self.judges_by_metric[metric_name] = self._build_predictor(
                    signature)
            return

        signature = self.judgement
        if self.reduce_to_signature_level:
            signature, self._flattened_metric_map = reduce_signature_to_metric_fields(
                signature=signature,
                judge_metrics=self.judge_metrics,
                omit_signature_prefix=self.omit_signature_prefix,
            )
        else:
            for metric_spec in self.judge_metrics:
                signature = signature.append(*metric_spec)

        signature = self._with_base_instructions(signature)
        self.judgement = signature
        self.judge = self._build_predictor(signature)

    def _with_base_instructions(self, signature):
        if self.base_instructions:
            return signature.with_instructions(instructions=self.base_instructions)
        return signature

    def _build_predictor(self, signature):
        predictor = dspy.ChainOfThought(
            signature) if self.cot else dspy.Predict(signature)
        predictor.set_lm(self.llm)  # type:ignore[arg-type]
        return predictor

    def _run_llm_judges(self, slide_payload: list[dict[str, Any]],
                        context: dict[str, Any], results: MetricResultMap) -> None:
        if self.one_call_per_metric and self.judges_by_metric:
            for metric_name, metric_judge in self.judges_by_metric.items():
                prediction = metric_judge(slides=slide_payload, **context)
                self._merge_llm_prediction(
                    prediction, results, metric_name=metric_name)
            return

        if self.judge and self.judge_metrics:
            prediction = self.judge(slides=slide_payload, **context)
            self._merge_llm_prediction(prediction, results)

    async def _run_llm_judges_async(self, slide_payload: list[dict[str, Any]],
                                    context: dict[str, Any], results: MetricResultMap) -> None:
        if self._should_use_parallel_async_calls():
            tasks = [
                judge.acall(slides=slide_payload, **context)
                for judge in self.judges_by_metric.values()
            ]
            predictions = await asyncio.gather(*tasks, return_exceptions=False)
            for (metric_name, _), prediction in zip(self.judges_by_metric.items(), predictions):
                self._merge_llm_prediction(
                    prediction, results, metric_name=metric_name)
            return

        self._run_llm_judges(slide_payload, context, results)

    def _should_use_parallel_async_calls(self) -> bool:
        return (
            self.one_call_per_metric
            and self.async_calls
            and len(self.judges_by_metric) > 1
        )

    def _merge_llm_prediction(self, result: dspy.Prediction, processed_results: MetricResultMap,
                              metric_name: str | None = None) -> None:
        if self.reduce_to_signature_level:
            metric_map = self._flattened_metric_map
            if metric_name:
                metric_map = self._flattened_metric_maps_by_metric.get(
                    metric_name, {})
            metric_results = restore_metrics_from_signature(
                prediction=result,
                metric_map=metric_map,
                rubric_models=self._rubric_models_by_name,
            )
        else:
            metric_results = []
            for key in result.toDict():
                metric_result = getattr(result, key, None)
                if isinstance(metric_result, BaseRubric):
                    metric_results.append(metric_result)

        for metric_result in metric_results:
            apply_metric_criteria_from_field_names(metric_result)
            store_metric_result(processed_results, metric_result)

    def _run_rule_based_metrics(self, slides: list[BaseComponent], processed_results: MetricResultMap) -> None:
        for metric in self.metrics:
            if not getattr(metric, "metric_type") == "rule_based":
                continue
            required_slide_type = metric.required_slide_type
            if not required_slide_type:
                result = metric(slides)  # type: ignore[misc]
                rubric_result = self._ensure_rule_rubric_result(
                    result, metric.metric_name)
                apply_metric_criteria_from_field_names(rubric_result)
                store_metric_result(processed_results, rubric_result)
                continue

            for i, slide in enumerate(slides):
                if required_slide_type != slide.slide_type:
                    continue
                result = metric(slide, index=i)  # type: ignore[misc]
                rubric_result = self._ensure_rule_rubric_result(
                    result, metric.metric_name)
                apply_metric_criteria_from_field_names(rubric_result)
                store_metric_result(processed_results, rubric_result)

        sort_slide_level_results(processed_results)

    def _ensure_rule_rubric_result(
        self,
        result: BaseRubric | BaseMetricType,
        metric_name: str,
    ) -> BaseRubric:
        if isinstance(result, BaseRubric):
            return result
        raise TypeError(
            f"Rule-based metric `{metric_name}` must return `BaseRubric`, got `{type(result).__name__}`."
        )
