from inspect import isclass
from typing import Any, Callable, Protocol, cast

import dspy
from creator.schemas.base import BaseComponent
from evaluation.lib.judge_utils import (
    FlattenedMetricMap,
    JudgeMetricSpec,
    MetricResultMap,
    merge_metric_results,
    reduce_signature_to_metric_fields,
    restore_metrics_from_signature,
    sort_slide_level_results,
    store_metric_result,
)
from evaluation.dimensions import BaseDimension, BaseRuleDimension
from evaluation.signatures.judgement import Judgement
from evaluation.types.assessment_types import BaseMetricType
from .evaluation import Evaluation

class _BaseJudge(dspy.Module):
    
    def __call__(self, *args, **kwargs) -> Evaluation:
        return super().__call__(*args, **kwargs) # type:ignore

# ====================================================================

class LLMJudge(_BaseJudge):
    """Evaluate slides with LLM dimensions."""

    # ---------------------------------------------------
    # Main Functionality
    # ---------------------------------------------------
    def __init__(
        self,
        llm: dspy.LM | None,
        llm_as_a_judge_metrics: list[type[BaseDimension]] = [],
        base_signature: type[dspy.Signature] | None = None,
        instructions:str|None = None,
        predictor_type: type[dspy.Module] | None = None,
        reduce_to_signature_level: bool = False,
        omit_signature_prefix: bool = False,
        input_transformer_func:None|Callable[[list[BaseComponent]],list[BaseComponent]] = None
    ):
        """Initialize a judge that combines LLM and rule-based dimension metrics.

        Args:
            llm: Language model used for LLM-based metrics. Required if at least one
                configured metric has `is_llm_judge=True`.
            llm_as_a_judge_metrics: dimension classes evaluated by the LLM.
            base_signature: Optional base DSPy signature. Defaults to `Judgement`.
            instructions: Optional string that overrides the current signatures doc 
                and therefore instructions.
            predictor_type: Optional predictor class (for example `dspy.ChainOfThought`).
                Defaults to `dspy.Predict`.
            reduce_to_signature_level: If true, flattens dimension fields into individual
                signature outputs and restores dimension models from prediction output.
            omit_signature_prefix: When flattening dimension fields, omit metric-name
                prefixes for output fields.
            input_transformer_func: Optional function that takes slides as an argument 
                and returns transformed slides. This is useful when a specific judge only depends on certain slides or values as input.
        """
        super().__init__()
        self.llm_as_a_judge_metrics = llm_as_a_judge_metrics
        self.reduce_to_signature_level = reduce_to_signature_level
        self.llm = llm
        self.predictor_type = predictor_type
        self.omit_signature_prefix = omit_signature_prefix
        self.input_transformer_func = input_transformer_func

        self.base_signature = base_signature
        self.instructions = instructions
        self.judge_metrics: list[JudgeMetricSpec] = []
        self._dimension_models_by_name: dict[str, type[BaseDimension]] = {}
        self._collect_llm_metrics()

        self.judge = None
        self._flattened_metric_map: FlattenedMetricMap = {}
        self.judgement = self._build_signature()
        self._initialize_llm_judge()


    def forward(self, slides: list[BaseComponent], **context: Any) -> Evaluation:
        slide_payload = self._prepare_forward(slides, **context)
        llm_results = self._run_llm_judge(slide_payload, context)
        results = merge_metric_results(llm_results)
        return Evaluation(results)

    async def aforward(self, slides: list[BaseComponent], **context: Any) -> Evaluation:
        slide_payload = self._prepare_forward(slides, **context)
        llm_results = await self._run_llm_judge_async(slide_payload, context)
        results = merge_metric_results(llm_results)
        return Evaluation(results)

    # ---------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------
    def _prepare_forward(self, slides: list[BaseComponent], **context: Any):
        if self.input_transformer_func:
            slides = self.input_transformer_func(slides)
        self.judgement = self._build_signature()
        self.judgement = self._dynamically_update_signature_with_context(self.judgement, context)
        self._initialize_llm_judge()
        slide_payload = [slide.model_dump() for slide in slides]
        return slide_payload
    
    def _build_signature(self) -> type[dspy.Signature]:
        signature = self.base_signature if self.base_signature is not None else Judgement
        if self.instructions:
            signature.__doc__ = self.instructions
        if self.reduce_to_signature_level:
            signature, self._flattened_metric_map = reduce_signature_to_metric_fields(
                signature=signature,
                judge_metrics=self.judge_metrics,
                omit_signature_prefix=self.omit_signature_prefix,
            )
        else:
            for metric_spec in self.judge_metrics:
                signature = signature.append(*metric_spec)
        return signature

    def _collect_llm_metrics(self) -> None:
        for metric in self.llm_as_a_judge_metrics:
            llm_metric = cast(type[BaseDimension], metric)
            self._dimension_models_by_name[llm_metric.metric_name] = llm_metric
            self.judge_metrics.append(
                (llm_metric.metric_name, dspy.OutputField(
                    desc=llm_metric.__doc__), llm_metric)
            )

    def _initialize_llm_judge(self) -> None:
        if not self.judge_metrics:
            return
        if self.llm is None:
            raise ValueError(
                "`llm` is required when at least one LLM-judge metric is configured.")
        self.judge = self._build_predictor(self.judgement)

    def _build_predictor(self, signature):
        predictor = self.predictor_type(
            signature) if self.predictor_type is not None else dspy.Predict(signature)
        predictor.set_lm(self.llm)  # type:ignore[arg-type]
        return predictor
    
    def _dynamically_update_signature_with_context(self, signature, context) -> type[dspy.Signature]:
        for k in context.keys():
            if k not in signature.input_fields:
                signature=signature.insert(1,k,dspy.InputField()) # type:ignore
        return signature

    def _run_llm_judge(
        self,
        slide_payload: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> MetricResultMap:
        results: MetricResultMap = {}
        if not (self.judge and self.judge_metrics):
            return results
        prediction = self.judge(slides=slide_payload, **context)
        results = self._merge_llm_prediction(prediction, results)
        return results

    async def _run_llm_judge_async(
        self,
        slide_payload: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> MetricResultMap:
        results: MetricResultMap = {}
        if not (self.judge and self.judge_metrics):
            return results
        prediction = await self.judge.acall(slides=slide_payload, **context)
        results = self._merge_llm_prediction(prediction, results)
        return results

    def _merge_llm_prediction(
        self,
        result: dspy.Prediction,
        processed_results: MetricResultMap,
    ) -> MetricResultMap:
        merged_results: MetricResultMap = {
            level: scope_map.copy() for level, scope_map in processed_results.items()
        }

        if self.reduce_to_signature_level:
            metric_results = restore_metrics_from_signature(
                prediction=result,
                metric_map=self._flattened_metric_map,
                dimension_models=self._dimension_models_by_name,
            )
        else:
            metric_results = []
            for key in result.toDict():
                metric_result = getattr(result, key, None)
                if isinstance(metric_result, BaseDimension):
                    metric_results.append(metric_result)

        for metric_result in metric_results:
            store_metric_result(merged_results, metric_result)
        return merged_results


# ====================================================================

class RuleBasedJudge(_BaseJudge):
    """Evaluate slides with rule-based dimensions."""

    # ---------------------------------------------------
    # Main Functionality
    # ---------------------------------------------------
    def __init__(
        self,
        rule_based_metrics: list[type[BaseRuleDimension]] = []
    ):
        """Initialize a judge that combines LLM and rule-based dimension metrics.

        Args:
            rule_based_metrics: dimension classes evaluated with deterministic rules.
        """
        super().__init__()
        self.rule_based_metrics = rule_based_metrics
        self._dimension_models_by_name: dict[str, type[BaseDimension]] = {}
        self._flattened_metric_map: FlattenedMetricMap = {}

    def forward(self, slides: list[BaseComponent], **context: Any) -> Evaluation:
        rule_results = self._run_rule_based_metrics(slides, **context)
        return Evaluation(rule_results)


    # ---------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------

    def _run_rule_based_metrics(self, slides: list[BaseComponent], **context) -> MetricResultMap:
        processed_results: MetricResultMap = {}
        for metric in self.rule_based_metrics:
            result = metric(slides, **context)  # type: ignore[misc]
            store_metric_result(processed_results, result)
        return processed_results

# ====================================================================

class FunctionsJudge(_BaseJudge):
    """Evaluate slides with functions"""
    class _AcceptedFunction(Protocol):
        __name__:str
        def __call__(self, slides: list[BaseComponent], llm=None, **kwargs: dict) -> dict[str,BaseMetricType]: ...
    
    def __init__(
        self,
        metric_functions: list[_AcceptedFunction],
        llm = None
    ):
        """

        Args:
            metric_functions: functions that return judgments.

        Example:
            ```
            def metric(slides:list[BaseComponent],**kwargs):
                return {}

            FunctionsJudge(metric_functions=[metric])´´´
        """
        super().__init__()
        self.metric_functions = metric_functions
        self._dimension_models_by_name: dict[str, type[BaseDimension]] = {}
        self._flattened_metric_map: FlattenedMetricMap = {}
        self.llm = llm

    def _run_func_metrics(self, slides: list[BaseComponent], **context) -> MetricResultMap:
        processed_results: MetricResultMap = {}
        for func in self.metric_functions:
            result = func(slides, llm=self.llm, **context)
            metric_name = func.__name__ 
            processed_results[metric_name] = result
        return processed_results

    def forward(self, slides: list[BaseComponent], **context: Any) -> Evaluation:
        results = self._run_func_metrics(slides, **context)
        return Evaluation(results)
