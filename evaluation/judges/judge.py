import asyncio
from typing import Any, ClassVar, List, Mapping, Tuple, TypeAlias, cast, get_origin
import dspy
from evaluation.rubrics import BaseRubric
from evaluation.rubrics import BaseRuleRubric
from evaluation.signatures.judgement import Judgement
from .evaluation import Evaluation
from creator.schemas.base import BaseComponent

MetricResultMap: TypeAlias = dict[str, dict[str, dict[str, Any]]]
JudgeMetricSpec: TypeAlias = tuple[str, Any, type[BaseRubric]]
FlattenedMetricMap: TypeAlias = dict[str, tuple[str, str]]


class Judge(dspy.Module):
    """Evaluate slides with configured LLM and rule-based rubrics.

    Parameters:
        llm: Optional DSPy language model used for LLM-judge metrics.
        metrics: Rubric classes to evaluate. LLM rubrics and rule-based rubrics can be mixed.
        cot: If True, use `dspy.ChainOfThought`; otherwise use `dspy.Predict`.
        base_instructions: Optional global instructions added to the judging signature.
        reduce_to_signature_level: If True, flatten rubric output fields into signature-level outputs.
        one_call_per_metric: If True, initialize one LLM judge per LLM metric. If False,
            initialize one combined judge for all LLM metrics. If no LLM metrics are configured,
            no judge is initialized.
        async_calls: Only relevant when `one_call_per_metric=True` and multiple LLM judges
            are initialized. If True (default), execute per-metric LLM calls in parallel.
        **context: Additional input fields appended to the signature as
            `name=(description, type)`.
    """

    def __init__(self, llm: dspy.LM | None, metrics: list[type[BaseRubric | BaseRuleRubric]],
                 cot=False, base_instructions: str | None = None,
                 reduce_to_signature_level: bool = False,
                 one_call_per_metric: bool = False,
                 async_calls: bool = True,
                 **context: Tuple[str, type]):
        super().__init__()
        self.metrics = metrics
        self.reduce_to_signature_level = reduce_to_signature_level
        self.one_call_per_metric = one_call_per_metric
        self.async_calls = async_calls
        self.llm = llm
        self.cot = cot
        self.base_instructions = base_instructions

        self.judgement = self._build_base_signature(context)
        self.judge_metrics: list[JudgeMetricSpec] = []
        self._rubric_models_by_name: dict[str, type[BaseRubric]] = {}
        self._collect_llm_metrics()

        self.judge = None
        self.judges_by_metric: dict[str, dspy.Module] = {}
        self._flattened_metric_map: FlattenedMetricMap = {}
        self._flattened_metric_maps_by_metric: dict[str, FlattenedMetricMap] = {}

        # Build either a single multi-metric LLM judge or one judge per LLM metric.
        self._initialize_llm_judges()

    def forward(self, slides: List[BaseComponent], **context: dict[str, Any]) -> Evaluation:
        if self._should_use_parallel_async_calls():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.aforward(slides, **context))
            # Fallback when already inside an event loop.

        results: MetricResultMap = {}
        slide_payload = [slide.model_dump() for slide in slides]

        self._run_llm_judges(slide_payload, context, results)
        self._run_rule_based_metrics(slides, results)
        return Evaluation(results)

    async def aforward(self, slides: List[BaseComponent], **context: dict[str, Any]) -> Evaluation:
        results: MetricResultMap = {}
        slide_payload = [slide.model_dump() for slide in slides]

        await self._run_llm_judges_async(slide_payload, context, results)
        self._run_rule_based_metrics(slides, results)
        return Evaluation(results)

    def __call__(self, slides: List, *args,  **context: dict[str, Any]) -> Evaluation:
        return super().__call__(slides, *args, **context)  # type:ignore

    def _build_base_signature(self, context: Mapping[str, Tuple[str, type]]):
        signature = Judgement
        for key, (desc, value_type) in context.items():
            # Extend the shared input signature with caller-provided context fields.
            signature = signature.append(
                key, dspy.InputField(desc=desc), value_type)
        return signature

    def _collect_llm_metrics(self) -> None:
        for metric in self.metrics:
            if not getattr(metric, "is_llm_judge", False):
                continue
            # Runtime guard above ensures this metric is intended for LLM judging.
            llm_metric = cast(type[BaseRubric], metric)
            self._rubric_models_by_name[llm_metric.metric_name] = llm_metric
            self.judge_metrics.append(
                (llm_metric.metric_name, dspy.OutputField(desc=llm_metric.__doc__), llm_metric)
            )

    def _initialize_llm_judges(self) -> None:
        if not self.judge_metrics:
            return
        if self.llm is None:
            raise ValueError("`llm` is required when at least one LLM-judge metric is configured.")

        if self.one_call_per_metric:
            for metric_name, metric_field, metric_rubric in self.judge_metrics:
                signature = self.judgement
                if self.reduce_to_signature_level:
                    # Flatten only this metric's rubric fields into primitive signature outputs.
                    signature, flattened_map = self._reduce_to_signature_level(
                        signature, [(metric_name, metric_field, metric_rubric)]
                    )
                    self._flattened_metric_maps_by_metric[metric_name] = flattened_map
                else:
                    signature = signature.append(metric_name, metric_field, metric_rubric)

                signature = self._with_base_instructions(signature)
                self.judges_by_metric[metric_name] = self._build_predictor(signature)
            return

        signature = self.judgement
        if self.reduce_to_signature_level:
            # Flatten all rubric fields into one shared signature for a single LLM call.
            signature, self._flattened_metric_map = self._reduce_to_signature_level(
                signature, self.judge_metrics
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
        predictor = dspy.ChainOfThought(signature) if self.cot else dspy.Predict(signature)
        predictor.set_lm(self.llm)  # type:ignore[arg-type]
        return predictor

    def _run_llm_judges(self, slide_payload: list[dict[str, Any]],
                        context: dict[str, Any], results: MetricResultMap) -> None:
        if self.one_call_per_metric and self.judges_by_metric:
            # Independent per-metric calls allow metric-specific prompts/signatures.
            for metric_name, metric_judge in self.judges_by_metric.items():
                prediction = metric_judge(slides=slide_payload, **context)
                self._merge_llm_prediction(prediction, results, metric_name=metric_name)
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
                self._merge_llm_prediction(prediction, results, metric_name=metric_name)
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
            # Convert flattened primitive outputs back into rubric model instances.
            metric_results = self._restore_metrics_from_signature(result, metric_name)
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

    def _restore_metrics_from_signature(self, result: dspy.Prediction,
                                        metric_name: str | None = None) -> list[BaseRubric]:
        payload_by_metric: dict[str, dict[str, Any]] = {}
        metric_map = self._flattened_metric_map
        if metric_name:
            # In per-metric mode, each judge has its own flattened field map.
            metric_map = self._flattened_metric_maps_by_metric.get(metric_name, {})
        for output_name, value in result.toDict().items():
            mapped = metric_map.get(output_name)
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

    def _run_rule_based_metrics(self, slides: List[BaseComponent], processed_results: MetricResultMap) -> None:
        for metric in self.metrics:
            if not getattr(metric, "metric_type") == "rule_based":
                continue
            required_slide_type = metric.required_slide_type
            if not required_slide_type:
                # Unit-level rules evaluate against the full deck.
                result = metric(slides)  # type:ignore
                processed_results.setdefault("unit_level", {}).setdefault(
                    f"rule_based", {}).setdefault(metric.metric_name, result)
                continue
            for i, slide in enumerate(slides):
                slide_type = slide.slide_type
                if required_slide_type == slide_type:
                    # Slide-level rules run only for matching slide types.
                    result = metric(slide, index=i)  # type:ignore
                    processed_results.setdefault("slide_level", {}).setdefault(
                        f"slide-{i}-{slide_type}", {}).setdefault("rule_based", result)

        if "slide_level" in processed_results:
            # Stabilize output order for deterministic downstream processing/tests.
            processed_results["slide_level"] = {
                key: processed_results["slide_level"][key] for key in
                sorted(processed_results["slide_level"].keys())
            }

    def _reduce_to_signature_level(self, signature, judge_metrics: list[JudgeMetricSpec]):
        flattened_fields: FlattenedMetricMap = {}

        for metric_name, _, rubric in judge_metrics:
            for field_name, field_info in rubric.model_fields.items():
                if get_origin(field_info.annotation) is ClassVar:
                    continue
                if not field_info.is_required():
                    continue
                # Prefix with metric name to avoid collisions across rubric models.
                output_name = f"{metric_name}_{field_name}"
                signature = signature.append(
                    output_name,
                    dspy.OutputField(
                        desc=field_info.description or f"{metric_name}.{field_name}"
                    ),
                    field_info.annotation,
                )
                flattened_fields[output_name] = (metric_name, field_name)

        return signature, flattened_fields
