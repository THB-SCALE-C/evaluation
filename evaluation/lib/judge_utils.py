from typing import Any, ClassVar, TypeAlias, get_origin
import dspy
from evaluation.rubrics.base import BaseRubric
from evaluation.types.assessment_types import BaseMetricType


MetricResultMap: TypeAlias = dict[str, dict[str, dict[str, Any]]]
JudgeMetricSpec: TypeAlias = tuple[str, Any, type[BaseRubric]]
FlattenedMetricMap: TypeAlias = dict[str, tuple[str, str]]


# ---------------------------------------------------
# Judge Output Processing
# ---------------------------------------------------
def apply_metric_criteria_from_field_names(metric_result: BaseRubric) -> None:
    for field_name in metric_result.__class__.model_fields:
        field_value = getattr(metric_result, field_name, None)
        if isinstance(field_value, BaseMetricType):
            field_value.criterion = field_name


def store_metric_result(results: MetricResultMap, metric_result: BaseRubric) -> None:
    if not metric_result.required_slide_type:
        metric_bucket = results.setdefault("unit_level", {}).setdefault(metric_result.metric_type, {})
        metric_bucket[metric_result.metric_name] = metric_result
        return

    slide_key = f"slide-{metric_result.index_}-{metric_result.metric_type}"
    slide_bucket = results.setdefault("slide_level", {}).setdefault(slide_key, {})
    slide_bucket[metric_result.metric_name] = metric_result


def sort_slide_level_results(results: MetricResultMap) -> None:
    slide_level = results.get("slide_level")
    if not slide_level:
        return
    results["slide_level"] = {key: slide_level[key] for key in sorted(slide_level)}


def restore_metrics_from_signature(
    prediction: dspy.Prediction,
    metric_map: FlattenedMetricMap,
    rubric_models: dict[str, type[BaseRubric]],
) -> list[BaseRubric]:
    payload_by_metric: dict[str, dict[str, Any]] = {}

    for output_name, value in prediction.toDict().items():
        mapped = metric_map.get(output_name)
        if not mapped:
            continue
        metric_name, field_name = mapped
        payload_by_metric.setdefault(metric_name, {})[field_name] = value

    restored: list[BaseRubric] = []
    for metric_name, payload in payload_by_metric.items():
        rubric_model = rubric_models.get(metric_name)
        if rubric_model:
            restored.append(rubric_model(**payload))
    return restored


# ---------------------------------------------------
# Judge Signature Construction
# ---------------------------------------------------
def reduce_signature_to_metric_fields(
    signature: Any,
    judge_metrics: list[JudgeMetricSpec],
    omit_signature_prefix: bool,
) -> tuple[Any, FlattenedMetricMap]:
    flattened_fields: FlattenedMetricMap = {}

    for metric_name, _, rubric in judge_metrics:
        for field_name, field_info in rubric.model_fields.items():
            if get_origin(field_info.annotation) is ClassVar:
                continue
            if not field_info.is_required():
                continue

            output_name = field_name if omit_signature_prefix else f"{metric_name}_{field_name}"
            if output_name in flattened_fields:
                flattened_fields[output_name] = (metric_name, field_name)
                continue

            signature = signature.append(
                output_name,
                dspy.OutputField(desc=field_info.description or f"{metric_name}.{field_name}"),
                field_info.annotation,
            )
            flattened_fields[output_name] = (metric_name, field_name)

    return signature, flattened_fields
