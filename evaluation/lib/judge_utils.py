from typing import Any, ClassVar, TypeAlias, get_origin
import dspy
from evaluation.dimensions.base import BaseDimension
from evaluation.types.assessment_types import BaseMetricType

MetricResultMap: TypeAlias = dict[str, BaseDimension|dict]
JudgeMetricSpec: TypeAlias = tuple[str, Any, type[BaseDimension]]
FlattenedMetricMap: TypeAlias = dict[str, tuple[str, str]]



def store_metric_result(results: MetricResultMap, metric_result: BaseDimension) -> None:
    results[metric_result.metric_name] = metric_result


def merge_metric_results(*result_maps: MetricResultMap) -> MetricResultMap:
    merged: MetricResultMap = {}

    for result_map in result_maps:
        for scope_key, metrics in result_map.items():
            merged.setdefault(scope_key, {}).update(metrics)

    return merged


def sort_slide_level_results(results: MetricResultMap) -> None:
    slide_level = results.get("slide_level")
    if not slide_level:
        return
    results["slide_level"] = {key: slide_level[key] for key in sorted(slide_level)}


def restore_metrics_from_signature(
    prediction: dspy.Prediction,
    metric_map: FlattenedMetricMap,
    dimension_models: dict[str, type[BaseDimension]],
) -> list[BaseDimension]:
    payload_by_metric: dict[str, dict[str, Any]] = {}

    for output_name, value in prediction.toDict().items():
        mapped = metric_map.get(output_name)
        if not mapped:
            continue
        metric_name, field_name = mapped
        payload_by_metric.setdefault(metric_name, {})[field_name] = value

    restored: list[BaseDimension] = []
    for metric_name, payload in payload_by_metric.items():
        dimension_model = dimension_models.get(metric_name)
        if dimension_model:
            restored.append(dimension_model(**payload))
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

    for metric_name, _, dimension in judge_metrics:
        for field_name, field_info in dimension.model_fields.items():
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
                dspy.OutputField(desc=field_info.description) if field_info.description else dspy.OutputField(),
                field_info.annotation,
            )
            flattened_fields[output_name] = (metric_name, field_name)

    return signature, flattened_fields
