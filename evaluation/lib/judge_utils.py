from typing import Any, ClassVar, TypeAlias, get_origin
import dspy
from evaluation.dimensions.base import BaseDimension
from evaluation.types.assessment_types import BaseMetricType

MetricResultMap: TypeAlias = dict[str, BaseDimension|dict]
JudgeMetricSpec: TypeAlias = tuple[str, Any, type[BaseDimension]]
FlattenedMetricMap: TypeAlias = dict[str, tuple[str, str]]



def store_metric_result(results: MetricResultMap, metric_result: BaseDimension) -> None:
    _attach_dimension_field_metadata(metric_result)
    results[metric_result.dimension_name] = metric_result


def merge_metric_results(*result_maps: MetricResultMap) -> MetricResultMap:
    merged: MetricResultMap = {}

    for result_map in result_maps:
        for scope_key, metrics in result_map.items():
            merged.setdefault(scope_key, {}).update(metrics)

    return merged




def restore_metrics_from_signature(
    prediction: dspy.Prediction,
    metric_map: FlattenedMetricMap,
    dimension_models: dict[str, type[BaseDimension]],
) -> list[BaseDimension]:
    payload_by_dimension: dict[str, dict[str, Any]] = {}

    for output_name, value in prediction.toDict().items():
        mapped = metric_map.get(output_name)
        if not mapped:
            continue
        dimension_name, field_name = mapped
        payload_by_dimension.setdefault(dimension_name, {})[field_name] = value

    restored: list[BaseDimension] = []
    for dimension_name, payload in payload_by_dimension.items():
        dimension_model = dimension_models.get(dimension_name)
        if dimension_model:
            restored_metric = dimension_model(**payload)
            _attach_dimension_field_metadata(restored_metric)
            restored.append(restored_metric)
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

    for dimension_name, _, dimension in judge_metrics:
        for field_name, field_info in dimension.model_fields.items():
            if get_origin(field_info.annotation) is ClassVar:
                continue
            if not field_info.is_required():
                continue

            output_name = field_name if omit_signature_prefix else f"{dimension_name}_{field_name}"
            if output_name in flattened_fields:
                flattened_fields[output_name] = (dimension_name, field_name)
                continue

            signature = signature.append(
                output_name,
                dspy.OutputField(desc=field_info.description) if field_info.description else dspy.OutputField(),
                field_info.annotation,
            )
            flattened_fields[output_name] = (dimension_name, field_name)

    return signature, flattened_fields


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def _attach_dimension_field_metadata(metric_result: BaseDimension) -> None:
    for field_name, field_info in metric_result.__class__.model_fields.items():
        metric_value = getattr(metric_result, field_name, None)
        if not isinstance(metric_value, BaseMetricType):
            continue
        metric_value._meta = _build_metric_meta(
            metric_result=metric_result,
            field_info=field_info,
        )


def _build_metric_meta(metric_result: BaseDimension, field_info: Any) -> dict[str, Any]:
    meta = dict(field_info.json_schema_extra or {})
    meta["description"] = field_info.description or ""
    meta["is_llm_judge"] = bool(getattr(metric_result.__class__, "is_llm_judge", False))
    return meta
