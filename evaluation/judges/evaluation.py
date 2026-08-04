from typing import Any, Callable, Iterable, Sequence, cast, override
import dspy
from evaluation.dimensions.base import BaseDimension
import numpy as np
from pydantic import create_model
from evaluation.types.assessment_types import BaseMetricType

PATH_DELIMITER = "."


def _flatten_results(results: dict[str, BaseDimension|dict], path_delimiter: str = "%"):
    _flattened: list[BaseMetricType] = []
    for dimension, dimension_value in results.items():
        for metric_name, metric_value in dict(dimension_value).items():
            if not isinstance(metric_value, BaseMetricType):
                continue
            metric_value._criterion = metric_name
            metric_value._meta["dimension"] = metric_value.meta.get("dimension",None) or dimension 
            _flattened.append(metric_value)
    return _flattened


class Evaluation(dspy.Prediction):
    # ---------------------------------------------------
    # Main Functionality
    # ---------------------------------------------------
    def __init__(self, results: dict[str, BaseDimension|dict], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = results
        self.flattened_results = _flatten_results(
            results, path_delimiter=PATH_DELIMITER)

    def __repr__(self):
        score = self.total_score()
        details = self.to_markdown_table(normalize=True)
        return f"Score: {score*100}/100\n\nDetails:\n{details}"

    @override
    def __add__(self, other: Any) -> "Evaluation":  # type: ignore
        if not isinstance(other, Evaluation):
            return NotImplemented
        return Evaluation(results=self._deep_merge_results(self.results, other.results))

    @override
    def __radd__(self, other: Any) -> "Evaluation":  # type: ignore
        if other == 0:
            return self
        if not isinstance(other, Evaluation):
            return NotImplemented
        return other.__add__(self)

    # @override
    def keys(self) -> list[Any]:
        return list(self.to_dict().keys())

    # @override
    def get(self, key, default: Any | None = None, normalize=True) -> (Any | None):
        return self.to_dict(normalize=normalize).get(key, default)

    def to_dict(self, normalize:bool=True) -> dict:
        return {e.criterion:e.model_dump() for e in self.flattened_results}

    def fields(self) -> list[str]:
        return _available_metric_fields(self.flattened_results)

    def to_group_level(self, group_key: str = "dimension") -> "Evaluation":
        group_results: dict[str, BaseDimension] = {}
        grouped_metrics = _group_metrics_by_field(
            self.flattened_results, group_key=group_key)

        for group_name, metrics in grouped_metrics.items():
            if not metrics:
                continue
            aggregated_metric = _build_group_metric(metrics)
            group_results[group_name] = _build_group_result(
                group_name=group_name,
                metric=aggregated_metric,
            )

        return Evaluation(results=group_results)

    def total_score(self, penalties: list[str] = []) -> float:
        """
        Return the mean normalized score across all flattened metrics.

        Use `penalties` to define failing criteria that force the total score
        to `0.0` when a matched metric receives its minimum score.

        Penalty matching rules:
        - exact match: `"section.subsection.metric"`
        - suffix wildcard: `"section.subsection.*"`
        - prefix wildcard: `"*.metric"`
        """
        for penalty in penalties:
            for metric in self.flattened_results:
                if penalty.endswith("*"):
                    penalized = penalty.removesuffix("*") in metric.criterion
                elif penalty.startswith("*"):
                    penalized = penalty.removeprefix("*") in metric.criterion
                else:
                    penalized = metric.criterion == penalty
                if penalized and metric.score == metric.min:  # type:ignore
                    return 0.0
        scores = [_normalize_score(val) for val in self.flattened_results]
        return float(np.mean(scores))

    def to_markdown_table(
        self,
        filter_fn: Callable[[BaseMetricType], bool] | None = None,
        group_by: str | None = None,
        normalize: bool = False,
        columns: Sequence[str] | None = None,
        sort_by: str | None = None,
        ascend: bool = True,
    ) -> str:
        """
        Render flattened evaluation metrics as a markdown table.

        The table is generated from `self.flattened_results` and keeps the
        implementation intentionally simple and fast:

        - `columns` fully defines which columns are included and in which order.
          When `columns=None`, all available columns are included.
        - Column widths are computed dynamically from the header and rendered
          cell values so the markdown output stays aligned and readable.
        - `normalize=True` renders the `score` column as a normalized value in
          the `[0, 1]` range while leaving all other columns unchanged.
        - `filter_fn` filters rows before rendering. It receives each
          `BaseMetricType` and keeps the row when it returns `True`.
        - `group_by` aggregates rows by the given field before rendering.
        - `sort_by` sorts rows by a single resolved column value when provided.
        - `ascend=True` keeps ascending order; `False` reverses it.

        Supported columns are:
        - `dimension`, `metric`: derived from the flattened
        - `score`: raw or normalized score depending on `normalize`.
        - `feedback`: metric feedback.
        - `description`: field description for `feedback` when available.
        - `scale`: metric scale.
        - `is_llm_judge`: whether the metric originates from an LLM judge.
        - any other column name: resolved through `getattr(metric, column, "")`.

        Returns:
            A markdown table string. If no rows remain after filtering, the
            header is still returned so the caller gets a valid empty table.
        """
        grouped_metrics = self.flattened_results if group_by is None else _aggregate_metrics_by_field(
            self.flattened_results,
            group_by,
        )
        filtered_metrics = grouped_metrics if filter_fn is None else [
            metric for metric in grouped_metrics if filter_fn(metric)
        ]
        resolved_columns = list(columns) if columns is not None else _default_markdown_columns(filtered_metrics)
        if sort_by is not None:
            filtered_metrics.sort(
                key=lambda metric: _stringify_markdown_value(
                    metric,
                    sort_by,
                    normalize,
                ),
                reverse=not ascend,
            )
        rows = [
            _metric_to_markdown_row(
                metric, columns=resolved_columns, normalize=normalize)
            for metric in filtered_metrics
        ]
        widths = _compute_markdown_widths(resolved_columns, rows)
        header = _render_markdown_row(resolved_columns, widths)
        separator = _render_markdown_separator(widths)
        body = [_render_markdown_row(row, widths) for row in rows]
        return "\n".join([header, separator, *body])

    @classmethod
    def _deep_merge_results(cls, left: Any, right: Any) -> Any:
        if isinstance(left, dict) and isinstance(right, dict):
            merged = dict(left)
            for key, right_value in right.items():
                if key in merged:
                    merged[key] = cls._deep_merge_results(
                        merged[key], right_value)
                else:
                    merged[key] = right_value
            return merged
        return right


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def _normalize_score(val: BaseMetricType) -> float:
    denominator = val.max - val.min # type:ignore
    if denominator == 0:
        return 0.0
    return (val.score - val.min) / denominator  # type:ignore


def _group_metrics_by_field(
    metrics: list[BaseMetricType],
    group_key: str,
) -> dict[str, list[tuple[str, BaseMetricType]]]:
    grouped_metrics: dict[str, list[tuple[str, BaseMetricType]]] = {}
    for metric in metrics:
        group_name = str(getattr(metric, group_key,None) or metric.meta.get(group_key, ""))
        metric_name = metric.criterion
        grouped_metrics.setdefault(group_name, []).append((metric_name, metric))
    return grouped_metrics


def _aggregate_metrics_by_field(
    metrics: list[BaseMetricType],
    group_key: str,
) -> list[BaseMetricType]:
    aggregated_metrics: list[BaseMetricType] = []
    for group_name, grouped_metrics in _group_metrics_by_field(metrics, group_key).items():
        if not grouped_metrics:
            continue
        aggregated_metric = _build_group_metric(grouped_metrics)
        aggregated_metric._criterion = group_key
        aggregated_metric._meta["dimension"] = group_name
        aggregated_metrics.append(aggregated_metric)
    return aggregated_metrics


def _build_group_metric(
    metrics: list[tuple[str, BaseMetricType]],
) -> BaseMetricType:
    metric_cls = _resolve_group_metric_class(metrics)
    mean_score = float(np.mean([metric.score for _, metric in metrics]))
    feedback = "\n".join(
        f"{criterion_name}: {metric.feedback}" for criterion_name, metric in metrics
    )
    aggregated_metric = metric_cls(
        score=mean_score,
        feedback=feedback,
    )
    aggregated_metric._meta = _build_group_meta(metrics)
    return aggregated_metric


def _resolve_group_metric_class(
    metrics: list[tuple[str, BaseMetricType]],
) -> type[BaseMetricType]:
    source_metric = metrics[0][1]
    aggregated_metric_cls = cast(
        type[BaseMetricType],
        create_model(
            f"{source_metric.__class__.__name__}DimensionLevel",
            __base__=BaseMetricType,
            score=(float, ...),
            feedback=(str, ...),
        ),
    )
    aggregated_metric_cls.type = source_metric.type
    aggregated_metric_cls.scale = source_metric.scale
    aggregated_metric_cls.max = source_metric.max
    aggregated_metric_cls.min = source_metric.min
    return aggregated_metric_cls


def _build_group_meta(
    metrics: list[tuple[str, BaseMetricType]],
) -> dict[str, Any]:
    source_metric = metrics[0][1]
    meta = dict(source_metric.meta)
    meta["old_metrics"] = [
        {
            "original_criterion": metric.criterion,
            "criterion": criterion_name,
            "score": metric.score,
            "feedback": metric.feedback,
        }
        for criterion_name, metric in metrics
    ]
    return meta


def _build_group_result(
    group_name: str,
    metric: BaseMetricType,
) -> BaseDimension:
    dimension_model = cast(
        type[BaseDimension],
        create_model(
            f"{group_name.title().replace('_', '')}GroupLevelResult",
            __base__=BaseDimension,
            **{group_name: (metric.__class__, ...)},
        ),
    )
    return dimension_model(**{group_name: metric})


def _metric_to_markdown_row(
    metric: BaseMetricType,
    columns: Iterable[str],
    normalize: bool = False,
) -> list[str]:
    row: list[str] = []
    for column in columns:
        row.append(_stringify_markdown_value(
            metric, column, normalize))
    return row


def _stringify_markdown_value(
    metric: BaseMetricType,
    column: str,
    normalize: bool,
) -> str:
    value = _resolve_metric_column(metric, column, normalize)
    return _escape_markdown_cell(value)


def _resolve_metric_column(
    metric: BaseMetricType,
    column: str,
    normalize: bool,
) -> Any:
    if column in metric.meta:
        return metric.meta[column]
    if column == "dimension":
        return metric.meta["dimension"]
    if column == "metric":
        return metric.criterion
    if column == "score":
        return _normalize_score(metric) if normalize else metric.score# type:ignore
    if column == "scale":
        return getattr(metric, "scale", "")
    return getattr(metric, column, "")


def _default_markdown_columns(metrics: list[BaseMetricType]) -> list[str]:
    return _available_metric_fields(metrics)


def _available_metric_fields(metrics: list[BaseMetricType]) -> list[str]:
    base_columns = ["dimension", "metric"]
    discovered_columns: list[str] = []
    seen_columns = set(base_columns)
    for metric in metrics:
        for column in _metric_columns(metric):
            if column in seen_columns:
                continue
            seen_columns.add(column)
            discovered_columns.append(column)
    return [*base_columns, *discovered_columns]


def _metric_columns(metric: BaseMetricType) -> list[str]:
    model_columns = list(metric.model_dump().keys())
    meta_columns = list(metric.meta.keys())
    trailing_columns = ["scale"]
    return [*model_columns, *meta_columns, *trailing_columns]


def _escape_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    elif isinstance(value, tuple):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|")


def _compute_markdown_widths(columns: Iterable[str], rows: list[list[str]]) -> list[int]:
    headers = [str(column) for column in columns]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            value_length = len(value)
            if value_length > widths[index]:
                widths[index] = value_length
    return widths


def _render_markdown_row(values: Iterable[str], widths: list[int]) -> str:
    padded = [str(value).ljust(widths[index])
              for index, value in enumerate(values)]
    return f"| {' | '.join(padded)} |"


def _render_markdown_separator(widths: list[int]) -> str:
    return f"| {' | '.join('-' * width for width in widths)} |"
