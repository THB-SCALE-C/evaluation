from typing import Any, Iterable, override
import dspy
from evaluation.dimensions.base import BaseDimension
import numpy as np
from evaluation.types.assessment_types import BaseMetricType

PATH_DELIMITER = "."


def _flatten_results(results: dict[str, BaseDimension|dict], path_delimiter: str = "%"):
    _flattened: list[BaseMetricType] = []
    for dimension_name, dimension_value in results.items():
        for metric_name, metric_value in dict(dimension_value).items():
            if not isinstance(metric_value, BaseMetricType):
                continue
            key = f"{dimension_name}{path_delimiter}{metric_name}"
            metric_value._criterion = key
            _flattened.append(metric_value)
    return _flattened


class Evaluation(dspy.Prediction):
    # ---------------------------------------------------
    # Main Functionality
    # ---------------------------------------------------
    def __init__(self, results: dict[str, BaseDimension|dict], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = results
        self._flattened_results = _flatten_results(
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

    @override
    def keys(self, include_dspy=False) -> list[Any]:
        return list(self.to_dict().keys())

    @override
    def get(self, key, default: Any | None = None, normalize=True) -> (Any | None):
        return self.to_dict(normalize=normalize).get(key, default)

    def to_dict(self) -> dict:
        pass

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
            for metric in self._flattened_results:
                if penalty.endswith("*"):
                    penalized = penalty.removesuffix("*") in metric.criterion
                elif penalty.startswith("*"):
                    penalized = penalty.removeprefix("*") in metric.criterion
                else:
                    penalized = metric.criterion == penalty
                if penalized and metric.score == metric.min:  # type:ignore
                    return 0.0
        scores = [_normalize_score(val) for val in self._flattened_results]
        return float(np.mean(scores))

    def to_markdown_table(self, exclude_positive: bool = False, normalize: bool = False, columns=[
        "dimension",
        "metric",
        "score",
        "feedback",
        "description",
        "scale",
        "is_llm_judge"
    ]) -> str:
        """
        Render flattened evaluation metrics as a markdown table.

        The table is generated from `self._flattened_results` and keeps the
        implementation intentionally simple and fast:

        - `columns` fully defines which columns are included and in which order.
        - Column widths are computed dynamically from the header and rendered
          cell values so the markdown output stays aligned and readable.
        - `normalize=True` renders the `score` column as a normalized value in
          the `[0, 1]` range while leaving all other columns unchanged.
        - `exclude_positive=True` filters out metrics whose raw `score` equals
          `metric.max`, so only non-perfect results remain in the output.

        Supported columns are:
        - `dimension`, `metric`: derived from the flattened
          `criterion` path split by `PATH_DELIMITER`.
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
        rows = [
            _metric_to_markdown_row(
                metric, columns=columns, normalize=normalize)
            for metric in self._flattened_results
            if not exclude_positive or metric.score != metric.max  # type:ignore
        ]
        widths = _compute_markdown_widths(columns, rows)
        header = _render_markdown_row(columns, widths)
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
    denominator = val.max - val.min
    if denominator == 0:
        return 0.0
    return (val.score - val.min) / denominator  # type:ignore


def _metric_to_markdown_row(
    metric: BaseMetricType,
    columns: Iterable[str],
    normalize: bool = False,
) -> list[str]:
    criterion_parts = _split_criterion(metric.criterion)
    row: list[str] = []
    for column in columns:
        row.append(_stringify_markdown_value(
            metric, column, criterion_parts, normalize))
    return row


def _split_criterion(criterion: str) -> list[str]:
    parts = criterion.split(PATH_DELIMITER, 2)
    if len(parts) < 3:
        parts.extend([""] * (3 - len(parts)))
    return parts


def _stringify_markdown_value(
    metric: BaseMetricType,
    column: str,
    criterion_parts: list[str],
    normalize: bool,
) -> str:
    value = _resolve_metric_column(metric, column, criterion_parts, normalize)
    return _escape_markdown_cell(value)


def _resolve_metric_column(
    metric: BaseMetricType,
    column: str,
    criterion_parts: list[str],
    normalize: bool,
) -> Any:
    if column == "dimension":
        return criterion_parts[0]
    if column == "metric":
        return criterion_parts[1]
    if column == "score":
        return _normalize_score(metric) if normalize else metric.score# type:ignore
    if column == "description":
        return _metric_description(metric)
    if column == "scale":
        return getattr(metric, "scale", "")
    if column == "is_llm_judge":
        return getattr(metric, "is_llm_judge", "")
    return getattr(metric, column, "")


def _metric_description(metric: BaseMetricType) -> str:
    field_info = metric.__class__.model_fields.get("feedback")
    if not field_info or not field_info.description:
        return ""
    return field_info.description


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
