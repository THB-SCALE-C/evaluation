import json
from typing import Any, Iterable, override

import dspy

from evaluation.lib.assessment_utils import (
    format_criterion_key,
    is_assessment_dict,
    is_negative_assessment,
    is_positive_assessment,
    normalize_metric,
    score_to_numeric,
    escape_markdown_cell,
)
from evaluation.rubrics.base import BaseRubric
from evaluation.types.assessment_types import BaseMetricType


class Evaluation(dspy.Prediction):
    """Container and rendering helpers for judge outputs."""

    TABLE_COLUMNS = ("criterion", "score", "feedback", "scale", "description", "path")
    TABLE_HEADERS = ("Criterion", "Score", "Feedback", "Scale", "Description", "Path")
    DATAFRAME_PRIORITY_COLUMNS = (
        "criterion",
        "score",
        "feedback",
        "scale",
        "description",
        "path",
        "path_depth",
        "min",
        "max",
    )
    DATAFRAME_REQUIRED_COLUMNS = frozenset({"path", "criterion", "description", "score", "feedback", "scale", "min", "max"})
    DATAFRAME_ASSESSMENT_KEYS = ("criterion", "description", "score", "feedback", "scale", "min", "max")

    # ---------------------------------------------------
    # Main Functionality
    # ---------------------------------------------------
    def __init__(self, results: dict[str, dict[str, dict[str, BaseRubric]]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = results

    def __repr__(self):
        return self.generate_assessment()

    def __add__(self, other: Any):
        if not isinstance(other, Evaluation):
            return NotImplemented
        return Evaluation(results=self._deep_merge_results(self.results, other.results))

    def __radd__(self, other: Any):
        if other == 0:
            return self
        if not isinstance(other, Evaluation):
            return NotImplemented
        return other.__add__(self)
    
    @override
    def get(self, key, default: Any | None = None, normalize=True) -> (Any | None):
        return self.to_dict(normalize=normalize).get(key,default)

    @classmethod
    def from_dataframe(cls, df):
        pd = cls._require_pandas("from_dataframe")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")

        missing = cls.DATAFRAME_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for from_dataframe(): {sorted(missing)}")

        results: dict[str, Any] = {}
        for idx, row in df.iterrows():
            raw_path = row.get("path")
            if not isinstance(raw_path, str):
                raise ValueError(f"Row {idx} has invalid `path`. Expected dot-separated string.")
            criterion = row.get("criterion")
            if not isinstance(criterion, str) or not criterion.strip():
                raise ValueError(f"Row {idx} has invalid `criterion`. Expected non-empty string.")

            parts = [part.strip() for part in raw_path.split(".") if part.strip()]
            leaf_key = criterion.strip().replace(" ", "_")
            cursor = results
            for part in parts:
                node = cursor.get(part)
                if node is None:
                    node = {}
                    cursor[part] = node
                elif not isinstance(node, dict):
                    raise ValueError(
                        f"Path conflict at row {idx}: `{raw_path}` collides with a non-dict node at `{part}`."
                    )
                cursor = node

            cursor[leaf_key] = {key: row.get(key) for key in cls.DATAFRAME_ASSESSMENT_KEYS}

        return cls(results=results)

    def generate_assessment(
        self,
        exclude_positive_feedback: bool = False,
        md_header_level: int = 1,
        rule_based_as_penalty: bool = False,
        prefix: str = "[### EVALUATION ###]\n",
        suffix: str = "[### EVALUATION END ###]",
        flatten: bool = True,
        exclude_headers: Iterable[str] | None = None,
    ) -> str:
        details = (
            self._render_markdown_feedback_flat(
                exclude_positive_feedback=exclude_positive_feedback,
                exclude_headers=exclude_headers,
            )
            if flatten
            else self._render_markdown_feedback_tree(
                exclude_positive_feedback=exclude_positive_feedback,
                md_header_level=md_header_level,
                exclude_headers=exclude_headers,
            )
        )
        total_score = self._compute_overall_score(rule_based_as_penalty=rule_based_as_penalty)
        return prefix + f"Total Score: {total_score:.4f}\n---\nDetails:\n{details}\n" + suffix

    def to_dict(self, exclude_positive_feedback: bool = False, normalize: bool = False):
        serialized = self._serialize_to_plain_dict(
            self.results,
            exclude_positive_feedback=exclude_positive_feedback,
            normalize=normalize,
        )
        if not isinstance(serialized, dict):
            return {}

        out: dict[str, dict[str, Any]] = {}
        for path, assessment in self._iter_serialized_assessments(serialized):
            row = self._build_dataframe_row(path, assessment)
            criterion = str(row.get("criterion") or format_criterion_key(path[-1] if path else ""))
            payload = dict(row)
            payload.pop("criterion", None)
            out[criterion] = payload
        return out

    def to_dataframe(self, exclude_positive_feedback: bool = False, normalize: bool = False):
        pd = self._require_pandas("to_dataframe")
        serialized = self.to_dict(
            exclude_positive_feedback=exclude_positive_feedback,
            normalize=normalize,
        )

        rows: list[dict[str, Any]] = [{"criterion": criterion, **assessment} for criterion, assessment in serialized.items()]
        if not rows:
            return pd.DataFrame(columns=self.DATAFRAME_PRIORITY_COLUMNS)

        df = pd.DataFrame(rows)
        path_columns = sorted(
            (column for column in df.columns if column.startswith("path_")),
            key=self._path_column_sort_key,
        )
        ordered = [column for column in self.DATAFRAME_PRIORITY_COLUMNS if column in df.columns]
        ordered.extend(column for column in path_columns if column not in ordered)
        ordered.extend(column for column in df.columns if column not in ordered)
        return df.reindex(columns=ordered)

    def get_total_score(self, rule_based_as_penalty: bool = False) -> float:
        return self._compute_overall_score(rule_based_as_penalty=rule_based_as_penalty)

    def get_feedback(self, exclude_positive_feedback: bool = False, json_formatted: bool = False, md_header_level: int = 1) -> str:
        if json_formatted:
            payload = self.to_dict(exclude_positive_feedback=exclude_positive_feedback)
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return self._render_markdown_feedback_tree(
            exclude_positive_feedback=exclude_positive_feedback,
            md_header_level=md_header_level,
        )

    # ---------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------
    def _compute_overall_score(self, rule_based_as_penalty: bool = False) -> float:
        scores: list[float] = []
        for metric, assessment in self._iter_assessments(self.results):
            if rule_based_as_penalty and getattr(metric, "metric_type", None) == "rule_based":
                if not is_negative_assessment(assessment):
                    continue
            scores.append(score_to_numeric(assessment))
        return sum(scores) / len(scores) if scores else 0.0

    def _render_markdown_feedback_tree(
        self,
        exclude_positive_feedback: bool = False,
        md_header_level: int = 1,
        exclude_headers: Iterable[str] | None = None,
    ) -> str:
        sections = self._build_markdown_sections(
            data=self.results,
            heading_level=md_header_level,
            exclude_positive_feedback=exclude_positive_feedback,
            exclude_headers=exclude_headers,
        )
        return "\n".join(section for section in sections if section.strip())

    def _render_markdown_feedback_flat(
        self,
        exclude_positive_feedback: bool = False,
        exclude_headers: Iterable[str] | None = None,
    ) -> str:
        serialized = self.to_dict(
            exclude_positive_feedback=exclude_positive_feedback,
            normalize=False,
        )
        rows = []
        for criterion, assessment in serialized.items():
            row = {
                "criterion": criterion,
                "score": assessment.get("score", ""),
                "feedback": assessment.get("feedback", ""),
                "scale": assessment.get("scale", ""),
                "description": assessment.get("description", ""),
                "path": assessment.get("path", ""),
            }
            if exclude_headers is not None:
                excluded = {header.lower() for header in exclude_headers}
                row = {key: value for key, value in row.items() if key.lower() not in excluded}
            rows.append(row)
        return self._render_feedback_rows(rows, exclude_positive_feedback, exclude_headers=exclude_headers)

    def _iter_assessments(self, data: Any):
        if isinstance(data, BaseRubric):
            yield from self._iter_metric_assessments(data)
            return
        if is_assessment_dict(data):
            yield None, data
            return
        if isinstance(data, dict):
            for value in data.values():
                yield from self._iter_assessments(value)

    def _iter_metric_assessments(self, metric: BaseRubric):
        for key in metric.__class__.model_fields:
            assessment = getattr(metric, key, None)
            if isinstance(assessment, BaseMetricType):
                assessment.criterion = assessment.criterion or format_criterion_key(key)
                yield metric, assessment

    def _build_markdown_sections(
        self,
        data: Any,
        heading_level: int,
        exclude_positive_feedback: bool,
        exclude_headers: Iterable[str] | None = None,
    ) -> list[str]:
        if isinstance(data, BaseRubric):
            rows = self._rows_from_metric(data, exclude_positive_feedback)
            return [self._render_feedback_rows(rows, exclude_positive_feedback, exclude_headers=exclude_headers)]

        if self._is_metric_assessment_map(data):
            rows = self._rows_from_assessment_map(data, exclude_positive_feedback)
            return [self._render_feedback_rows(rows, exclude_positive_feedback, exclude_headers=exclude_headers)]

        if is_assessment_dict(data):
            rows = self._rows_from_assessment_map({"": data}, exclude_positive_feedback)
            return [self._render_feedback_rows(rows, exclude_positive_feedback, exclude_headers=exclude_headers)]

        if not isinstance(data, dict):
            return []

        sections: list[str] = []
        for key, value in data.items():
            if not isinstance(value, (dict, BaseRubric)):
                continue
            child_sections = self._build_markdown_sections(
                data=value,
                heading_level=heading_level + 1,
                exclude_positive_feedback=exclude_positive_feedback,
                exclude_headers=exclude_headers,
            )
            child_sections = [section for section in child_sections if section]
            if child_sections:
                sections.append(f"{'#' * min(heading_level, 6)} {key}")
                sections.extend(child_sections)
        return sections

    def _rows_from_metric(self, metric: BaseRubric, exclude_positive_feedback: bool) -> list[dict[str, str]]:
        rows: list[dict[str, Any]] = []
        for key, field_info in metric.__class__.model_fields.items():
            assessment = getattr(metric, key, None)
            if not isinstance(assessment, BaseMetricType):
                continue
            if exclude_positive_feedback and is_positive_assessment(assessment):
                continue
            rows.append(
                {
                    "criterion": assessment.criterion or format_criterion_key(key),
                    "score": assessment.score,
                    "feedback": assessment.feedback,
                    "scale": getattr(assessment, "scale", ""),
                    "description": field_info.description or "",
                    "path": "",
                }
            )
        return rows

    def _rows_from_assessment_map(self, assessments: dict[str, dict[str, Any]], exclude_positive_feedback: bool) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for key, assessment in assessments.items():
            if not isinstance(assessment, dict):
                continue
            if exclude_positive_feedback and is_positive_assessment(assessment):
                continue
            rows.append(
                {
                    "criterion": assessment.get("criterion") or format_criterion_key(str(key)),
                    "score": assessment.get("score", ""),
                    "feedback": assessment.get("feedback", ""),
                    "scale": assessment.get("scale", ""),
                    "description": assessment.get("description", ""),
                    "path": "",
                }
            )
        return rows

    def _render_feedback_rows(
        self,
        rows: list[dict[str, Any]],
        exclude_positive_feedback: bool,
        exclude_headers: Iterable[str] | None = None,
    ) -> str:
        if not rows:
            return "" if exclude_positive_feedback else "_No feedback entries._"
        return self._render_markdown_table(rows, exclude_headers=exclude_headers)

    def _render_markdown_table(self, rows: list[dict[str, Any]], exclude_headers: Iterable[str] | None = None) -> str:
        excluded = {header.lower() for header in exclude_headers} if exclude_headers is not None else set()
        included_columns: list[tuple[str, str]] = [
            (column, header)
            for column, header in zip(self.TABLE_COLUMNS, self.TABLE_HEADERS)
            if header.lower() not in excluded and column.lower() not in excluded
        ]
        if not included_columns:
            return "" if rows else "_No feedback entries._"

        escaped_rows = [
            tuple(escape_markdown_cell(row.get(column, "")) for column, _ in included_columns)
            for row in rows
        ]
        headers = tuple(header for _, header in included_columns)
        widths = [len(header) for header in headers]
        for row in escaped_rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        def format_row(values: tuple[str, ...]) -> str:
            return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

        header = format_row(headers)
        separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
        return "\n".join([header, separator, *(format_row(row) for row in escaped_rows)])

    def _build_dataframe_row(self, path: tuple[str, ...], assessment: dict[str, Any]) -> dict[str, Any]:
        criterion = assessment.get("criterion") or format_criterion_key(path[-1] if path else "")
        parent_path = path[:-1] if path else ()
        row = {
            "criterion": criterion,
            "score": assessment.get("score"),
            "feedback": assessment.get("feedback"),
            "scale": assessment.get("scale"),
            "description": assessment.get("description"),
            "path": ".".join(parent_path),
            "path_depth": len(parent_path),
            "min": assessment.get("min"),
            "max": assessment.get("max"),
        }
        for idx, part in enumerate(parent_path):
            row[f"path_{idx}"] = part
        for key, value in assessment.items():
            if key not in row:
                row[key] = value
        return row

    def _iter_serialized_assessments(self, node: Any, path: tuple[str, ...] = ()):
        if isinstance(node, dict):
            if is_assessment_dict(node):
                yield path, node
                return
            for key, value in node.items():
                yield from self._iter_serialized_assessments(value, (*path, str(key)))
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                yield from self._iter_serialized_assessments(value, (*path, str(idx)))

    def _is_metric_assessment_map(self, value: Any) -> bool:
        return isinstance(value, dict) and bool(value) and all(is_assessment_dict(item) for item in value.values())

    def _serialize_to_plain_dict(self, value: Any, exclude_positive_feedback: bool, normalize: bool = False) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, BaseMetricType):
            if exclude_positive_feedback and is_positive_assessment(value):
                return None
            payload = {
                **value.model_dump(),
                "criterion": value.criterion,
                "description": "",
                "scale": getattr(value, "scale", None),
                "min": getattr(value, "min", None),
                "max": getattr(value, "max", None),
            }
            if normalize:
                payload.update(normalize_metric(value))
            return payload

        if isinstance(value, BaseRubric):
            out: dict[str, Any] = {}
            for key, field_info in value.__class__.model_fields.items():
                converted = self._serialize_to_plain_dict(
                    getattr(value, key, None),
                    exclude_positive_feedback=exclude_positive_feedback,
                    normalize=normalize,
                )
                if converted is None:
                    continue
                if isinstance(converted, dict) and isinstance(getattr(value, key, None), BaseMetricType):
                    converted["criterion"] = converted.get("criterion") or format_criterion_key(key)
                    converted["description"] = field_info.description or ""
                out[key] = converted
            return out

        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                converted = self._serialize_to_plain_dict(
                    item,
                    exclude_positive_feedback=exclude_positive_feedback,
                    normalize=normalize,
                )
                if converted is not None:
                    out[str(key)] = converted
            return out

        if isinstance(value, (list, tuple, set)):
            return [
                converted
                for converted in (
                    self._serialize_to_plain_dict(
                        item,
                        exclude_positive_feedback=exclude_positive_feedback,
                        normalize=normalize,
                    )
                    for item in value
                )
                if converted is not None
            ]

        if hasattr(value, "model_dump") and callable(value.model_dump):
            return self._serialize_to_plain_dict(
                value.model_dump(),
                exclude_positive_feedback=exclude_positive_feedback,
                normalize=normalize,
            )

        if hasattr(value, "__dict__"):
            return self._serialize_to_plain_dict(
                vars(value),
                exclude_positive_feedback=exclude_positive_feedback,
                normalize=normalize,
            )

        return str(value)

    @staticmethod
    def _path_column_sort_key(column_name: str) -> int:
        suffix = column_name.split("_", 1)[1] if "_" in column_name else ""
        return int(suffix) if suffix.isdigit() else 10**9

    @staticmethod
    def _require_pandas(caller: str):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(f"`pandas` is required for `{caller}()`. Install it with `pip install pandas`.") from exc
        return pd

    @classmethod
    def _deep_merge_results(cls, left: Any, right: Any) -> Any:
        if isinstance(left, dict) and isinstance(right, dict):
            merged = dict(left)
            for key, right_value in right.items():
                if key in merged:
                    merged[key] = cls._deep_merge_results(merged[key], right_value)
                else:
                    merged[key] = right_value
            return merged
        return right
