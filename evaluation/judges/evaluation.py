import json
import math
import re
from typing import Any

import dspy

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

    def __init__(self, results: dict[str, dict[str, dict[str, BaseRubric]]], *args, **kwargs):
        """Initialize an evaluation object with nested rubric results."""
        super().__init__(*args, **kwargs)
        self.results = results

    def __repr__(self):
        """Return the default human-readable assessment."""
        return self.generate_assessment()

    def __add__(self, other: Any):
        """Return a new evaluation with recursively merged results."""
        if not isinstance(other, Evaluation):
            return NotImplemented
        return Evaluation(results=self._deep_merge_results(self.results, other.results))

    def __radd__(self, other: Any):
        """Support `sum([...], Evaluation(...))` patterns."""
        if other == 0:
            return self
        if not isinstance(other, Evaluation):
            return NotImplemented
        return other.__add__(self)

    @classmethod
    def from_dataframe(cls, df):
        """Create an evaluation from flat dataframe rows."""
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

            # Empty `path` is treated as root-level insertion.
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
        flattened: bool = False,
    ) -> str:
        """Render the full assessment text with score and detailed feedback."""
        details = (
            self._render_markdown_feedback_flat(exclude_positive_feedback)
            if flattened
            else self._render_markdown_feedback_tree(exclude_positive_feedback, md_header_level)
        )
        total_score = self._compute_overall_score(rule_based_as_penalty=rule_based_as_penalty)
        return prefix + f"Total Score: {total_score:.4f}\n---\nDetails:\n{details}\n" + suffix

    def to_dict(self, exclude_positive_feedback: bool = False, flatten: bool = False, normalize: bool = False):
        """Serialize results to a plain dict, optionally flattened and normalized."""
        serialized = self._serialize_to_plain_dict(
            self.results,
            exclude_positive_feedback=exclude_positive_feedback,
            normalize=normalize,
        )
        if flatten:
            return self._flatten_scores(serialized, prefix="")
        return serialized if isinstance(serialized, dict) else {"results": serialized}

    def to_dataframe(self, exclude_positive_feedback: bool = False, normalize: bool = False):
        """Convert results to a flat pandas dataframe."""
        pd = self._require_pandas("to_dataframe")
        serialized = self.to_dict(
            exclude_positive_feedback=exclude_positive_feedback,
            flatten=False,
            normalize=normalize,
        )

        rows: list[dict[str, Any]] = [
            self._build_dataframe_row(path, assessment)
            for path, assessment in self._iter_serialized_assessments(serialized)
        ]
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
        """Return the averaged score over all serialized assessments."""
        return self._compute_overall_score(rule_based_as_penalty=rule_based_as_penalty)

    def get_feedback(self, exclude_positive_feedback: bool = False, json_formatted: bool = False, md_header_level: int = 1) -> str:
        """Render feedback in markdown or JSON format."""
        if json_formatted:
            payload = self.to_dict(exclude_positive_feedback=exclude_positive_feedback, flatten=False)
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return self._render_markdown_feedback_tree(
            exclude_positive_feedback=exclude_positive_feedback,
            md_header_level=md_header_level,
        )

    def _compute_overall_score(self, rule_based_as_penalty: bool = False) -> float:
        """Compute the arithmetic mean score from available assessments."""
        scores: list[float] = []
        for metric, assessment in self._iter_assessments(self.results):
            # In penalty mode we only count rule-based findings when they are negative.
            if rule_based_as_penalty and getattr(metric, "metric_type", None) == "rule_based":
                if not self._is_negative_assessment(assessment):
                    continue
            scores.append(self._score_to_numeric(assessment))
        return sum(scores) / len(scores) if scores else 0.0

    def _render_markdown_feedback_tree(self, exclude_positive_feedback: bool = False, md_header_level: int = 1) -> str:
        """Render hierarchical markdown feedback grouped by nested result keys."""
        sections = self._build_markdown_sections(
            data=self.results,
            heading_level=md_header_level,
            exclude_positive_feedback=exclude_positive_feedback,
        )
        return "\n".join(section for section in sections if section.strip())

    def _render_markdown_feedback_flat(self, exclude_positive_feedback: bool = False) -> str:
        """Render one flat markdown table across all assessments."""
        serialized = self.to_dict(
            exclude_positive_feedback=exclude_positive_feedback,
            flatten=False,
            normalize=False,
        )
        rows = [
            self._build_markdown_row(path, assessment)
            for path, assessment in self._iter_serialized_assessments(serialized)
        ]
        return self._render_feedback_rows(rows, exclude_positive_feedback)

    def _iter_assessments(self, data: Any):
        """Yield `(metric, assessment)` pairs from nested result structures."""
        if isinstance(data, BaseRubric):
            yield from self._iter_metric_assessments(data)
            return
        if self._is_assessment_dict(data):
            yield None, data
            return
        if isinstance(data, dict):
            for value in data.values():
                yield from self._iter_assessments(value)

    def _iter_metric_assessments(self, metric: BaseRubric):
        """Yield metric-assessment pairs for all `BaseMetricType` fields in a rubric."""
        for key in metric.__class__.model_fields:
            assessment = getattr(metric, key, None)
            if isinstance(assessment, BaseMetricType):
                assessment.criterion = assessment.criterion or self._format_criterion_key(key)
                yield metric, assessment

    def _flatten_scores(self, payload: Any, prefix: str = "judge") -> dict[str, float | str]:
        """Flatten only score fields into dot-path keys."""
        flattened: dict[str, float | str] = {}

        def visit(node: Any, path: str) -> None:
            if isinstance(node, (int, float, bool)):
                if path.endswith(".score"):
                    value = float(node)
                    if math.isfinite(value):
                        flattened[path] = value
                return
            if isinstance(node, str):
                if path.endswith(".score"):
                    flattened[path] = node
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    # Keys are sanitized to keep output paths machine-safe.
                    safe_key = self._sanitize_metric_key(str(key))
                    visit(value, f"{path}.{safe_key}" if path else safe_key)
                return
            if isinstance(node, (list, tuple)):
                for idx, value in enumerate(node):
                    visit(value, f"{path}.{idx}" if path else str(idx))

        visit(payload, prefix)
        return flattened

    def _build_markdown_sections(self, data: Any, heading_level: int, exclude_positive_feedback: bool) -> list[str]:
        """Build hierarchical markdown sections for the nested result tree."""
        if isinstance(data, BaseRubric):
            return [self._render_feedback_rows(self._rows_from_metric(data, exclude_positive_feedback), exclude_positive_feedback)]
        if self._is_metric_assessment_map(data):
            return [self._render_feedback_rows(self._rows_from_assessment_map(data, exclude_positive_feedback), exclude_positive_feedback)]
        if self._is_assessment_dict(data):
            return [self._render_feedback_rows(self._rows_from_assessment_map({"": data}, exclude_positive_feedback), exclude_positive_feedback)]
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
            )
            child_sections = [section for section in child_sections if section]
            if child_sections:
                sections.append(f"{'#' * min(heading_level, 6)} {key}")
                sections.extend(child_sections)
        return sections

    def _rows_from_metric(self, metric: BaseRubric, exclude_positive_feedback: bool) -> list[dict[str, str]]:
        """Collect markdown-table rows from a rubric instance."""
        rows: list[dict[str, Any]] = []
        for key, field_info in metric.__class__.model_fields.items():
            assessment = getattr(metric, key, None)
            if not isinstance(assessment, BaseMetricType):
                continue
            if exclude_positive_feedback and self._is_positive_assessment(assessment):
                continue
            rows.append(
                {
                    "criterion": assessment.criterion or self._format_criterion_key(key),
                    "score": assessment.score,
                    "feedback": assessment.feedback,
                    "scale": getattr(assessment, "scale", ""),
                    "description": field_info.description or "",
                    "path": "",
                }
            )
        return rows

    def _rows_from_assessment_map(self, assessments: dict[str, dict[str, Any]], exclude_positive_feedback: bool) -> list[dict[str, str]]:
        """Collect markdown-table rows from a dict of assessments."""
        rows: list[dict[str, str]] = []
        for key, assessment in assessments.items():
            if not isinstance(assessment, dict):
                continue
            if exclude_positive_feedback and self._is_positive_assessment(assessment):
                continue
            rows.append(
                {
                    "criterion": assessment.get("criterion") or self._format_criterion_key(str(key)),
                    "score": assessment.get("score", ""),
                    "feedback": assessment.get("feedback", ""),
                    "scale": assessment.get("scale", ""),
                    "description": assessment.get("description", ""),
                    "path": "",
                }
            )
        return rows

    def _render_feedback_rows(self, rows: list[dict[str, Any]], exclude_positive_feedback: bool) -> str:
        """Render rows to markdown or return empty placeholder text."""
        if not rows:
            return "" if exclude_positive_feedback else "_No feedback entries._"
        return self._render_markdown_table(rows)

    def _render_markdown_table(self, rows: list[dict[str, Any]]) -> str:
        """Render rows into a fixed-column markdown table."""
        escaped_rows = [
            tuple(self._escape_md(row.get(column, "")) for column in self.TABLE_COLUMNS)
            for row in rows
        ]
        widths = [len(header) for header in self.TABLE_HEADERS]
        for row in escaped_rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        def format_row(values: tuple[str, ...]) -> str:
            return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

        header = format_row(self.TABLE_HEADERS)
        separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
        return "\n".join([header, separator, *(format_row(row) for row in escaped_rows)])

    def _build_dataframe_row(self, path: tuple[str, ...], assessment: dict[str, Any]) -> dict[str, Any]:
        """Build one dataframe row from an assessment leaf and its path."""
        criterion = assessment.get("criterion") or self._format_criterion_key(path[-1] if path else "")
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

    def _build_markdown_row(self, path: tuple[str, ...], assessment: dict[str, Any]) -> dict[str, Any]:
        """Build one markdown-table row from an assessment leaf and its path."""
        return {
            "criterion": assessment.get("criterion") or self._format_criterion_key(path[-1] if path else ""),
            "score": assessment.get("score", ""),
            "feedback": assessment.get("feedback", ""),
            "scale": assessment.get("scale", ""),
            "description": assessment.get("description", ""),
            "path": ".".join(path[:-1]) if path else "",
        }

    def _iter_serialized_assessments(self, node: Any, path: tuple[str, ...] = ()):
        """Yield assessment leaves from serialized dict/list structures."""
        # A leaf is recognized by the assessment signature (`score` + `feedback`).
        if isinstance(node, dict):
            if self._is_assessment_dict(node):
                yield path, node
                return
            for key, value in node.items():
                yield from self._iter_serialized_assessments(value, (*path, str(key)))
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                yield from self._iter_serialized_assessments(value, (*path, str(idx)))

    def _is_positive_assessment(self, assessment: Any) -> bool:
        """Return whether an assessment is at its positive bound."""
        score = self._assessment_value(assessment, "score")
        max_value = self._assessment_value(assessment, "max")
        if isinstance(score, str):
            return score.lower() == (max_value.lower() if isinstance(max_value, str) else "yes")
        if isinstance(score, (int, float)):
            numeric_max = self._to_float(max_value)
            return numeric_max is not None and float(score) >= numeric_max
        return False

    def _is_negative_assessment(self, assessment: Any) -> bool:
        """Return whether an assessment is at its negative bound."""
        score = self._assessment_value(assessment, "score")
        min_value = self._assessment_value(assessment, "min")
        if isinstance(score, str):
            lowered = score.lower()
            if lowered in {"yes", "no"}:
                return lowered == "no"
            return isinstance(min_value, str) and lowered == min_value.lower()
        if isinstance(score, (int, float)):
            numeric_min = self._to_float(min_value)
            return numeric_min is not None and float(score) <= numeric_min
        return False

    def _score_to_numeric(self, assessment: Any) -> float:
        """Convert assessment score to a numeric value for averaging."""
        score = self._assessment_value(assessment, "score")
        if isinstance(score, str):
            lowered = score.lower()
            if lowered == "yes":
                return 1.0
            if lowered == "no":
                return 0.0
            return self._to_float(score) or 0.0
        if not isinstance(score, (int, float)):
            return 0.0

        min_value = self._to_float(self._assessment_value(assessment, "min"))
        max_value = self._to_float(self._assessment_value(assessment, "max"))
        if min_value is None or max_value is None or max_value == min_value:
            return float(score)
        return (float(score) - min_value) / (max_value - min_value)

    def _normalize(self, value: BaseMetricType) -> dict[str, float | int]:
        """Normalize a metric score to the [0, 1] interval using metric bounds."""
        if isinstance(value.score, str):
            max_value = self._assessment_value(value, "max")
            return {"min": 0, "max": 1, "score": 1 if isinstance(max_value, str) and value.score == max_value else 0}
        if isinstance(value.score, (int, float)):
            numeric_min = self._to_float(self._assessment_value(value, "min"))
            numeric_max = self._to_float(self._assessment_value(value, "max"))
            if numeric_min is None or numeric_max is None or numeric_max == numeric_min:
                raise ValueError("numeric `min` and `max` are required to normalize numeric `score`")
            normalized_score = (float(value.score) - numeric_min) / (numeric_max - numeric_min)
            return {"min": 0, "max": 1, "score": normalized_score}
        raise ValueError("no correct value of `score`")

    def _serialize_to_plain_dict(self, value: Any, exclude_positive_feedback: bool, normalize: bool = False) -> Any:
        """Recursively convert rubrics/metrics into plain python containers."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, BaseMetricType):
            if exclude_positive_feedback and self._is_positive_assessment(value):
                return None
            payload = {
                **value.model_dump(),
                "criterion": value.criterion,
                "description": "",
                # These are ClassVars on metric types, but attaching them here keeps
                # downstream rendering and normalization format-independent.
                "scale": getattr(value, "scale", None),
                "min": getattr(value, "min", None),
                "max": getattr(value, "max", None),
            }
            if normalize:
                payload.update(self._normalize(value))
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
                    converted["criterion"] = converted.get("criterion") or self._format_criterion_key(key)
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

    def _sanitize_metric_key(self, key: str) -> str:
        """Sanitize dynamic metric keys for dot-path flattening."""
        sanitized = re.sub(r"[^a-zA-Z0-9_./ -]", "_", key).strip()
        return sanitized[:240] if sanitized else "judge.metric"

    @staticmethod
    def _format_criterion_key(key: str) -> str:
        """Convert snake_case keys into a human-friendly label."""
        return key.replace("_", " ").strip()

    @staticmethod
    def _assessment_value(assessment: Any, key: str) -> Any:
        """Read assessment values from object or dict representations."""
        return assessment.get(key) if isinstance(assessment, dict) else getattr(assessment, key, None)

    @staticmethod
    def _is_assessment_dict(value: Any) -> bool:
        """Return true if a dict has assessment-shaped keys."""
        return isinstance(value, dict) and "score" in value and "feedback" in value

    def _is_metric_assessment_map(self, value: Any) -> bool:
        """Return true if all values in a dict look like assessments."""
        return isinstance(value, dict) and bool(value) and all(self._is_assessment_dict(item) for item in value.values())

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Safely convert primitive values to float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _escape_md(value: Any) -> str:
        """Escape markdown cell text for table rendering."""
        return "" if value is None else str(value).replace("|", "\\|").replace("\n", "<br>")

    @staticmethod
    def _path_column_sort_key(column_name: str) -> int:
        """Sort `path_#` columns by numeric suffix."""
        suffix = column_name.split("_", 1)[1] if "_" in column_name else ""
        return int(suffix) if suffix.isdigit() else 10**9

    @staticmethod
    def _require_pandas(caller: str):
        """Import pandas and raise a clear error when unavailable."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(f"`pandas` is required for `{caller}()`. Install it with `pip install pandas`.") from exc
        return pd

    @classmethod
    def _deep_merge_results(cls, left: Any, right: Any) -> Any:
        """Recursively merge two nested structures, preferring right-side leaf values on conflicts."""
        if isinstance(left, dict) and isinstance(right, dict):
            merged = dict(left)
            for key, right_value in right.items():
                if key in merged:
                    merged[key] = cls._deep_merge_results(merged[key], right_value)
                else:
                    merged[key] = right_value
            return merged
        return right

    # Backward-compatible private aliases for internal callers that may use legacy names.
    def _combine_feedback_markdown(self, exclude_positive_feedback: bool = False, md_header_level: int = 1) -> str:
        """Alias for `_render_markdown_feedback_tree`."""
        return self._render_markdown_feedback_tree(exclude_positive_feedback, md_header_level)

    def _combine_feedback_markdown_flattened(self, exclude_positive_feedback: bool = False) -> str:
        """Alias for `_render_markdown_feedback_flat`."""
        return self._render_markdown_feedback_flat(exclude_positive_feedback)

    def _flatten_metrics(self, payload: Any, prefix: str = "judge") -> dict[str, float | str]:
        """Alias for `_flatten_scores`."""
        return self._flatten_scores(payload, prefix)

    def _to_plain_dict(self, value: Any, exclude_positive_feedback: bool, normalize: bool = False) -> Any:
        """Alias for `_serialize_to_plain_dict`."""
        return self._serialize_to_plain_dict(value, exclude_positive_feedback, normalize)
