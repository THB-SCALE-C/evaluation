from typing import ClassVar, Tuple
import re

from creator.dspy_components import Text
from evaluation.rubrics import BaseRuleRubric

class TextRuleBased(BaseRuleRubric[Text]):
    metric_name: ClassVar = "text"
    metric_type: ClassVar = "rule_based"
    required_slide_type: ClassVar = "text"

    def check_text_has_any_character(self, data: Text) -> Tuple[bool, str]:
        text = getattr(data, "text", None)
        has_content = bool(text and text.strip())
        return has_content, "`text` has content" if has_content else "`text` is empty"

    def check_text_max_length(self, data: Text) -> Tuple[bool, str]:
        text = getattr(data, "text", None)
        if text is None:
            return False, "`text` missing"
        is_valid = len(text) <= 1000
        return (
            is_valid,
            f"`text` length is {len(text)} (max 1000)" if not is_valid else "`text` length is within limit",
        )

    def check_title_is_not_start_of_text(self, data: Text) -> Tuple[bool, str]:
        title = getattr(data, "title", None)
        text = getattr(data, "text", None)
        if not title:
            return False, "`title` missing"
        if text is None:
            return False, "`text` missing"
        text_start = text.lstrip()
        starts_with_title = text_start.casefold().startswith(title.strip().casefold())
        return (
            not starts_with_title,
            "`text` starts with `title`" if starts_with_title else "`text` does not start with `title`",
        )

    def check_text_has_no_html_h1(self, data: Text) -> Tuple[bool, str]:
        text = getattr(data, "text", None)
        if text is None:
            return False, "`text` missing"
        has_h1 = bool(re.search(r"</?\s*h1\b[^>]*>", text, flags=re.IGNORECASE))
        return (
            not has_h1,
            "`text` contains `<h1>` HTML" if has_h1 else "`text` has no `<h1>` HTML",
        )
