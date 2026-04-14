from typing import ClassVar, Tuple
from evaluation.metrics import BaseRuleMetric
from creator.dspy_components import ClozeTest

class DragTextRuleBased(BaseRuleMetric[ClozeTest]):
    metric_name:ClassVar = "drag_text"
    metric_type:ClassVar = "rule_based"
    required_slide_type:ClassVar = "drag_text"

    def _cloze_stats(self, data: ClozeTest) -> dict[str, int | float | bool]:
        text = getattr(data, "cloze_text", None)
        cached = getattr(self, "_cached_cloze_stats", None)
        if cached and cached[0] == text:
            return cached[1]
        if not text:
            stats: dict[str, int | float | bool] = {}
        else:
            star_count = text.count("*")
            is_even = star_count % 2 == 0
            stats = {
                "total_word_count": len(text.split(" ")),
                "is_even": is_even,
                "blank_words_count": star_count / 2,
                "has_adjacent_words": has_adjacent_blanks(text),
                "text_contains_double_line_break": "\n\n" in text,
                "has_too_long_blanks": has_too_long_blanks(text) if is_even else False,
            }
        self._cached_cloze_stats = (text, stats)
        return stats

    def check_has_user_instructions(self, data: ClozeTest) -> Tuple[bool, str]:
        has_user_instruction = bool(getattr(data, "user_instruction", "").strip())
        return (
            has_user_instruction,
            "`user_instruction` present" if has_user_instruction else "`user_instruction` missing",
        )

    def check_has_cloze_text(self, data: ClozeTest) -> Tuple[bool, str]:
        has_text = bool(getattr(data, "cloze_text", None))
        return has_text, "`cloze_text` present" if has_text else "`cloze_text` missing"

    def check_has_slide_title(self, data: ClozeTest) -> Tuple[bool, str]:
        has_title = bool(getattr(data, "title", None))
        return has_title, "`title` present" if has_title else "`title` missing"

    def check_appropriate_title_length(self, data: ClozeTest) -> Tuple[bool, str]:
        title = getattr(data, "title", None)
        if not title:
            return False, "`title` missing"
        title_too_long = len(title) > 50
        return (not title_too_long), f"`title` has {'not' if title_too_long else ''} appropriate length"

    def check_valid_word_count(self, data: ClozeTest) -> Tuple[bool, str]:
        text = getattr(data, "cloze_text", None)
        if not text:
            return False, "`cloze_text` missing"
        is_even = bool(self._cloze_stats(data)["is_even"])
        return is_even, f"`cloze_text` has an {'even' if is_even else 'uneven'} number of `*`"

    def check_correct_blank_distance(self, data: ClozeTest) -> Tuple[bool, str]:
        text = getattr(data, "cloze_text", None)
        if not text:
            return False, "`cloze_text` missing"
        has_adjacent_words = bool(self._cloze_stats(data)["has_adjacent_words"])
        return (
            not has_adjacent_words,
            "`cloze_text` has adjacent blanks"
            if has_adjacent_words
            else "`cloze_text` has no adjacent blanks",
        )

    def check_cloze_text_appropriate_length(self, data: ClozeTest) -> Tuple[bool, str]:
        text = getattr(data, "cloze_text", None)
        if not text:
            return False, "`cloze_text` missing"
        total_word_count = int(self._cloze_stats(data)["total_word_count"])
        not_appropriate_length = total_word_count > 85 or total_word_count < 65
        return (
            not not_appropriate_length,
            f"`cloze_text` has not appropriate length, having {total_word_count} words."
            if not_appropriate_length
            else "`cloze_text` has appropriate length",
        )

    def check_has_no_double_linebreaks(self, data: ClozeTest) -> Tuple[bool, str]:
        text = getattr(data, "cloze_text", None)
        if not text:
            return False, "`cloze_text` missing"
        text_contains_double_line_break = bool(self._cloze_stats(data)["text_contains_double_line_break"])
        return (
            not text_contains_double_line_break,
            "`cloze_text` wastes space with double linebreaks"
            if text_contains_double_line_break
            else "`cloze_text` has not double linebreaks",
        )

    def check_blank_count(self, data: ClozeTest) -> Tuple[bool, str]:
        text = getattr(data, "cloze_text", None)
        if not text:
            return False, "`cloze_text` missing"
        blank_words_count = float(self._cloze_stats(data)["blank_words_count"])
        too_few_blank_words = blank_words_count < 4
        too_many_blank_words = blank_words_count > 6
        if too_few_blank_words:
            return False, "`cloze_text` has not enough blanked words"
        if too_many_blank_words:
            return False, "`cloze_text` has too many blanked words"
        return True, "`cloze_text` has an appropriate amount of blanked words"


    def check_has_too_long_blanks(self, data: ClozeTest) -> Tuple[bool, str]:
        text = getattr(data, "cloze_text", None)
        if not text:
            return False, "`cloze_text` missing"
        is_even = bool(self._cloze_stats(data)["is_even"])
        if not is_even:
            return False, "`cloze_text` has an uneven number of `*`"
        too_long = bool(self._cloze_stats(data)["has_too_long_blanks"])
        return (
            not too_long,
            "`cloze_text` has too long blanked words"
            if too_long
            else "`cloze_text` has appropriate long blanked words",
        )

def has_too_long_blanks(text: str) -> bool:
    blank_words = [s for i, s in enumerate(text.split("*")) if i % 2 != 0]
    return any(len(s) > 35 for s in blank_words)


def has_adjacent_blanks(text: str) -> bool:
    separators = ("", ", ", " and ", " or ",)
    i = 0
    while True:
        start = text.find("*", i)
        if start == -1:
            return False
        end = text.find("*", start + 1)
        if end == -1:
            return False
        for sep in separators:
            if text.startswith(sep + "*", end + 1):
                next_start = end + 1 + len(sep)
                next_end = text.find("*", next_start + 1)
                if next_end != -1:
                    return True
        i = end + 1
