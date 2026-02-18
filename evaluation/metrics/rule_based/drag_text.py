from functools import cached_property
from typing import ClassVar
from pydantic import computed_field
from evaluation.metrics.rule_based.base import BaseRuleMetric
from evaluation.types.assessment_types import BinaryAssessment


class DragTextRuleBased(BaseRuleMetric):
    metric_name:ClassVar = "drag_text"
    metric_type:ClassVar = "rule_based"
    required_slide_type:ClassVar = "cloze_text"
    title: str | None = None
    cloze_text: str | None = None
    user_instructions: str | None = None

    @cached_property
    def _cloze_stats(self) -> dict[str, int | float | bool]:
        if not self.cloze_text:
            return {}
        star_count = self.cloze_text.count("*")
        is_even = star_count % 2 == 0
        blank_words_count = star_count / 2
        return {
            "total_word_count": len(self.cloze_text.split(" ")),
            "is_even": is_even,
            "blank_words_count": blank_words_count,
            "has_adjacent_words": has_adjacent_blanks(self.cloze_text),
            "text_contains_double_line_break": "\n\n" in self.cloze_text,
            "has_too_long_blanks": has_too_long_blanks(self.cloze_text) if is_even else False,
        }
    ### derived fields ###
    @computed_field
    @property
    def has_user_instructions(self) -> BinaryAssessment:
        has_user_instruction = bool(self.user_instructions)
        return BinaryAssessment(
            score="yes" if has_user_instruction else "no",
            feedback="`user_instruction` present"
            if has_user_instruction
            else "`user_instruction` missing",
        )

    @computed_field
    @property
    def has_cloze_text(self) -> BinaryAssessment:
        has_cloze_text = bool(self.cloze_text)
        return BinaryAssessment(
            score="yes" if has_cloze_text else "no",
            feedback="`cloze_text` present" if has_cloze_text else "`cloze_text` missing",
        )

    @computed_field
    @property
    def has_slide_title(self) -> BinaryAssessment:
        has_title = bool(self.title)
        return BinaryAssessment(
            score="yes" if has_title else "no",
            feedback="`title` present" if has_title else "`title` missing",
        )
    
    @computed_field
    @property
    def appropriate_title_length(self) -> BinaryAssessment | None:
        if not self.title:
            return None
        title_too_long = len(self.title) > 50
        return BinaryAssessment(
            score="yes" if not title_too_long else "no",
            feedback=f"`title` has {'not' if title_too_long else ''} appropriate length",
        )

    @computed_field
    @property
    def valid_word_count(self) -> BinaryAssessment | None:
        if not self.cloze_text:
            return None
        is_even = bool(self._cloze_stats["is_even"])
        return BinaryAssessment(
            score="yes" if is_even else "no",
            feedback=f"`cloze_text` has an {'even' if is_even else 'uneven'} number of `*`",
        )

    @computed_field
    @property
    def correct_blank_distance(self) -> BinaryAssessment | None:
        if not self.cloze_text:
            return None
        has_adjacent_words = bool(self._cloze_stats["has_adjacent_words"])
        return BinaryAssessment(
            score="no" if has_adjacent_words else "yes",
            feedback="`cloze_text` has adjacent blanks"
            if has_adjacent_words
            else "`cloze_text` has no adjacent blanks",
        )

    @computed_field
    @property
    def cloze_text_appropriate_length(self) -> BinaryAssessment | None:
        if not self.cloze_text:
            return None
        total_word_count = int(self._cloze_stats["total_word_count"])
        not_appropriate_length = total_word_count > 85 or total_word_count < 65
        return BinaryAssessment(
            score="no" if not_appropriate_length else "yes",
            feedback=f"`cloze_text` has not appropriate length, having {total_word_count} words."
            if not_appropriate_length
            else "`cloze_text` has appropriate length",
        )

    @computed_field
    @property
    def has_no_double_linebreaks(self) -> BinaryAssessment | None:
        if not self.cloze_text:
            return None
        text_contains_double_line_break = bool(
            self._cloze_stats["text_contains_double_line_break"])
        return BinaryAssessment(
            score="no" if text_contains_double_line_break else "yes",
            feedback="`cloze_text` wastes space with double linebreaks"
            if text_contains_double_line_break
            else "`cloze_text` has not double linebreaks",
        )

    @computed_field
    @property
    def enough_blanked_words(self) -> BinaryAssessment | None:
        if not self.cloze_text:
            return None
        too_few_blank_words = float(self._cloze_stats["blank_words_count"]) < 4
        return BinaryAssessment(
            score="no" if too_few_blank_words else "yes",
            feedback="`cloze_text` has not enough blanked words"
            if too_few_blank_words
            else "`cloze_text` has enough blanked words",
        )

    @computed_field
    @property
    def not_too_many_blanked_words(self) -> BinaryAssessment | None:
        if not self.cloze_text:
            return None
        too_many_blank_words = float(
            self._cloze_stats["blank_words_count"]) > 6
        return BinaryAssessment(
            score="no" if too_many_blank_words else "yes",
            feedback="`cloze_text` has too many blanked words"
            if too_many_blank_words
            else "`cloze_text` has not too many blanked words",
        )

    @computed_field
    @property
    def has_too_long_blanks(self) -> BinaryAssessment | None:
        if not self.cloze_text:
            return None
        if not bool(self._cloze_stats["is_even"]):
            return None
        too_long = bool(self._cloze_stats["has_too_long_blanks"])
        return BinaryAssessment(
            score="no" if too_long else "yes",
            feedback="`cloze_text` has too long blanked words"
            if too_long
            else "`cloze_text` has appropriate long blanked words",
        )


def has_too_long_blanks(text: str) -> bool:
    blank_words = [s for i, s in enumerate(text.split("*")) if i % 2 != 0]
    return any(len(s) > 35 for s in blank_words)


def has_adjacent_blanks(text: str) -> bool:
    separators = ("", ", ", " and ", " or ")
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
