from typing import ClassVar, Tuple

from creator.dspy_components import SingleChoice
from evaluation.metrics import BaseRuleMetric


class SingleChoiceRuleBased(BaseRuleMetric[SingleChoice]):
    metric_name: ClassVar = "single_choice"
    metric_type: ClassVar = "rule_based"
    required_slide_type: ClassVar = "single_choice"

    def _question_stats(self, data: SingleChoice) -> dict[str, int | bool]:
        questions = getattr(data, "questions", None) or []
        cached = getattr(self, "_cached_question_stats", None)
        if cached and cached[0] == id(questions):
            return cached[1]

        question_count = len(questions)
        empty_question_text_count = 0
        empty_correct_answer_count = 0
        question_with_invalid_wrong_answer_count = 0
        question_with_duplicate_options_count = 0

        for q in questions:
            question_text = getattr(q, "question", "") or ""
            correct_answer = getattr(q, "correct_answer", "") or ""
            wrong_answers = getattr(q, "wrong_answers", None) or []

            if not question_text.strip():
                empty_question_text_count += 1
            if not correct_answer.strip():
                empty_correct_answer_count += 1

            wrong_answer_values = [a for a in wrong_answers if isinstance(a, str)]
            has_empty_wrong_answer = any(not a.strip() for a in wrong_answer_values)
            wrong_count = len(wrong_answer_values)
            if has_empty_wrong_answer or wrong_count < 1 or wrong_count > 3:
                question_with_invalid_wrong_answer_count += 1

            normalized_options = [correct_answer.strip().casefold()] + [
                a.strip().casefold() for a in wrong_answer_values
            ]
            normalized_options = [o for o in normalized_options if o]
            if len(normalized_options) != len(set(normalized_options)):
                question_with_duplicate_options_count += 1

        stats: dict[str, int | bool] = {
            "question_count": question_count,
            "empty_question_text_count": empty_question_text_count,
            "empty_correct_answer_count": empty_correct_answer_count,
            "question_with_invalid_wrong_answer_count": question_with_invalid_wrong_answer_count,
            "question_with_duplicate_options_count": question_with_duplicate_options_count,
            "has_questions": question_count > 0,
        }
        self._cached_question_stats = (id(questions), stats)
        return stats

    def check_has_slide_title(self, data: SingleChoice) -> Tuple[bool, str]:
        has_title = bool(getattr(data, "title", "").strip())
        return has_title, "`title` present" if has_title else "`title` missing"

    def check_appropriate_title_length(self, data: SingleChoice) -> Tuple[bool, str]:
        title = getattr(data, "title", None)
        if not title:
            return False, "`title` missing"
        title_too_long = len(title.strip()) > 50
        return (not title_too_long), f"`title` has {'not ' if title_too_long else ''}appropriate length"

    def check_has_tip(self, data: SingleChoice) -> Tuple[bool, str]:
        has_tip = bool(getattr(data, "tip", "").strip())
        return has_tip, "`tip` present" if has_tip else "`tip` missing"

    def check_has_positive_feedback(self, data: SingleChoice) -> Tuple[bool, str]:
        has_positive_feedback = bool(getattr(data, "positive_feedback", "").strip())
        return (
            has_positive_feedback,
            "`positive_feedback` present" if has_positive_feedback else "`positive_feedback` missing",
        )

    def check_has_negative_feedback(self, data: SingleChoice) -> Tuple[bool, str]:
        has_negative_feedback = bool(getattr(data, "negative_feedback", "").strip())
        return (
            has_negative_feedback,
            "`negative_feedback` present" if has_negative_feedback else "`negative_feedback` missing",
        )

    def check_has_questions(self, data: SingleChoice) -> Tuple[bool, str]:
        has_questions = bool(self._question_stats(data)["has_questions"])
        return has_questions, "`questions` present" if has_questions else "`questions` missing"

    def check_questions_count(self, data: SingleChoice) -> Tuple[bool, str]:
        question_count = int(self._question_stats(data)["question_count"])
        too_many = question_count > 5
        if too_many:
            return False, f"`questions` has too many items ({question_count}), expected 2-4"
        return True, f"`questions` has appropriate number of items ({question_count})"

    def check_questions_have_text(self, data: SingleChoice) -> Tuple[bool, str]:
        missing_count = int(self._question_stats(data)["empty_question_text_count"])
        if missing_count:
            return False, f"`questions` has {missing_count} item(s) with missing `question` text"
        return True, "all `questions` have `question` text"

    def check_questions_have_correct_answers(self, data: SingleChoice) -> Tuple[bool, str]:
        missing_count = int(self._question_stats(data)["empty_correct_answer_count"])
        if missing_count:
            return False, f"`questions` has {missing_count} item(s) with missing `correct_answer`"
        return True, "all `questions` have `correct_answer`"
    
    def check_correct_answers_not_longest(self, data: SingleChoice) -> Tuple[bool, str]:
        for q in data.questions:
            if any(len(w)<len(q.correct_answer) for w in q.wrong_answers):
                return False, f"At least one question has a correct answer that is longer than any other wrong answer."
        return True, "all `correct_answers` are shorter than any wrong answer."

    def check_questions_have_valid_wrong_answers(self, data: SingleChoice) -> Tuple[bool, str]:
        invalid_count = int(self._question_stats(data)["question_with_invalid_wrong_answer_count"])
        if invalid_count:
            return (
                False,
                f"`questions` has {invalid_count} item(s) with invalid `wrong_answers` (requires 1-3 non-empty items)",
            )
        return True, "all `questions` have valid `wrong_answers`"

    def check_questions_have_unique_options(self, data: SingleChoice) -> Tuple[bool, str]:
        duplicate_count = int(self._question_stats(data)["question_with_duplicate_options_count"])
        if duplicate_count:
            return (
                False,
                f"`questions` has {duplicate_count} item(s) where answer options are duplicated",
            )
        return True, "all `questions` have unique answer options"
