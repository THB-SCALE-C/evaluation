import dspy

from evaluation.types.assessment_types import BinaryAssessment


class ExpectedSlide(dspy.Prediction):
    user_instruction: str
    cloze_text: str
    title: str


class ClozeTextRuleChecker(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.evaluations = dict()

    def forward(self, slide: ExpectedSlide):
        has_user_instruction = bool(slide.user_instruction)
        self.evaluations["has_user_instructions"] = BinaryAssessment(
            score="yes" if has_user_instruction else "no",
            feedback="`user_instruction` present"
            if has_user_instruction
            else "`user_instruction` missing",
        )
        has_cloze_text = bool(slide.cloze_text)
        self.evaluations["has_cloze_text"] = BinaryAssessment(
            score="yes" if has_cloze_text else "no",
            feedback="`cloze_text` present"
            if has_cloze_text
            else "`cloze_text` missing",
        )
        has_title = bool(slide.title)
        self.evaluations["has_slide_title"] = BinaryAssessment(
            score="yes" if has_title else "no",
            feedback="`title` present" if has_title else "`title` missing",
        )

        if slide.title:
            title_too_long = len(slide.title)>50
            self.evaluations["appropriate_title_length"] = BinaryAssessment(
                score="yes" if not title_too_long else "no",
                feedback=f"`title` has {"not" if title_too_long else ""} appropriate length",
            )

        if slide.cloze_text:
            total_word_count = len(slide.cloze_text.split(" "))
            is_even = (slide.cloze_text.count("*") % 2 == 0)
            self.evaluations["valid_word_count"] = BinaryAssessment(
                score="yes" if is_even else "no",
                feedback=f"`cloze_text` has an {"even" if is_even else "uneven"} number of `*`",
            )

            has_adjacent_words = has_adjacent_blanks(slide.cloze_text)
            self.evaluations["correct_blank_distance"] = BinaryAssessment(
                score="no" if has_adjacent_words else "yes",
                feedback="`cloze_text` has adjacent blanks"
                if has_adjacent_words
                else "`cloze_text` has no adjacent blanks",
            )

            text_too_long = len(slide.cloze_text) > 1000
            self.evaluations["cloze_text_not_too_long"] = BinaryAssessment(
                score="no" if text_too_long else "yes",
                feedback="`cloze_text` exceed 1000 characters"
                if text_too_long
                else "`cloze_text` does not exceed 1000 characters",
            )

            text_too_short = len(slide.cloze_text) < 200
            self.evaluations["cloze_text_not_too_short"] = BinaryAssessment(
                score="no" if text_too_short else "yes",
                feedback="`cloze_text` has less than 200 characters"
                if text_too_short
                else "`cloze_text` does exceed 200 characters",
            )

            text_contains_double_line_break = slide.cloze_text.count("\n\n")>0
            self.evaluations["has_no_double_linebreaks"] = BinaryAssessment(
                score="no" if text_contains_double_line_break else "yes",
                feedback="`cloze_text` wastes space with double linebreaks"
                if text_contains_double_line_break
                else "`cloze_text` has not double linebreaks",
            )

            blank_words_count = slide.cloze_text.count("*")/2
            too_many_blank_words = blank_words_count/total_word_count>0.33
            too_few_blank_words = blank_words_count/total_word_count<0.05

            self.evaluations["enough_blanked_words"] = BinaryAssessment(
                score="no" if too_few_blank_words else "yes",
                feedback="`cloze_text` has not enough blanked words"
                if too_few_blank_words
                else "`cloze_text` has enough blanked words",
            )

            self.evaluations["not_too_many_blanked_words"] = BinaryAssessment(
                score="no" if too_many_blank_words else "yes",
                feedback="`cloze_text` has too many blanked words"
                if too_many_blank_words
                else "`cloze_text` has not too many blanked words",
            )

        return dspy.Prediction(**self.evaluations)


def has_adjacent_blanks(text:str):
    separators = ("", ", ", " and ", " or ")
    words_next_to_each_other = False
    i = 0
    while True:
        start = text.find("*", i)
        if start == -1:
            break
        end = text.find("*", start + 1)
        if end == -1:
            break
        for sep in separators:
            if text.startswith(sep + "*", end + 1):
                next_start = end + 1 + len(sep)
                next_end = text.find("*", next_start + 1)
                if next_end != -1:
                    words_next_to_each_other = True
                    break
        if words_next_to_each_other:
            break
        i = end + 1
    return words_next_to_each_other