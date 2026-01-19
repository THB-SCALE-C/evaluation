import dspy

from evaluation.types.assessment_types import BinaryAssessment


class ExpectedSlide(dspy.Prediction):
    user_instruction: str
    cloze_text: str
    title: str


class ClozeTextJudge(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.evaluations = dict()

    def forward(self, slide: ExpectedSlide):
        has_user_instruction = bool(slide.user_instruction)
        self.evaluations["missing_user_instructions"] = BinaryAssessment(
            score="yes" if has_user_instruction else "no",
            feedback="`user_instruction` present"
            if has_user_instruction
            else "`user_instruction` missing",
        )
        has_cloze_text = bool(slide.cloze_text)
        self.evaluations["missing_cloze_text"] = BinaryAssessment(
            score="yes" if has_cloze_text else "no",
            feedback="`cloze_text` present"
            if has_cloze_text
            else "`cloze_text` missing",
        )
        has_title = bool(slide.title)
        self.evaluations["missing_slide_title"] = BinaryAssessment(
            score="yes" if has_title else "no",
            feedback="`title` present" if has_title else "`title` missing",
        )

        if slide.cloze_text:
            is_even = (slide.cloze_text.count("*") % 2 == 0)
            self.evaluations["valid_word_count"] = BinaryAssessment(
                score="yes" if is_even else "no",
                feedback=f"`cloze_text` has an {"even" if is_even else "uneven"} number of `*`",
            )

            separators = ("", ", ", " and ", " or ")
            words_next_to_each_other = False
            i = 0
            text = slide.cloze_text
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

            self.evaluations["correct_blank_distance"] = BinaryAssessment(
                score="no" if words_next_to_each_other else "yes",
                feedback="`cloze_text` has adjacent blanks"
                if words_next_to_each_other
                else "`cloze_text` has no adjacent blanks",
            )

        return dspy.Prediction(**self.evaluations)
