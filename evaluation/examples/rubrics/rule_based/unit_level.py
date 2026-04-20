from typing import ClassVar, Tuple
import re

from creator.dspy_components import Text, ClozeTest, SingleChoice
from evaluation.rubrics import BaseRuleRubric


class UnitConsistency(BaseRuleRubric[list[Text | ClozeTest | SingleChoice]]):
    metric_name: ClassVar = "unit_consistency"
    metric_type: ClassVar = "rule_based"
    required_slide_type: ClassVar = None

    def check_quantity_slides(self, data: list[Text | ClozeTest | SingleChoice]) -> Tuple[bool, str]:
        slides_len = len(data)

        if slides_len<5:
            return False, "too few slides."
        if slides_len>15:
            return False, "too many slides for micro-learning."
        return True, f"Having {slides_len} slides, the unit has an accepted amount of slides."

    def check_activity_proportionality(self, data: list[Text | ClozeTest | SingleChoice]) -> Tuple[bool, str]:
        slides_len = len(data)
        text_amount = len([s for s in data if s.slide_type == "text"])
        activity_amount = slides_len - text_amount

        if activity_amount<2:
            return False, "The unit has fewer than 2 activity slides which is too few."
        if (activity_amount / slides_len)>0.67:
            return False, "The unit has more than 2/3 activity slides which is too high."
        if (text_amount / slides_len)>0.8:
            return False, "Regarding the amount of slides, the unit has too few activities."
        return True, f"Unit shows a good proportion of activity and text slides."