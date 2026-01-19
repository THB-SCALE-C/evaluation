from evaluation.types.assessment_types import BinaryAssessment
from evaluation.judges.pointwise.binary.base import  BaseJudge
import dspy

class ClozeTestJudge(BaseJudge):
    """Assess the provided `slide` which is a cloze styled test with the following metrics.
    It is very important to be critical and exact because your assessment is required for important decisions to make.
    Avoid false optimism.
    """
    ###############################################################################################################
    # OUTPUT FIELDS
    ###############################################################################################################


    distractor_difficulty:BinaryAssessment = dspy.OutputField(desc="Does every blanked word (marked by `*`) has at least one similar word among all other marked words to make up a challenging exercise?")

    user_instruction_do_not_name_asterisk:BinaryAssessment = dspy.OutputField(desc="Whether the `user_instruction` does NOT name in any way that blanked words are marked by asterisks.")
