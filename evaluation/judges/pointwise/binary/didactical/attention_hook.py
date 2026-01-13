from evaluation.types.assessment_types import BinaryAssessment
from evaluation.judges.pointwise.binary.base import  BaseJudge
import dspy

class AttentionHookJudge(BaseJudge):
    """Assess the provided `slide` text and the following metrics. You task is to evaluate the first event of Gagné's 9 events of instruction: 'Gain Attention'.
    It is very important to be critical and exact because you assessment is required for important decisions to make.
    Avoid false optimism.
    """

    learning_objective:str = dspy.InputField(desc="The learning objective; Only use it to answer `alignment_learning_objective`.")

    further_content:str = dspy.InputField(desc="The content which appears later in the unit; Only use it to answer `alignment_further_content`.")

    ###############################################################################################################
    # OUTPUT FIELDS
    ###############################################################################################################

    presence_of_hook:BinaryAssessment = dspy.OutputField(desc="Is the hook presented as a realistic cyber security scenario or surprising cyber security fact that grabs attention while clearly connected to the overall topic?")

    salience:BinaryAssessment = dspy.OutputField(desc="Is the hook surprising, emotionally salient, or personally relevant?")

    personal_salience:BinaryAssessment = dspy.OutputField(desc="Does the hook address the reader personally or involve a personalized scenario?")

    conciseness:BinaryAssessment = dspy.OutputField(desc="Is the hook shorter than or exactly two sentences? And if 'yes', are the sentences short and comprehensive?")

    alignment_learning_objective:BinaryAssessment = dspy.OutputField(desc="Does the hook clearly relate to the learning objectives rather than introducing unrelated fun facts?")
    
    alignment_further_content:BinaryAssessment = dspy.OutputField(desc="Does the hook clearly relate to the content?")