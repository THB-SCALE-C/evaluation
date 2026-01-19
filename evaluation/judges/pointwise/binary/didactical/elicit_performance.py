from evaluation.types.assessment_types import BinaryAssessment
from evaluation.judges.pointwise.binary.base import  BaseJudge
import dspy

class ElicitPerformanceJudge(BaseJudge):
    """Assess the provided `slide` (that is a practice task) text and the following metrics. You task is to evaluate the first event of Gagné's 9 events of instruction: 'elicit performance'.
    It is very important to be critical and exact because your assessment is required for important decisions to make.
    Avoid false optimism.
    """

    learning_objective:str = dspy.InputField(desc="The learning objective; Only use it to answer `alignment_learning_objective`.")

    previous_content:str = dspy.InputField(desc="The content which the learner has already gone through in the unit; Only use it to answer `alignment_previous_content`.")

    ###############################################################################################################
    # OUTPUT FIELDS
    ###############################################################################################################


    cognitive_demand:BinaryAssessment = dspy.OutputField(desc="Does the practice task require a level of reasoning that matches the learning objective (beyond simple recall if the objective implies analysis or decision-making)?")

    alignment_learning_objective:BinaryAssessment = dspy.OutputField(desc="Does successful completion of the practice task demonstrate achievement of the stated learning objective?")
    
    alignment_previous_content:BinaryAssessment = dspy.OutputField(desc="Does the hook clearly relate to the content?")

    authenticity:BinaryAssessment = dspy.OutputField(desc="Does the practice task describe a realistic workplace scenario the learner could plausibly face in their job context?")

    practice_load:BinaryAssessment = dspy.OutputField(desc="Are the number and length of practice tasks appropriate to be completed within a short micro-learning unit (e.g. ≤3 items, short responses)?")

    practice_load:BinaryAssessment = dspy.OutputField(desc="Can the correct answer and its discrimination from the distractors be found in `previous_content`?")
