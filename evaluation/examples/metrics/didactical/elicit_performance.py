from typing import ClassVar
from pydantic import Field
from evaluation.metrics.base import BaseMetric
from evaluation.metrics.base_didactical import BaseDidacticalMetric
from evaluation.types.assessment_types import BinaryAssessment


class ElicitPerformanceMetric(BaseDidacticalMetric):
    metric_name: ClassVar[str] = "elicit_performance"
    metric_type:ClassVar = "didactical"

    presence_of_practice: BinaryAssessment = Field(
        description="Does the slide contain at least one explicit practice or performance task requiring an active learner response?")
    cognitive_demand: BinaryAssessment = Field(
        description="Does the practice task require a level of reasoning that matches the learning objective (beyond simple recall if the objective implies analysis or decision-making)?")
    authenticity: BinaryAssessment = Field(
        description="Does the practice task describe a realistic workplace scenario the learner could plausibly face in their job context?")
    practice_load_suitability: BinaryAssessment = Field(
        description="Are the number and length of practice tasks appropriate to be completed within a short micro-learning unit (e.g. <=3 items, short responses)?")
    alignment_learning_objective: BinaryAssessment = Field(
        description="Does successful completion of the practice task demonstrate achievement of the stated learning objective?")
    feasibility_given_content: BinaryAssessment = Field(
        description="Can the correct answer and its discrimination from the distractors be found in `previous_content`?")
