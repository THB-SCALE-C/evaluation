from typing import ClassVar
from pydantic import Field
from evaluation.rubrics.base import BaseRubric
from evaluation.rubrics.base_didactical import BaseDidacticalRubric
from evaluation.types.assessment_types import BinaryMetricType


class ElicitPerformanceMetric(BaseDidacticalRubric):
    metric_name: ClassVar[str] = "elicit_performance"
    metric_type:ClassVar = "didactical"

    presence_of_practice: BinaryMetricType = Field(
        description="Does the slide contain at least one explicit practice or performance task requiring an active learner response?")
    cognitive_demand: BinaryMetricType = Field(
        description="Does the practice task require a level of reasoning that matches the learning objective (beyond simple recall if the objective implies analysis or decision-making)?")
    authenticity: BinaryMetricType = Field(
        description="Does the practice task describe a realistic workplace scenario the learner could plausibly face in their job context?")
    practice_load_suitability: BinaryMetricType = Field(
        description="Are the number and length of practice tasks appropriate to be completed within a short micro-learning unit (e.g. <=3 items, short responses)?")
    alignment_learning_objective: BinaryMetricType = Field(
        description="Does successful completion of the practice task demonstrate achievement of the stated learning objective?")
    feasibility_given_content: BinaryMetricType = Field(
        description="Can the correct answer and its discrimination from the distractors be found in `previous_content`?")
