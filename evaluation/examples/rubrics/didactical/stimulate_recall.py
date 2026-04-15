from typing import ClassVar
from pydantic import Field
from evaluation.rubrics.base import BaseRubric
from evaluation.rubrics.base_didactical import BaseDidacticalRubric
from evaluation.types.assessment_types import BinaryMetricType


class StimulateRecallMetric(BaseDidacticalRubric):
    metric_name: ClassVar[str] = "stimulate_recall"
    metric_type:ClassVar = "didactical"

    presence_of_activation_stimulus: BinaryMetricType = Field(
        description="Does the unit begin with an explicit question or reflection prompt that asks the learner to recall prior knowledge or experience?")
    depth_of_cognitive_activation: BinaryMetricType = Field(
        description="Does the activation stimulus require explanation, comparison, or justification rather than simple recall or definition?")
    microlearning_suitability: BinaryMetricType = Field(
        description="Is the activation stimulus brief and simple enough to be completed within approximately one minute?")
    experience_space_fit: BinaryMetricType = Field(
        description="Does the activation stimulus reference a concrete, realistic workplace or organizational situation?")
    alignment_learning_objective: BinaryMetricType = Field(
        description="Is the activation stimulus semantically aligned with the stated learning objective?")
    alignment_content: BinaryMetricType = Field(
        description="Does the activation stimulus introduce concepts or situations that are directly addressed in the immediately following content?")
