from typing import ClassVar
from pydantic import Field
from evaluation.rubrics.base import BaseRubric
from evaluation.rubrics.base_didactical import BaseDidacticalRubric
from evaluation.types.assessment_types import BinaryMetricType


class ProvideFeedbackMetric(BaseDidacticalRubric):
    metric_name: ClassVar[str] = "provide_feedback"
    metric_type:ClassVar = "didactical"

    presence_of_feedback: BinaryMetricType = Field(
        description="Does the learner receive feedback after submitting an answer?")
    correctness_of_verification_feedback: BinaryMetricType = Field(
        description="Is the learner response correctly marked as correct or incorrect based on the predefined solution?")
    presence_of_elaborated_feedback: BinaryMetricType = Field(
        description="Does the feedback include an explanation, hint, or reasoning instead of only correct/incorrect?")
    motivational_tone: BinaryMetricType = Field(
        description="Is the feedback phrased in a supportive, constructive tone?")
    brevity_and_clarity: BinaryMetricType = Field(
        description="Is the feedback clear and no longer than 2-3 sentences?")
    real_world_relevance: BinaryMetricType = Field(
        description="Does the feedback explain why this matters in a real workplace situation?")
    alignment_learning_objective: BinaryMetricType = Field(
        description="Does the feedback directly reinforce the stated learning objective?")
    alignment_content: BinaryMetricType = Field(
        description="Does the feedback reference concepts that were explained in the instructional content?")
    feedback_exceeds_task: BinaryMetricType = Field(
        description="Does the feedback provide information that is not explicitly part of the underlying task?")
