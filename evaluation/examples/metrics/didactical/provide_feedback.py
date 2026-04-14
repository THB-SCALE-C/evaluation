from typing import ClassVar
from pydantic import Field
from evaluation.metrics.base import BaseMetric
from evaluation.metrics.base_didactical import BaseDidacticalMetric
from evaluation.types.assessment_types import BinaryAssessment


class ProvideFeedbackMetric(BaseDidacticalMetric):
    metric_name: ClassVar[str] = "provide_feedback"
    metric_type:ClassVar = "didactical"

    presence_of_feedback: BinaryAssessment = Field(
        description="Does the learner receive feedback after submitting an answer?")
    correctness_of_verification_feedback: BinaryAssessment = Field(
        description="Is the learner response correctly marked as correct or incorrect based on the predefined solution?")
    presence_of_elaborated_feedback: BinaryAssessment = Field(
        description="Does the feedback include an explanation, hint, or reasoning instead of only correct/incorrect?")
    motivational_tone: BinaryAssessment = Field(
        description="Is the feedback phrased in a supportive, constructive tone?")
    brevity_and_clarity: BinaryAssessment = Field(
        description="Is the feedback clear and no longer than 2-3 sentences?")
    real_world_relevance: BinaryAssessment = Field(
        description="Does the feedback explain why this matters in a real workplace situation?")
    alignment_learning_objective: BinaryAssessment = Field(
        description="Does the feedback directly reinforce the stated learning objective?")
    alignment_content: BinaryAssessment = Field(
        description="Does the feedback reference concepts that were explained in the instructional content?")
    feedback_exceeds_task: BinaryAssessment = Field(
        description="Does the feedback provide information that is not explicitly part of the underlying task?")
