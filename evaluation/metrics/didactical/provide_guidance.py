from typing import ClassVar
from pydantic import Field
from evaluation.metrics.base import BaseMetric
from evaluation.types.assessment_types import BinaryAssessment


class ProvideGuidanceMetric(BaseMetric):
    metric_name: ClassVar[str] = "provide_guidance"

    presence_of_guidance_elements: BinaryAssessment = Field(
        description="Does the unit include at least one explicit guidance element (e.g., tip, checklist, hint, rule)?")
    guidance_linked_to_relevant_content: BinaryAssessment = Field(
        description="Is the guidance clearly linked to specific concepts or content presented immediately before or after it?")
    scaffolding_without_full_solution: BinaryAssessment = Field(
        description="Does the guidance support decision-making without explicitly stating the correct answer?")
    microlearning_suitability: BinaryAssessment = Field(
        description="Is the guidance brief (e.g., <=3 bullet points or <=2 sentences) and easy to process quickly?")
    real_world_applicability: BinaryAssessment = Field(
        description="Does the guidance reference realistic workplace situations, tools, or decisions relevant to the learner context?")
    alignment_learning_objective: BinaryAssessment = Field(
        description="Does the guidance clearly help learners achieve the stated learning objective?")
    alignment_content: BinaryAssessment = Field(
        description="Is the guidance semantically consistent with and derived from the instructional content?")
