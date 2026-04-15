from typing import ClassVar
from pydantic import Field
from evaluation.rubrics.base import BaseRubric
from evaluation.rubrics.base_didactical import BaseDidacticalRubric
from evaluation.types.assessment_types import BinaryMetricType


class ProvideGuidanceMetric(BaseDidacticalRubric):
    metric_name: ClassVar[str] = "provide_guidance"
    metric_type:ClassVar = "didactical"

    presence_of_guidance_elements: BinaryMetricType = Field(
        description="Does the unit include at least one explicit guidance element (e.g., tip, checklist, hint, rule)?")
    guidance_linked_to_relevant_content: BinaryMetricType = Field(
        description="Is the guidance clearly linked to specific concepts or content presented immediately before or after it?")
    scaffolding_without_full_solution: BinaryMetricType = Field(
        description="Does the guidance support decision-making without explicitly stating the correct answer?")
    microlearning_suitability: BinaryMetricType = Field(
        description="Is the guidance brief (e.g., <=3 bullet points or <=2 sentences) and easy to process quickly?")
    real_world_applicability: BinaryMetricType = Field(
        description="Does the guidance reference realistic workplace situations, tools, or decisions relevant to the learner context?")
    alignment_learning_objective: BinaryMetricType = Field(
        description="Does the guidance clearly help learners achieve the stated learning objective?")
    alignment_content: BinaryMetricType = Field(
        description="Is the guidance semantically consistent with and derived from the instructional content?")
