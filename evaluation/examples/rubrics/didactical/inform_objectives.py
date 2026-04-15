from typing import ClassVar
from pydantic import Field
from evaluation.rubrics.base import BaseRubric
from evaluation.rubrics.base_didactical import BaseDidacticalRubric
from evaluation.types.assessment_types import BinaryMetricType


class InformObjectivesMetric(BaseDidacticalRubric):
    metric_name: ClassVar[str] = "inform_objectives"
    metric_type:ClassVar = "didactical"

    explicit_objective_statement: BinaryMetricType = Field(
        description="Is the learning objective clearly stated?")
    single_focus_objective: BinaryMetricType = Field(
        description="Does the learning objective present exactly one well-defined, specific learning objective rather than several loosely related goals?")
    actionable_formulation: BinaryMetricType = Field(
        description="Is the learning objective actionable: it involves an action to perform, a behavior to adapt, or a skill to learn?")
    microlearning_suitability: BinaryMetricType = Field(
        description="Can the learning objective be achieved within 10 minutes of training or 10 micro-learning slides?")
    workplace_relevance: BinaryMetricType = Field(
        description="Is the learning objective relevant in the learner's workplace context?")
    alignment_assessment: BinaryMetricType = Field(
        description="Do the assessment items directly assess whether the learning objective was learned?")
    alignment_content: BinaryMetricType = Field(
        description="Does the content directly provide information in accordance with the learning objective?")
